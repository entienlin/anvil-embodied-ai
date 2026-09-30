"""Open-loop replay of a pre-recorded action sequence through the real
action-processing pipeline (EEReferenceAndDelta -> EEBoundsAndSafety ->
InverseKinematicsRLStep -> AnvilOpenArmFollower.send_action), with NO
policy in the loop at all.

Why this exists (2026-09-29): ACT (trained via a completely separate
pipeline -- anvil-trainer, plain supervised loss, no RL, no SAC/HIL-SERL
code at all) produced the same wrong, "shaky/terrible" motion as the SAC
actor did. Two independently-trained models agreeing on the same wrong
motion is a strong signal the problem isn't either model's predictions --
it's something upstream (the recorded actions themselves) or downstream
(this env's action processing) that both models were consistently trained
against / deployed through.

This script isolates that: take the RAW recorded actions from one known-
good, successful human demo episode (ds[i]["action"] for a chosen episode
of demos_v2 -- these are literally what a human's real teleop produced,
proven successful at recording time) and replay them through the exact
same real-hardware action pipeline both policies go through, verbatim,
with no model, no camera, no state feedback deciding anything. If this
reproduces the same successful push, the pipeline is faithful and the
problem is in what the models learned to predict. If replay ALSO fails or
looks wrong, the deployment/action-processing pipeline itself is the bug,
and no amount of retraining either model would ever fix it.

Usage -- replay one specific saved episode:
    REPLAY_ACTIONS_NPZ=/path/to/episode.npz \
    ros2 run lerobot_control hilserl_replay_actions --config_path <path>

Usage -- "replay right away": replay the most recent successful episode in
a directory (e.g. the same one a just-finished hilserl_actor/hilserl_eval_
policy session was writing to via -e HILSERL_TRAJ_DIR=...):
    REPLAY_EPISODE_DIR=/workspace/outputs/some_session \
    ros2 run lerobot_control hilserl_replay_actions --config_path <path>
No extra recording step needed -- every actor/eval session already saves
one ep_*.npz per episode (save_actor_trajectory() in patches.py); this
just finds the newest one with reward>0 and replays its "action" array.

The npz must have one array, "action", shape (N, 3) -- exactly
lerobot_control.hilserl.bc_pretrain's/demos_v2's raw recorded action
convention (delta_x/delta_y/delta_z multipliers, NOT clipped to [-1,1] --
same as what a live human's raw teleop produces, so no special-casing is
needed relative to a real intervention). Both save_actor_trajectory()'s
episode files and a demos_v2-derived extraction (ds[i]["action"]) already
match this shape.
"""

import logging
import os

from lerobot_control.hilserl import robot_anvil_openarm  # noqa: F401 -- registration side effect
from lerobot_control.hilserl import teleop_anvil_openarm_leader  # noqa: F401 -- registration side effect


def _find_latest_successful_episode(episode_dir: str) -> str:
    """The "replay right away" path: every hilserl_actor/hilserl_eval_policy session
    already saves one ep_*.npz per episode (save_actor_trajectory() in patches.py),
    filenames timestamp-sorted, each with a real "reward" array -- no new recording
    infrastructure needed, just point REPLAY_EPISODE_DIR at the same directory
    -e HILSERL_TRAJ_DIR/--config_path pointed the session at and rerun this script."""
    import glob

    import numpy as np

    candidates = sorted(glob.glob(os.path.join(episode_dir, "ep_*.npz")))
    for path in reversed(candidates):  # most recent first
        z = np.load(path)
        if float(z["reward"].sum()) > 0:
            return path
    raise RuntimeError(f"No successful (reward>0) episode found in {episode_dir} ({len(candidates)} episodes checked).")


def run_action_replay(cfg) -> None:
    import time

    import numpy as np
    import torch
    from lerobot.rl.gym_manipulator import create_transition, make_processors, make_robot_env
    from lerobot.types import TransitionKey
    from lerobot.utils.robot_utils import precise_sleep

    npz_path = os.environ.get("REPLAY_ACTIONS_NPZ")
    episode_dir = os.environ.get("REPLAY_EPISODE_DIR")
    if npz_path:
        pass
    elif episode_dir:
        npz_path = _find_latest_successful_episode(episode_dir)
        print(f"[REPLAY] REPLAY_EPISODE_DIR set -- auto-selected the most recent successful "
              f"(reward>0) episode: {npz_path}", flush=True)
    else:
        raise RuntimeError(
            "Set either REPLAY_ACTIONS_NPZ (a specific episode's action array) or "
            "REPLAY_EPISODE_DIR (a directory of ep_*.npz files -- e.g. the same "
            "HILSERL_TRAJ_DIR an actor session just wrote to -- to auto-replay its "
            "latest successful episode). See this module's docstring."
        )
    actions = np.load(npz_path)["action"]
    print(f"[REPLAY] loaded {len(actions)} recorded actions from {npz_path}", flush=True)

    # How many leading steps to print full pipeline detail for -- HILSERL_REPLAY_DEBUG_STEPS=0
    # disables this and prints only the summary line every 20 steps. Defaults to the whole
    # episode: episodes here are short (~90-200 steps) and a real demo's motion doesn't
    # necessarily start at step 0 (idle lead-in before the human starts moving, as episode 0
    # of demos_v2 turned out to have for its first 15+ steps) -- a truncated debug window can
    # silently show only the idle segment and look uninformative for the wrong reason.
    debug_steps = int(os.environ.get("HILSERL_REPLAY_DEBUG_STEPS", str(len(actions))))

    logging.info("[REPLAY] make_env")
    env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.policy.device)

    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()
    transition = env_processor(create_transition(observation=obs, info=info))

    def _joint_pos(observation: dict) -> dict:
        return {k: round(float(v), 2) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")}

    print(f"[REPLAY] starting -- replaying {len(actions)} recorded steps verbatim, no policy involved", flush=True)
    if debug_steps:
        print(f"[REPLAY] joints BEFORE step 0: {_joint_pos(transition[TransitionKey.OBSERVATION])}", flush=True)

    # gym_manipulator.py's own control_loop() paces every real step to dt=1/fps
    # (precise_sleep(max(dt - elapsed, 0.0)) at the end of each tick) -- this loop had
    # no such pacing at all, so it ran flat-out at whatever speed the pipeline allowed
    # (no camera-capture wait, no policy-inference delay) instead of at the real control
    # rate. Result (found 2026-09-30): a ~13s real episode replayed in a couple of
    # seconds -- correct actions, wrong speed, looking "extremely short"/rushed. This
    # also matters for physical correctness, not just appearances: EEReferenceAndDelta's
    # delta is a per-TICK step size, so replaying the same deltas faster than they were
    # recorded doesn't just look fast, it commands a faster real end-effector velocity
    # than the demo actually had.
    dt = 1.0 / cfg.env.fps
    reward_sum = 0.0
    step_i = 0
    try:
        for step_i, raw_action in enumerate(actions):
            step_start_time = time.perf_counter()
            action = torch.as_tensor(raw_action, dtype=torch.float32)

            # Inlined from lerobot.rl.gym_manipulator.step_env_and_process_transition() --
            # only difference from calling it directly is the debug print of
            # `processed_action` (the FINAL command about to reach env.step() ->
            # AnvilOpenArmFollower.send_action()) and the observed joint state right
            # before/after, so a "nothing moved" run shows exactly where the signal
            # disappeared: computed-as-near-zero upstream (IK/enabled/EE-bounds), or
            # computed correctly but not reaching the real motors.
            transition[TransitionKey.ACTION] = action
            joints_before = _joint_pos(transition[TransitionKey.OBSERVATION])
            transition[TransitionKey.OBSERVATION] = (
                env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
            )
            processed_action_transition = action_processor(transition)
            processed_action = processed_action_transition[TransitionKey.ACTION]

            if step_i < debug_steps:
                # MapDeltaActionToRobotActionStep's own "enabled" test (position_magnitude >
                # noise_threshold=1e-3), recomputed here since "enabled" itself is consumed
                # internally by EEReferenceAndDelta and not visible on the transition afterward.
                position_magnitude = float(np.linalg.norm(raw_action[:3]))
                print(
                    f"[REPLAY-DEBUG] step {step_i}: raw_action={np.round(raw_action, 4).tolist()} "
                    f"position_magnitude={position_magnitude:.4f} (enabled if > 1e-3) "
                    f"processed_action(-> env.step)={processed_action} joints_before={joints_before}",
                    flush=True,
                )

            step_obs, step_reward, terminated, truncated, step_info = env.step(processed_action)
            joints_after = _joint_pos(step_obs)
            if step_i < debug_steps:
                print(f"[REPLAY-DEBUG] step {step_i}: joints_after ={joints_after}", flush=True)

            reward = step_reward + processed_action_transition[TransitionKey.REWARD]
            terminated = terminated or processed_action_transition[TransitionKey.DONE]
            truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
            complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
            new_info = processed_action_transition[TransitionKey.INFO].copy()
            new_info.update(step_info)
            transition = create_transition(
                observation=step_obs, action=processed_action, reward=reward, done=terminated,
                truncated=truncated, info=new_info, complementary_data=complementary_data,
            )
            transition = env_processor(transition)

            reward = float(transition[TransitionKey.REWARD])
            reward_sum += reward
            done = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)
            if step_i % 20 == 0 or reward:
                print(f"[REPLAY] step {step_i}/{len(actions)} reward={reward} cumulative={reward_sum}", flush=True)

            precise_sleep(max(dt - (time.perf_counter() - step_start_time), 0.0))

            if done or truncated:
                print(f"[REPLAY] episode ended early at step {step_i} (done={bool(done)} truncated={bool(truncated)})", flush=True)
                break
        else:
            print(f"[REPLAY] replayed all {len(actions)} recorded steps without the episode ending on its own "
                  "(no done/truncated) -- recording-time episode length and this replay's control_time_s may "
                  "not match exactly; check the log above for whether reward=1.0 appeared anyway.", flush=True)
    except KeyboardInterrupt:
        print(f"[REPLAY] stopped early by Ctrl+C at step {step_i}.", flush=True)
    finally:
        print(f"[REPLAY] finished. total reward this run: {reward_sum} over {step_i + 1} steps.", flush=True)
        if hasattr(env, "close"):
            env.close()


def main() -> None:
    from lerobot.configs import parser
    from lerobot.configs.train import TrainRLServerPipelineConfig

    from lerobot_control.hilserl.patches import apply_all_patches

    apply_all_patches()

    @parser.wrap()
    def _cli(cfg: TrainRLServerPipelineConfig):
        run_action_replay(cfg)

    _cli()


if __name__ == "__main__":
    main()
