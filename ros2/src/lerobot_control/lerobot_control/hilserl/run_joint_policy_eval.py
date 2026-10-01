"""Real-hardware eval loop for a JOINT-SPACE policy (ACT trained on
demos_v2_jointspace, action = the follower's own next-tick observation.state
in degrees), bypassing EE-delta/IK entirely -- same dispatch shape as
run_joint_replay.py, but driven by a policy's live prediction instead of a
pre-recorded array.

Why this exists (2026-09-30): the original 3-dim EE-delta action column was
confirmed derived from the LEADER arm's motion, not the follower's actual
observed motion -- every model trained on it (BC, DAgger, SAC/RL, ACT)
learned a label that didn't match what the robot really did. Direct
joint-space replay of the follower's own real recorded states was the only
thing that reproduced the real trajectory. This is the corresponding
deployment path for a policy retrained on that corrected data: policy
predicts a joint-position target directly, and it goes straight to
env.step() (-> AnvilOpenArmFollower.send_action()), with NO
action_processor, NO EEReferenceAndDelta, NO IK involved at all -- exactly
mirroring run_policy_eval.py's structure (make_robot_env + make_policy +
a plain step loop), just without the action_processor stage.

Usage:
    ros2 run lerobot_control hilserl_eval_joint_policy --config_path <path>

HILSERL_MAX_EPISODES=N stops cleanly after N episodes (same mechanism as
hilserl_actor/hilserl_eval_policy -- reuses patches._request_stop()).

JOINT_EVAL_TRACE_OUT=/path/to/trace.npz saves a full per-tick trace (target,
before, after -- degrees -- plus wall-clock timestamp per tick), saved even
on crash via a finally block, mirroring run_joint_replay.py's REPLAY_TRACE_OUT.
Added 2026-09-30 after three "commands cleanly stop mid-stream, no error"
occurrences on real hardware that could only be reconstructed after the fact
from aggregate log timestamps -- this gives precise, first-hand per-tick data
for whichever occurrence happens next instead of more reconstruction.
"""

import logging
import os

from lerobot_control.hilserl import robot_anvil_openarm  # noqa: F401 -- registration side effect
from lerobot_control.hilserl import teleop_anvil_openarm_leader  # noqa: F401 -- registration side effect


def _max_episodes() -> int:
    try:
        return int(os.environ.get("HILSERL_MAX_EPISODES", "0"))
    except ValueError:
        return 0


def run_joint_policy_eval(cfg) -> None:
    import time

    import numpy as np
    import torch
    from lerobot.policies.factory import make_policy
    from lerobot.rl.gym_manipulator import create_transition, make_processors, make_robot_env
    from lerobot.types import TransitionKey
    from lerobot.utils.random_utils import set_seed
    from lerobot.utils.robot_utils import precise_sleep

    from lerobot_control.hilserl import patches
    from lerobot_control.hilserl.run_policy_eval import load_policy_processors, load_sac_action_bounds

    logging.info("[JOINT-EVAL] make_env")
    env, teleop_device = make_robot_env(cfg=cfg.env)
    # env_processor still needed: it's what turns RobotEnv's raw {"follower_r_joint1.pos": ...,
    # "wrist_r": <HWC array>} observation into the policy's expected observation.state /
    # observation.images.wrist_r batched tensors (VanillaObservationProcessorStep, image
    # crop/resize, AddBatchDimensionProcessorStep, DeviceProcessorStep). Only action_processor
    # (the EE-delta/IK chain) is skipped -- the policy's raw joint-position output goes
    # straight to env.step(), same as run_joint_replay.py.
    env_processor, _action_processor = make_processors(env, teleop_device, cfg.env, cfg.policy.device)

    set_seed(cfg.seed)

    logging.info("[JOINT-EVAL] make_policy")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env).eval()
    pre_processor, post_processor = load_policy_processors(cfg.policy.pretrained_path, cfg.policy.device)
    if cfg.policy.type == "sac":
        # SAC checkpoints don't use PolicyProcessorPipeline -- see
        # load_sac_action_bounds()'s docstring for why this is needed for
        # a joint-space SAC checkpoint specifically (bc_jointspace_v1 and
        # onward), and a no-op for the original EE-delta checkpoints.
        load_sac_action_bounds(policy, cfg.policy.pretrained_path, cfg.policy.device)
    input_keys = list(cfg.policy.input_features)

    obs, info = env.reset()
    policy.reset()
    transition = env_processor(create_transition(observation=obs, info=info))

    n_target = _max_episodes()
    episode_idx, episode_steps = 0, 0

    print(
        f"[JOINT-EVAL] starting -- policy={cfg.policy.type}, joint-space (no EE-delta/IK), "
        f"{'stopping after ' + str(n_target) + ' episodes' if n_target > 0 else 'running until Ctrl+C'}. "
        "Note: bypasses action_processor, so reward/success detection never runs here -- "
        "judge by watching the arm and using s/f manually if you want a label.",
        flush=True,
    )

    # gym_manipulator.py's own control_loop() paces every real step to
    # dt=1/fps (precise_sleep(max(dt - elapsed, 0.0)) at the end of each
    # tick) -- this loop had no such pacing at all, the same gap found and
    # fixed in run_action_replay.py earlier tonight ("a ~13s real episode
    # replayed in a couple of seconds"). Here it means the policy runs at
    # whatever speed select_action()/env.step() allow (measured ~20Hz
    # against a configured 10Hz), roughly doubling the effective commanded
    # velocity relative to what the 10Hz training data represents, and
    # halving the liveness watchdog's real time margin (10 ticks was tuned
    # to be ~1s of headroom above jitter at 10Hz; at ~20Hz it's only ~0.5s).
    dt = 1.0 / cfg.env.fps

    trace_path = os.environ.get("JOINT_EVAL_TRACE_OUT")
    trace_time, trace_target, trace_before, trace_after = [], [], [], []
    joint_names = [f"follower_r_joint{i}" for i in range(1, 8)]

    def _joint_arr(observation: dict) -> np.ndarray:
        return np.array([observation[f"{j}.pos"] for j in joint_names], dtype=np.float64)

    try:
        while True:
            step_start_time = time.perf_counter()
            joints_before = obs
            observation = {k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in input_keys}
            if pre_processor is not None:
                observation = pre_processor(observation)
            with torch.no_grad():
                action = policy.select_action(batch=observation)
            if post_processor is not None:
                action = post_processor.process_action(action)
            # AnvilOpenArmFollower.send_action() indexes into this with plain Python floats
            # (action[f"{j}.pos"] -> np.array(...)) -- needs a CPU tensor, unlike the SAC actor
            # path which already ran on whatever device cfg.policy.device set (usually cpu there).
            action = action.detach().cpu()
            raw_action = action.squeeze(0) if action.dim() > 1 else action

            obs, reward, terminated, truncated, info = env.step(raw_action)
            episode_steps += 1
            transition = env_processor(create_transition(observation=obs, info=info))

            if trace_path:
                trace_time.append(time.time())
                trace_target.append(np.asarray(raw_action, dtype=np.float64))
                trace_before.append(_joint_arr(joints_before))
                trace_after.append(_joint_arr(obs))

            precise_sleep(max(dt - (time.perf_counter() - step_start_time), 0.0))

            if terminated or truncated:
                print(f"[JOINT-EVAL] episode {episode_idx} ended after {episode_steps} steps "
                      f"(done={bool(terminated)} truncated={bool(truncated)})", flush=True)
                episode_idx += 1
                episode_steps = 0

                if n_target > 0 and episode_idx >= n_target:
                    patches._request_stop("JOINT-EVAL", n_target)

                obs, info = env.reset()
                policy.reset()
                transition = env_processor(create_transition(observation=obs, info=info))

                if n_target > 0 and episode_idx >= n_target:
                    break
    except KeyboardInterrupt:
        print("[JOINT-EVAL] stopping (Ctrl+C or auto-stop).", flush=True)
    finally:
        if trace_path and trace_target:
            np.savez(
                trace_path,
                time=np.array(trace_time),
                target=np.stack(trace_target),
                before=np.stack(trace_before),
                after=np.stack(trace_after),
            )
            print(f"[JOINT-EVAL] saved {len(trace_target)}-step trace (time/target/before/after, degrees) "
                  f"to {trace_path}", flush=True)
        if hasattr(env, "close"):
            env.close()


def main() -> None:
    from lerobot.configs import parser
    from lerobot.configs.train import TrainRLServerPipelineConfig

    from lerobot_control.hilserl.patches import apply_all_patches

    apply_all_patches()

    @parser.wrap()
    def _cli(cfg: TrainRLServerPipelineConfig):
        run_joint_policy_eval(cfg)

    _cli()


if __name__ == "__main__":
    main()
