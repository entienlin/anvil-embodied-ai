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

Usage:
    REPLAY_ACTIONS_NPZ=/path/to/episode.npz \
    ros2 run lerobot_control hilserl_replay_actions --config_path <path>

The npz must have one array, "action", shape (N, 3) -- exactly
lerobot_control.hilserl.bc_pretrain's/demos_v2's raw recorded action
convention (delta_x/delta_y/delta_z multipliers, NOT clipped to [-1,1] --
same as what a live human's raw teleop produces, so no special-casing is
needed relative to a real intervention).
"""

import logging
import os

from lerobot_control.hilserl import robot_anvil_openarm  # noqa: F401 -- registration side effect
from lerobot_control.hilserl import teleop_anvil_openarm_leader  # noqa: F401 -- registration side effect


def run_action_replay(cfg) -> None:
    import numpy as np
    import torch
    from lerobot.rl.gym_manipulator import create_transition, make_processors, make_robot_env, step_env_and_process_transition
    from lerobot.types import TransitionKey

    npz_path = os.environ.get("REPLAY_ACTIONS_NPZ")
    if not npz_path:
        raise RuntimeError("REPLAY_ACTIONS_NPZ env var must point to a saved episode's action array (see this module's docstring).")
    actions = np.load(npz_path)["action"]
    print(f"[REPLAY] loaded {len(actions)} recorded actions from {npz_path}", flush=True)

    logging.info("[REPLAY] make_env")
    env, teleop_device = make_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.policy.device)

    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()
    transition = env_processor(create_transition(observation=obs, info=info))

    print(f"[REPLAY] starting -- replaying {len(actions)} recorded steps verbatim, no policy involved", flush=True)

    reward_sum = 0.0
    step_i = 0
    try:
        for step_i, raw_action in enumerate(actions):
            action = torch.as_tensor(raw_action, dtype=torch.float32)
            transition = step_env_and_process_transition(
                env=env,
                transition=transition,
                action=action,
                env_processor=env_processor,
                action_processor=action_processor,
            )
            reward = float(transition[TransitionKey.REWARD])
            reward_sum += reward
            done = transition.get(TransitionKey.DONE, False)
            truncated = transition.get(TransitionKey.TRUNCATED, False)
            if step_i % 20 == 0 or reward:
                print(f"[REPLAY] step {step_i}/{len(actions)} reward={reward} cumulative={reward_sum}", flush=True)
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
