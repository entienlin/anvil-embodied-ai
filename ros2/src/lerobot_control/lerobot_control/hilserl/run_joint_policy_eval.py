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
    import torch
    from lerobot.policies.factory import make_policy
    from lerobot.rl.gym_manipulator import create_transition, make_processors, make_robot_env
    from lerobot.types import TransitionKey
    from lerobot.utils.random_utils import set_seed

    from lerobot_control.hilserl import patches

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

    try:
        while True:
            observation = {k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in input_keys}
            with torch.no_grad():
                action = policy.select_action(batch=observation)
            # AnvilOpenArmFollower.send_action() indexes into this with plain Python floats
            # (action[f"{j}.pos"] -> np.array(...)) -- needs a CPU tensor, unlike the SAC actor
            # path which already ran on whatever device cfg.policy.device set (usually cpu there).
            action = action.detach().cpu()
            raw_action = action.squeeze(0) if action.dim() > 1 else action

            obs, reward, terminated, truncated, info = env.step(raw_action)
            episode_steps += 1
            transition = env_processor(create_transition(observation=obs, info=info))

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
