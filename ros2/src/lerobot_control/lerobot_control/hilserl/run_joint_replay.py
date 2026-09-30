"""Open-loop replay of pre-recorded JOINT positions, bypassing EE-delta and
inverse kinematics entirely -- the truest "just replay what actually
happened" test.

Why this exists (2026-09-30): replaying a demo's action column (EE-delta
multipliers, whether the original recorded ones or a "corrected" version
derived straight from the follower's own observed motion) still came out
shaky on real hardware. Both of those go through EEReferenceAndDelta ->
InverseKinematicsRLStep every tick -- and this OpenArm is 7-DOF, i.e.
kinematically REDUNDANT for a 6-DoF end-effector pose (position + no
orientation control even, just position): infinitely many joint
configurations can reach the same EE position. The IK solver is reseeded
from the real current joints every tick (patch_inverse_kinematics_to_
always_reseed_from_real_joints) but still solves FRESH each tick, so even
a perfectly correct EE-delta sequence isn't guaranteed to reconstruct the
same JOINT-space path each time -- small per-tick differences in which
null-space solution the solver settles on can look shaky even when the
Cartesian target sequence is fine.

This script removes that variable completely: it commands the recorded
JOINT positions (observation.state / state_deg from the same ep_*.npz
files -- already exactly what a human's real teleoperation produced,
proven successful at recording time) straight to AnvilOpenArmFollower.
send_action(), with NO action_processor, NO EEReferenceAndDelta, NO IK
involved at all. If this is smooth and reproduces the real motion, the
IK-redundancy hypothesis above is confirmed and the EE-delta layer itself
needs to change (e.g. avoid re-solving IK from scratch every tick, or
constrain/bias the null space). If THIS is also shaky, the problem is
elsewhere -- the limiter, the heartbeat, or the hardware/controller layer
itself -- and the EE-delta/IK layer is exonerated.

Usage -- same env vars as hilserl_replay_actions, but reads "state_deg"
(the recorded observation.state) instead of "action":
    REPLAY_ACTIONS_NPZ=/path/to/episode.npz \
    ros2 run lerobot_control hilserl_replay_joints --config_path <path>
or
    REPLAY_EPISODE_DIR=/workspace/outputs/some_session \
    ros2 run lerobot_control hilserl_replay_joints --config_path <path>
"""

import logging
import os

from lerobot_control.hilserl import robot_anvil_openarm  # noqa: F401 -- registration side effect
from lerobot_control.hilserl import teleop_anvil_openarm_leader  # noqa: F401 -- registration side effect
from lerobot_control.hilserl.run_action_replay import _find_latest_successful_episode


def run_joint_replay(cfg) -> None:
    import time

    import numpy as np
    import torch
    from lerobot.rl.gym_manipulator import make_robot_env
    from lerobot.utils.robot_utils import precise_sleep

    npz_path = os.environ.get("REPLAY_ACTIONS_NPZ")
    episode_dir = os.environ.get("REPLAY_EPISODE_DIR")
    if npz_path:
        pass
    elif episode_dir:
        npz_path = _find_latest_successful_episode(episode_dir)
        print(f"[JOINT-REPLAY] REPLAY_EPISODE_DIR set -- auto-selected: {npz_path}", flush=True)
    else:
        raise RuntimeError("Set REPLAY_ACTIONS_NPZ or REPLAY_EPISODE_DIR (see this module's docstring).")

    states = np.load(npz_path)["state_deg"]
    print(f"[JOINT-REPLAY] loaded {len(states)} recorded joint states (degrees) from {npz_path}", flush=True)

    logging.info("[JOINT-REPLAY] make_env")
    env, _teleop_device = make_robot_env(cfg=cfg.env)

    obs, _info = env.reset()

    def _joint_pos(observation: dict) -> dict:
        return {k: round(float(v), 2) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")}

    print(f"[JOINT-REPLAY] starting -- commanding {len(states)} recorded joint states directly, "
          "no EE-delta/IK involved at all", flush=True)
    print("[JOINT-REPLAY] note: this bypasses action_processor entirely, so reward/success detection "
          "never runs -- reward will always print 0.0 here regardless of outcome. Judge this one by "
          "watching the arm, not by the printed reward.", flush=True)
    print(f"[JOINT-REPLAY] joints BEFORE step 0: {_joint_pos(obs)}", flush=True)

    dt = 1.0 / cfg.env.fps
    step_i = 0
    try:
        for step_i, joint_deg in enumerate(states):
            step_start_time = time.perf_counter()
            action = torch.as_tensor(joint_deg, dtype=torch.float32)
            joints_before = _joint_pos(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            joints_after = _joint_pos(obs)
            if step_i % 10 == 0 or reward:
                print(f"[JOINT-REPLAY] step {step_i}/{len(states)} reward={float(reward)} "
                      f"target={np.round(joint_deg, 2).tolist()} before={joints_before} after={joints_after}",
                      flush=True)
            precise_sleep(max(dt - (time.perf_counter() - step_start_time), 0.0))
            if terminated or truncated:
                print(f"[JOINT-REPLAY] episode ended early at step {step_i} "
                      f"(done={bool(terminated)} truncated={bool(truncated)})", flush=True)
                break
    except KeyboardInterrupt:
        print(f"[JOINT-REPLAY] stopped early by Ctrl+C at step {step_i}.", flush=True)
    finally:
        print(f"[JOINT-REPLAY] finished at step {step_i}.", flush=True)
        if hasattr(env, "close"):
            env.close()


def main() -> None:
    from lerobot.configs import parser
    from lerobot.configs.train import TrainRLServerPipelineConfig

    from lerobot_control.hilserl.patches import apply_all_patches

    apply_all_patches()

    @parser.wrap()
    def _cli(cfg: TrainRLServerPipelineConfig):
        run_joint_replay(cfg)

    _cli()


if __name__ == "__main__":
    main()
