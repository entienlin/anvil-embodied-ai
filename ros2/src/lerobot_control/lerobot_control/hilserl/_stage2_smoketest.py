"""Stage 2 smoke test: AnvilOpenArmFollower alone, against fake hardware.

Constructs a bare RobotEnv directly (no SAC, no teleop) and exercises
connect()/get_observation()/send_action()/reset()/step() a few times,
to validate the Robot ABC implementation and the robot.bus.motors /
robot.cameras shims RobotEnv needs.

Run inside the fake-hardware ROS2 container, with mock-robot already up:
    ros2 run lerobot_control hilserl_gym_manipulator  # NOT this -- see below
    python3 -m lerobot_control.hilserl._stage2_smoketest
"""

import sys
import time

from lerobot_control.hilserl.robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig


def main() -> int:
    print("[stage2] constructing AnvilOpenArmFollower...", flush=True)
    config = AnvilOpenArmFollowerConfig(
        ros_prefix="follower_r",
        camera_topics={"wrist_r": "/cam_wrist_r/image_raw/compressed"},
        obs_timeout_s=5.0,
    )
    robot = AnvilOpenArmFollower(config)

    print("[stage2] connecting...", flush=True)
    robot.connect()
    print("[stage2] connected OK", flush=True)

    print(f"[stage2] bus.motors keys: {list(robot.bus.motors.keys())}", flush=True)
    print(f"[stage2] cameras keys: {list(robot.cameras.keys())}", flush=True)

    print("[stage2] get_observation() x5...", flush=True)
    for i in range(5):
        obs = robot.get_observation()
        joint_vals = {k: v for k, v in obs.items() if k.endswith(".pos")}
        img_keys = [k for k in obs if k not in joint_vals]
        print(f"  [{i}] joints={joint_vals}  images={img_keys}", flush=True)
        time.sleep(0.1)

    print("[stage2] send_action() x5 (small deltas from current position)...", flush=True)
    obs = robot.get_observation()
    for i in range(5):
        action = {f"{j}.pos": obs[f"{j}.pos"] for j in robot.joint_names}
        sent = robot.send_action(action)
        print(f"  [{i}] sent={sent}", flush=True)
        time.sleep(0.1)
        obs = robot.get_observation()

    print("[stage2] disconnecting...", flush=True)
    robot.disconnect()
    print("[stage2] PASSED", flush=True)
    return 0


def test_bare_robot_env() -> int:
    """Second half: construct RobotEnv directly against a fresh connection,
    to exercise the robot.bus.motors/robot.cameras shims the way
    gym_manipulator.py's RobotEnv actually does (not just our own Robot
    methods in isolation)."""
    from lerobot.rl.gym_manipulator import RobotEnv

    config = AnvilOpenArmFollowerConfig(
        ros_prefix="follower_r",
        camera_topics={"wrist_r": "/cam_wrist_r/image_raw/compressed"},
        obs_timeout_s=5.0,
    )
    robot = AnvilOpenArmFollower(config)
    robot.connect()

    print("[stage2/RobotEnv] constructing RobotEnv...", flush=True)
    try:
        env = RobotEnv(robot=robot, use_gripper=False, display_cameras=False)
    except Exception as e:
        print(f"[stage2/RobotEnv] FAILED to construct: {type(e).__name__}: {e}", flush=True)
        robot.disconnect()
        return 1

    print(f"[stage2/RobotEnv] joint_names from robot.bus.motors: {env._joint_names}", flush=True)
    print(f"[stage2/RobotEnv] image_keys from robot.cameras: {list(env._image_keys)}", flush=True)

    print("[stage2/RobotEnv] reset()...", flush=True)
    obs, info = env.reset()
    print(f"[stage2/RobotEnv] reset OK, obs keys={list(obs.keys()) if hasattr(obs, 'keys') else type(obs)}", flush=True)

    print("[stage2/RobotEnv] step() x3 with joint-space action (bare RobotEnv expects", flush=True)
    print("  one value per robot.bus.motors entry -- EE-delta actions only apply once", flush=True)
    print("  wrapped by make_processors(), tested properly in Stage 3)...", flush=True)
    import numpy as np
    current_joint_action = np.array([obs[f"{j}.pos"] for j in env._joint_names], dtype=np.float32)
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(current_joint_action)
        print(f"  [{i}] reward={reward} terminated={terminated}", flush=True)
        current_joint_action = np.array([obs[f"{j}.pos"] for j in env._joint_names], dtype=np.float32)

    robot.disconnect()
    print("[stage2/RobotEnv] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    if rc == 0:
        rc = test_bare_robot_env()
    sys.exit(rc)
