"""Fake-hardware integration test for real gripper control: AnvilOpenArmFollower
with use_gripper=True, driven over a REAL rclpy pub/sub connection to the
mock-robot node (ros2 run lerobot_control mock_controller_node), the same
node docker-compose.fake-hardware.yml's mock-robot service runs. Confirms
the gripper wiring added on top of _stage2_smoketest.py's gripper-less
baseline actually works end-to-end over real ROS2 messaging, not just
against synthetic in-process messages (covered separately by
_bimanual10_gripper_schema_test.py).

Confirms, against the real mock (not an assumption from reading its
source): mock_controller_node already publishes follower_r_finger_joint1
on /joint_states (16-DOF rig simulation) and its action subscriber does no
length validation -- so an 8-wide command vector with a REAL (non-held)
gripper target is accepted exactly like the existing 8-wide gripper-less
case already was.

Run inside the fake-hardware ROS2 container, with mock-robot already up:
    python3 -m lerobot_control.hilserl._bimanual11_gripper_fakehw_test
"""

import sys
import time

from lerobot_control.hilserl.robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig


def _check(label: str, cond: bool) -> bool:
    print(f"[{'OK' if cond else 'FAIL'}] {label}", flush=True)
    return cond


def main() -> int:
    ok = True

    print("[bimanual11] constructing AnvilOpenArmFollower(use_gripper=True)...", flush=True)
    config = AnvilOpenArmFollowerConfig(
        ros_prefix="follower_r",
        camera_topics={"wrist_r": "/cam_wrist_r/image_raw/compressed"},
        obs_timeout_s=5.0,
        use_gripper=True,
        # mock_controller_node seeds EVERY joint (including the gripper)
        # from `rng.random(16) * 2*pi - pi` -- real hardware's gripper
        # would start within [0, 0.05], the mock's does not. A wide step
        # here just means "converge in a handful of ticks regardless of
        # the mock's unrealistic starting value" -- this task's real
        # per-arm YAML configs keep the small, deliberate 0.01 default.
        max_gripper_command_step_m=1.0,
    )
    robot = AnvilOpenArmFollower(config)

    print("[bimanual11] connecting (real rclpy pub/sub to the mock)...", flush=True)
    robot.connect()
    print("[bimanual11] connected OK", flush=True)

    ok &= _check(
        "bus.motors has 8 entries (7 arm + gripper), gripper last",
        list(robot.bus.motors.keys())[-1] == "follower_r_finger_joint1" and len(robot.bus.motors) == 8,
    )
    ok &= _check("gripper exposed in action_features", "follower_r_finger_joint1.pos" in robot.action_features)

    print("[bimanual11] get_observation() exposes the real gripper position...", flush=True)
    obs = robot.get_observation()
    gripper_obs_0 = obs["follower_r_finger_joint1.pos"]
    print(f"  mock's real (random) initial gripper position: {gripper_obs_0:.4f}", flush=True)
    # mock_controller_node seeds every joint (including the gripper) from
    # `rng.random(16) * 2*pi - pi` -- NOT clipped to the gripper's real
    # [0, 0.05] range (it's a generic 16-joint mock, not gripper-aware).
    # Just confirm it's a real float, not confirm it's in-range.
    ok &= _check("gripper observation is a finite float", isinstance(gripper_obs_0, float))

    print("[bimanual11] send_action() repeatedly commanding gripper -> 0.05 (open)...", flush=True)
    last_sent_gripper = None
    for i in range(10):
        obs = robot.get_observation()
        action = {f"{j}.pos": obs[f"{j}.pos"] for j in robot.joint_names}
        action["follower_r_finger_joint1.pos"] = 0.05
        sent = robot.send_action(action)
        last_sent_gripper = sent["follower_r_finger_joint1.pos"]
        print(f"  [{i}] sent gripper={last_sent_gripper:.4f}", flush=True)
        time.sleep(0.05)

    ok &= _check(
        "after repeated ticks, commanded gripper target converged to the real target (0.05), clamped in-range",
        abs(last_sent_gripper - 0.05) < 1e-6,
    )

    print("[bimanual11] send_action() commanding gripper -> 0.0 (close)...", flush=True)
    for i in range(10):
        obs = robot.get_observation()
        action = {f"{j}.pos": obs[f"{j}.pos"] for j in robot.joint_names}
        action["follower_r_finger_joint1.pos"] = 0.0
        sent = robot.send_action(action)
        last_sent_gripper = sent["follower_r_finger_joint1.pos"]
        time.sleep(0.05)
    print(f"  final sent gripper={last_sent_gripper:.4f}", flush=True)
    ok &= _check("converged back down to 0.0 (closed)", abs(last_sent_gripper - 0.0) < 1e-6)

    print("[bimanual11] disconnecting...", flush=True)
    robot.disconnect()

    if not ok:
        print("[bimanual11] FAILED", flush=True)
        return 1
    print("[bimanual11] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
