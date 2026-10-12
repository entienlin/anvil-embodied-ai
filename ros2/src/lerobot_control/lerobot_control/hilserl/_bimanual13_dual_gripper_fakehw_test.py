"""Fake-hardware integration test for BOTH followers having a real
gripper at once (widened from _bimanual11_gripper_fakehw_test.py's
single-arm/follower_r-only coverage, 2026-10-13 -- see
_bimanual12_dual_gripper_schema_test.py's module docstring for why "it
worked for one gripper" doesn't prove "it works for two").

Drives the real AnvilOpenArmMultiFollower composite (not a bare
AnvilOpenArmFollower) over real rclpy pub/sub against the SAME
mock_controller_node docker-compose.fake-hardware.yml's mock-robot runs,
confirming the composite's 16-wide bus/action plumbing and BOTH arms'
independent gripper commanding work end-to-end over real ROS2 messaging,
not just synthetic in-process messages.

Run inside the fake-hardware ROS2 container, with mock-robot already up:
    python3 -m lerobot_control.hilserl._bimanual13_dual_gripper_fakehw_test
"""

import sys
import time

from lerobot_control.hilserl.robot_anvil_openarm import AnvilOpenArmFollowerConfig
from lerobot_control.hilserl.robot_anvil_openarm_multi import (
    AnvilOpenArmMultiFollower,
    AnvilOpenArmMultiFollowerConfig,
)


def _check(label: str, cond: bool) -> bool:
    print(f"[{'OK' if cond else 'FAIL'}] {label}", flush=True)
    return cond


def main() -> int:
    ok = True

    print("[bimanual13] constructing AnvilOpenArmMultiFollower, BOTH arms use_gripper=True...", flush=True)
    config = AnvilOpenArmMultiFollowerConfig(
        arms=[
            AnvilOpenArmFollowerConfig(
                ros_prefix="follower_l",
                camera_topics={"wrist_l": "/cam_wrist_l/image_raw/compressed"},
                obs_timeout_s=5.0,
                use_gripper=True,
                # see _bimanual11's identical comment: the mock seeds every
                # joint (incl. gripper) from an unrealistic [-pi, pi] range,
                # unlike real hardware's [0, 0.05] -- a wide step here just
                # means "converge in a handful of ticks regardless of that".
                max_gripper_command_step_m=1.0,
            ),
            AnvilOpenArmFollowerConfig(
                ros_prefix="follower_r",
                camera_topics={"wrist_r": "/cam_wrist_r/image_raw/compressed"},
                obs_timeout_s=5.0,
                use_gripper=True,
                max_gripper_command_step_m=1.0,
            ),
        ]
    )
    robot = AnvilOpenArmMultiFollower(config)

    print("[bimanual13] connecting (real rclpy pub/sub to the mock, both arms)...", flush=True)
    robot.connect()
    print("[bimanual13] connected OK", flush=True)

    bus_keys = list(robot.bus.motors.keys())
    ok &= _check("composite bus.motors has 16 entries", len(bus_keys) == 16)
    ok &= _check(
        "order is [left x7, left gripper, right x7, right gripper]",
        bus_keys
        == [f"follower_l_joint{i}" for i in range(1, 8)]
        + ["follower_l_finger_joint1"]
        + [f"follower_r_joint{i}" for i in range(1, 8)]
        + ["follower_r_finger_joint1"],
    )

    print("[bimanual13] get_observation() exposes BOTH grippers' real positions...", flush=True)
    obs = robot.get_observation()
    ok &= _check(
        "both gripper observations present",
        "follower_l_finger_joint1.pos" in obs and "follower_r_finger_joint1.pos" in obs,
    )
    print(
        f"  mock's real initial gripper positions: left={obs['follower_l_finger_joint1.pos']:.4f} "
        f"right={obs['follower_r_finger_joint1.pos']:.4f}",
        flush=True,
    )

    print("[bimanual13] send_action(): left -> open (0.05), right -> close (0.0), simultaneously...", flush=True)
    last_left = last_right = None
    for i in range(10):
        obs = robot.get_observation()
        action = {f"{j}.pos": obs[f"{j}.pos"] for j in robot.joint_names}
        action["follower_l_finger_joint1.pos"] = 0.05
        action["follower_r_finger_joint1.pos"] = 0.0
        sent = robot.send_action(action)
        last_left = sent["follower_l_finger_joint1.pos"]
        last_right = sent["follower_r_finger_joint1.pos"]
        print(f"  [{i}] left={last_left:.4f} right={last_right:.4f}", flush=True)
        time.sleep(0.05)

    ok &= _check("left converged to its own target (0.05, open)", abs(last_left - 0.05) < 1e-6)
    ok &= _check("right converged to its own, DIFFERENT target (0.0, closed)", abs(last_right - 0.0) < 1e-6)
    ok &= _check("the two arms' gripper commands did not cross-wire (left != right)", abs(last_left - last_right) > 0.04)

    print("[bimanual13] disconnecting...", flush=True)
    robot.disconnect()

    if not ok:
        print("[bimanual13] FAILED", flush=True)
        return 1
    print("[bimanual13] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
