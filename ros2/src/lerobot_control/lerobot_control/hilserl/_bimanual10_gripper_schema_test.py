"""Offline test for real gripper control (follower_r/leader_r only), added
on top of the existing bimanual joint-space pipeline (see _bimanual1..9's
own tests for the gripper-less baseline this builds on).

Same pure-Python, no-connect(), no-ROS2-daemon pattern as the other
_bimanualN_*_test.py files -- messages are constructed directly and fed
into internal state (_obs_manager.update_joint_state(), _on_joint_state()),
never a real rclpy.init()/node/executor.

Validates:
  - AnvilOpenArmFollowerConfig.use_gripper=False (default) leaves
    observation_features/action_features/bus.motors byte-for-byte
    unchanged from the gripper-less baseline (regression guard for every
    existing single-arm/follower_l config).
  - use_gripper=True adds exactly one extra key
    ("<prefix>_finger_joint1.pos") to observation_features/action_features
    and one extra entry to bus.motors.
  - AnvilOpenArmMultiFollower composed of [follower_l (no gripper),
    follower_r (gripper)] has 15 total bus.motors entries, with the
    gripper last (end of follower_r's own block, matching this project's
    left-then-right joint order convention).
  - send_action() with use_gripper=True actually COMMANDS the requested
    gripper target (clamped to [gripper_min_m, gripper_max_m], rate-limited
    by max_gripper_command_step_m) instead of holding the current position.
  - send_action() with use_gripper=False (default) is an exact behavioral
    regression check: still holds the gripper at its current observed
    position, publishing the unchanged 8th wire value.
  - get_observation() with use_gripper=True exposes the gripper's RAW
    (meters, not rad2deg'd) position.
  - AnvilOpenArmLeaderConfig(use_gripper=True, action_mode="ee_delta")
    raises ValueError at construction (no gripper slot in ee_delta mode).
  - AnvilOpenArmLeader._get_action_joint_space() with use_gripper=True
    appends the LEADER's own raw (meters) finger_joint1 reading, not the
    follower's.
  - AnvilOpenArmMultiLeader action_features/get_action() correctly handle
    a MIXED arms list (one gripper-enabled, one not).

Run: python3 -m lerobot_control.hilserl._bimanual10_gripper_schema_test
"""

import sys

import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from lerobot_control.hilserl.robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig
from lerobot_control.hilserl.robot_anvil_openarm_multi import (
    AnvilOpenArmMultiFollower,
    AnvilOpenArmMultiFollowerConfig,
)
from lerobot_control.hilserl.teleop_anvil_openarm_leader import AnvilOpenArmLeader, AnvilOpenArmLeaderConfig
from lerobot_control.hilserl.teleop_anvil_openarm_multi_leader import (
    AnvilOpenArmMultiLeader,
    AnvilOpenArmMultiLeaderConfig,
)


def _check(label: str, cond: bool) -> bool:
    print(f"[{'OK' if cond else 'FAIL'}] {label}", flush=True)
    return cond


def _follower_joint_state(prefix: str, arm_deg: np.ndarray, gripper_m: float) -> JointState:
    msg = JointState()
    msg.name = [f"{prefix}_joint{i}" for i in range(1, 8)] + [f"{prefix}_finger_joint1"]
    msg.position = list(np.deg2rad(arm_deg)) + [gripper_m]
    return msg


def main() -> int:
    ok = True

    print("[bimanual10] use_gripper=False (default) is byte-for-byte unchanged...", flush=True)
    plain_cfg = AnvilOpenArmFollowerConfig(ros_prefix="follower_l", camera_topics={"wrist_l": "/x"})
    plain = AnvilOpenArmFollower(plain_cfg)
    ok &= _check("no gripper in observation_features", "follower_l_finger_joint1.pos" not in plain.observation_features)
    ok &= _check("no gripper in action_features", "follower_l_finger_joint1.pos" not in plain.action_features)
    ok &= _check("bus.motors has 7 entries", len(plain.bus.motors) == 7)

    print("\n[bimanual10] use_gripper=True adds exactly one extra feature...", flush=True)
    # No cameras: get_observation()/send_action() below need no real
    # connection, but get_observation() DOES block on
    # has_complete_observation(required_cameras) -- an empty camera_topics
    # dict makes that trivially satisfied by joint_state alone, keeping
    # this test pure-Python with no fake image plumbing needed.
    grip_cfg = AnvilOpenArmFollowerConfig(ros_prefix="follower_r", camera_topics={}, use_gripper=True)
    grip = AnvilOpenArmFollower(grip_cfg)
    grip._is_connected = True
    ok &= _check("gripper in observation_features", "follower_r_finger_joint1.pos" in grip.observation_features)
    ok &= _check("gripper in action_features", "follower_r_finger_joint1.pos" in grip.action_features)
    ok &= _check("bus.motors has 8 entries", len(grip.bus.motors) == 8)
    ok &= _check("gripper is the LAST bus.motors entry", list(grip.bus.motors.keys())[-1] == "follower_r_finger_joint1")

    print("\n[bimanual10] composite [follower_l (no gripper), follower_r (gripper)]...", flush=True)
    multi = AnvilOpenArmMultiFollower(
        AnvilOpenArmMultiFollowerConfig(
            arms=[
                AnvilOpenArmFollowerConfig(ros_prefix="follower_l", camera_topics={"wrist_l": "/x"}),
                AnvilOpenArmFollowerConfig(ros_prefix="follower_r", camera_topics={"wrist_r": "/x"}, use_gripper=True),
            ]
        )
    )
    ok &= _check("composite bus.motors has 15 entries (7+7+1)", len(multi.bus.motors) == 15)
    ok &= _check(
        "gripper is the very last entry, after both arms' joints",
        list(multi.bus.motors.keys())[-1] == "follower_r_finger_joint1",
    )
    ok &= _check(
        "left arm has no gripper feature anywhere in the merged action_features",
        "follower_l_finger_joint1.pos" not in multi.action_features,
    )

    print("\n[bimanual10] get_observation() exposes the gripper's RAW (meters) position...", flush=True)
    grip._obs_manager.update_joint_state(_follower_joint_state("follower_r", np.zeros(7), gripper_m=0.023))
    obs = grip.get_observation()
    ok &= _check(
        "gripper observation is raw meters, no rad2deg applied",
        abs(obs["follower_r_finger_joint1.pos"] - 0.023) < 1e-9,
    )

    print("\n[bimanual10] send_action() with use_gripper=True COMMANDS the target (not a hold)...", flush=True)
    grip._is_connected = True

    class _Pub:
        def publish(self, m):
            pass

    grip._pub = _Pub()
    grip._obs_manager.update_joint_state(_follower_joint_state("follower_r", np.zeros(7), gripper_m=0.0))
    start = Float64MultiArray()
    start.data = [0.0] * 8
    grip._last_sent_msg = start
    action = {f"follower_r_joint{i}.pos": 0.0 for i in range(1, 8)}
    action["follower_r_finger_joint1.pos"] = 0.05  # full open, far from current (0.0) and from the hold value
    sent = grip.send_action(action)
    ok &= _check(
        "returned commanded gripper value moved toward the target (not held at 0.0)",
        sent["follower_r_finger_joint1.pos"] > 0.0,
    )
    ok &= _check(
        "published wire value matches the returned commanded value",
        abs(grip._last_sent_msg.data[-1] - sent["follower_r_finger_joint1.pos"]) < 1e-9,
    )
    ok &= _check(
        "per-tick step limit respected (max_gripper_command_step_m=0.01 default)",
        sent["follower_r_finger_joint1.pos"] <= 0.01 + 1e-9,
    )

    print("\n[bimanual10] send_action() clamps an out-of-range gripper target...", flush=True)
    grip._last_sent_msg.data = [0.0] * 7 + [0.04]  # start close to the max so one big step saturates the clamp
    action["follower_r_finger_joint1.pos"] = 10.0  # absurd, out of [0, 0.05]
    sent2 = grip.send_action(action)
    ok &= _check(
        "commanded gripper value clamped to gripper_max_m=0.05",
        sent2["follower_r_finger_joint1.pos"] <= 0.05 + 1e-9,
    )

    print("\n[bimanual10] send_action() with use_gripper=False still holds (regression)...", flush=True)
    plain._is_connected = True
    plain._pub = _Pub()
    plain._obs_manager.update_joint_state(_follower_joint_state("follower_l", np.zeros(7), gripper_m=0.017))
    plain_start = Float64MultiArray()
    plain_start.data = [0.0] * 8
    plain._last_sent_msg = plain_start
    plain_action = {f"follower_l_joint{i}.pos": 0.0 for i in range(1, 8)}
    plain_sent = plain.send_action(plain_action)
    ok &= _check("no gripper key in the gripper-less return dict", "follower_l_finger_joint1.pos" not in plain_sent)
    ok &= _check(
        "published wire value still just echoes the observed gripper position (0.017), unaffected by any target",
        abs(plain._last_sent_msg.data[-1] - 0.017) < 1e-9,
    )

    print("\n[bimanual10] AnvilOpenArmLeaderConfig rejects use_gripper=True with action_mode='ee_delta'...", flush=True)
    try:
        AnvilOpenArmLeader(AnvilOpenArmLeaderConfig(ros_prefix="leader_r", action_mode="ee_delta", use_gripper=True))
        ok &= _check("raises ValueError", False)
    except ValueError:
        ok &= _check("raises ValueError", True)

    print("\n[bimanual10] AnvilOpenArmLeader joint_space + use_gripper=True...", flush=True)
    leader_cfg = AnvilOpenArmLeaderConfig(ros_prefix="leader_r", action_mode="joint_space", use_gripper=True)
    leader = AnvilOpenArmLeader(leader_cfg)
    ok &= _check("action_features shape is (8,)", leader.action_features["shape"] == (8,))

    leader_js = JointState()
    leader_js.name = [f"leader_r_joint{i}" for i in range(1, 8)] + ["leader_r_finger_joint1"]
    leader_js.position = list(np.deg2rad(np.arange(1, 8, dtype=float))) + [0.031]
    leader._on_joint_state(leader_js)
    leader_action = leader._get_action_joint_space()
    ok &= _check("leader action has 8 elements", leader_action.shape == (8,))
    ok &= _check("first 7 elements are the arm joints in degrees", np.allclose(leader_action[:7], np.arange(1, 8)))
    ok &= _check(
        "8th element is the LEADER's own raw gripper reading (meters, no rad2deg)",
        abs(leader_action[7] - 0.031) < 1e-6,
    )

    print("\n[bimanual10] AnvilOpenArmMultiLeader with a MIXED arms list (one gripper, one not)...", flush=True)
    ml = AnvilOpenArmMultiLeader(
        AnvilOpenArmMultiLeaderConfig(
            arms=[
                AnvilOpenArmLeaderConfig(ros_prefix="leader_l", action_mode="joint_space"),
                AnvilOpenArmLeaderConfig(ros_prefix="leader_r", action_mode="joint_space", use_gripper=True),
            ]
        )
    )
    ok &= _check("combined action_features shape is (15,) (7+7+1)", ml.action_features["shape"] == (15,))

    ml._is_connected = True
    ml._arms[0]._is_connected = True
    ml._arms[1]._is_connected = True
    left_js = JointState()
    left_js.name = [f"leader_l_joint{i}" for i in range(1, 8)]
    left_js.position = list(np.deg2rad(np.arange(11, 18, dtype=float)))
    ml._arms[0]._on_joint_state(left_js)
    ml._arms[1]._on_joint_state(leader_js)

    combined = ml.get_action()
    ok &= _check("combined get_action() has 15 elements", combined.shape == (15,))
    ok &= _check("left arm's 7 joints come first", np.allclose(combined[:7], np.arange(11, 18)))
    ok &= _check("right arm's 7 joints follow", np.allclose(combined[7:14], np.arange(1, 8)))
    ok &= _check("right arm's gripper is the final element", abs(combined[14] - 0.031) < 1e-6)

    if not ok:
        print("\n[bimanual10] FAILED", flush=True)
        return 1
    print("\n[bimanual10] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
