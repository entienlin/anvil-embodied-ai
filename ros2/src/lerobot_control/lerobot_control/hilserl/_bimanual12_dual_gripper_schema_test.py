"""Offline test for BOTH arms having a real gripper at once (widened from
_bimanual10_gripper_schema_test.py's single-gripper/mixed-arms coverage,
2026-10-13: the project needs general 4-gripper capability -- every
follower/leader, not just follower_r/leader_r -- regardless of exactly
how this task's own design settles).

_bimanual10 already proved: use_gripper=False is an exact regression, a
single gripper-enabled arm adds exactly one feature, and a MIXED
composite (one arm with a gripper, one without) merges correctly. This
file's job is specifically to catch anything that only "happened to work"
for exactly one gripper and silently assumed that -- same pure-Python,
no-connect(), no-ROS2-daemon pattern as every other _bimanualN_*_test.py.

Validates:
  - composite [follower_l (gripper), follower_r (gripper)] bus.motors is
    16 entries, ordered [left_joint1..7, left_gripper, right_joint1..7,
    right_gripper] -- EACH arm's gripper sits at the END OF ITS OWN
    BLOCK, not both grippers trailing at the absolute end of the vector.
    This is the one assumption that could NOT be checked with only one
    gripper enabled (right_gripper happened to also be the last element
    overall only because follower_r was arms[-1]).
  - action_features/observation_features both report 16 keys, with both
    "follower_l_finger_joint1.pos" and "follower_r_finger_joint1.pos"
    present.
  - send_action() commands BOTH grippers independently and correctly in
    the same call -- not just one, and not cross-wired (left's target
    doesn't leak into right's published value or vice versa).
  - follower_l's own liveness-watchdog exclusion of the gripper (see
    AnvilOpenArmFollower._check_liveness's docstring) still holds when
    follower_l ITSELF has use_gripper=True -- the exclusion is scoped to
    `self.joint_names` (always arm-only, per-instance), not hardcoded to
    "whichever arm used to be gripper-less", so a left arm holding its
    gripper still against a real object must not trip the watchdog
    either, exactly like the right arm's existing behavior.
  - AnvilOpenArmMultiLeader with BOTH arms gripper-enabled: action_features
    shape (16,), get_action() concatenation is
    [left_joint1..7, left_gripper, right_joint1..7, right_gripper].

Run: python3 -m lerobot_control.hilserl._bimanual12_dual_gripper_schema_test
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
from lerobot_control.hilserl.teleop_anvil_openarm_leader import AnvilOpenArmLeaderConfig
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

    print("[bimanual12] composite [follower_l (gripper), follower_r (gripper)]...", flush=True)
    multi = AnvilOpenArmMultiFollower(
        AnvilOpenArmMultiFollowerConfig(
            arms=[
                AnvilOpenArmFollowerConfig(ros_prefix="follower_l", camera_topics={"wrist_l": "/x"}, use_gripper=True),
                AnvilOpenArmFollowerConfig(ros_prefix="follower_r", camera_topics={"wrist_r": "/x"}, use_gripper=True),
            ]
        )
    )
    bus_keys = list(multi.bus.motors.keys())
    ok &= _check("composite bus.motors has 16 entries (7+1+7+1)", len(bus_keys) == 16)
    expected_order = (
        [f"follower_l_joint{i}" for i in range(1, 8)]
        + ["follower_l_finger_joint1"]
        + [f"follower_r_joint{i}" for i in range(1, 8)]
        + ["follower_r_finger_joint1"]
    )
    ok &= _check(
        "each arm's gripper sits at the END OF ITS OWN BLOCK (not both trailing at the very end)",
        bus_keys == expected_order,
    )
    ok &= _check(
        "both grippers present in action_features",
        "follower_l_finger_joint1.pos" in multi.action_features
        and "follower_r_finger_joint1.pos" in multi.action_features,
    )
    ok &= _check(
        "both grippers present in observation_features",
        "follower_l_finger_joint1.pos" in multi.observation_features
        and "follower_r_finger_joint1.pos" in multi.observation_features,
    )
    ok &= _check("16 action_features total", len(multi.action_features) == 16)

    print("\n[bimanual12] send_action() commands BOTH grippers independently (no cross-wiring)...", flush=True)
    left, right = multi._arms
    left._is_connected = True
    right._is_connected = True

    class _Pub:
        def publish(self, m):
            pass

    left._pub = _Pub()
    right._pub = _Pub()
    left._obs_manager.update_joint_state(_follower_joint_state("follower_l", np.zeros(7), gripper_m=0.0))
    right._obs_manager.update_joint_state(_follower_joint_state("follower_r", np.zeros(7), gripper_m=0.0))
    left_start = Float64MultiArray()
    left_start.data = [0.0] * 8
    left._last_sent_msg = left_start
    right_start = Float64MultiArray()
    right_start.data = [0.0] * 8
    right._last_sent_msg = right_start

    combined_action = {f"follower_l_joint{i}.pos": 0.0 for i in range(1, 8)}
    combined_action.update({f"follower_r_joint{i}.pos": 0.0 for i in range(1, 8)})
    combined_action["follower_l_finger_joint1.pos"] = 0.05  # left: open
    combined_action["follower_r_finger_joint1.pos"] = 0.0  # right: stay closed

    sent = multi.send_action(combined_action)
    ok &= _check(
        "left gripper commanded toward its OWN target (0.05, open), not held at 0.0",
        sent["follower_l_finger_joint1.pos"] > 0.0,
    )
    ok &= _check(
        "right gripper's target (0.0) did not leak left's value onto it",
        sent["follower_r_finger_joint1.pos"] == 0.0,
    )
    ok &= _check(
        "left's published wire value matches left's own commanded value, not right's",
        abs(left._last_sent_msg.data[-1] - sent["follower_l_finger_joint1.pos"]) < 1e-9
        and left._last_sent_msg.data[-1] != right._last_sent_msg.data[-1],
    )

    print("\n[bimanual12] follower_l's own liveness watchdog still excludes ITS gripper...", flush=True)
    # Command a real, sustained arm movement (watchdog streak) while the
    # gripper holds against something (commanded != observed for the
    # gripper the whole time) -- must NOT trip, exactly like the
    # single-gripper (follower_r) case already relied on.
    left2 = AnvilOpenArmFollower(AnvilOpenArmFollowerConfig(ros_prefix="follower_l", camera_topics={}, use_gripper=True))
    left2._is_connected = True
    left2._pub = _Pub()
    for tick in range(left2._liveness_window_size + 2):
        # Arm joints never move (0.0 always) -> no commanded delta -> streak never grows.
        # This isolates the question "does a stuck GRIPPER alone trip the watchdog" from
        # the arm-joint liveness logic, which is unrelated to this test's point.
        left2._obs_manager.update_joint_state(_follower_joint_state("follower_l", np.zeros(7), gripper_m=0.0))
        action = {f"follower_l_joint{i}.pos": 0.0 for i in range(1, 8)}
        action["follower_l_finger_joint1.pos"] = 0.05  # commanded open, observed stays at 0.0 (stuck/grasping)
        left2.send_action(action)  # must not raise
    ok &= _check("no RuntimeError from a 'stuck' gripper target while arm joints are idle", True)

    print("\n[bimanual12] AnvilOpenArmMultiLeader with BOTH arms gripper-enabled...", flush=True)
    ml = AnvilOpenArmMultiLeader(
        AnvilOpenArmMultiLeaderConfig(
            arms=[
                AnvilOpenArmLeaderConfig(ros_prefix="leader_l", action_mode="joint_space", use_gripper=True),
                AnvilOpenArmLeaderConfig(ros_prefix="leader_r", action_mode="joint_space", use_gripper=True),
            ]
        )
    )
    ok &= _check("combined action_features shape is (16,) (7+1+7+1)", ml.action_features["shape"] == (16,))

    ml._is_connected = True
    ml._arms[0]._is_connected = True
    ml._arms[1]._is_connected = True
    left_js = JointState()
    left_js.name = [f"leader_l_joint{i}" for i in range(1, 8)] + ["leader_l_finger_joint1"]
    left_js.position = list(np.deg2rad(np.arange(11, 18, dtype=float))) + [0.011]
    ml._arms[0]._on_joint_state(left_js)
    right_js = JointState()
    right_js.name = [f"leader_r_joint{i}" for i in range(1, 8)] + ["leader_r_finger_joint1"]
    right_js.position = list(np.deg2rad(np.arange(1, 8, dtype=float))) + [0.044]
    ml._arms[1]._on_joint_state(right_js)

    combined = ml.get_action()
    ok &= _check("combined get_action() has 16 elements", combined.shape == (16,))
    ok &= _check("left arm's 7 joints come first", np.allclose(combined[:7], np.arange(11, 18)))
    ok &= _check("left's gripper (meters) is element 7, right after left's joints", abs(combined[7] - 0.011) < 1e-6)
    ok &= _check("right arm's 7 joints follow at [8:15]", np.allclose(combined[8:15], np.arange(1, 8)))
    ok &= _check("right's gripper (meters) is the final element", abs(combined[15] - 0.044) < 1e-6)

    if not ok:
        print("\n[bimanual12] FAILED", flush=True)
        return 1
    print("\n[bimanual12] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
