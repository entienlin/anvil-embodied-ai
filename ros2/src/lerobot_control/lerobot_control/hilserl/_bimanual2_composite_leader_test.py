"""Offline test for AnvilOpenArmMultiLeader (teleop_anvil_openarm_multi_leader.py),
the composite Teleoperator that drives 1-or-N AnvilOpenArmLeader sub-arms
through ONE shared KeyboardIntervention instance.

Same pure-Python, no-connect(), no-ROS2-daemon pattern as every other
_stageN_*_test.py / _bimanualN_*_test.py file in this package.

Validates:
  - every sub-leader shares the literal SAME KeyboardIntervention instance
    (not one each) -- the real hazard this design exists to avoid (two
    independent raw-stdin readers racing for every keystroke, see
    keyboard_intervention.py's module docstring) can only be ruled out by
    checking object identity, not just behavior.
  - sub-leaders constructed with a shared keyboard correctly defer
    start()/stop() lifecycle to the composite (own_keyboard_lifecycle=False).
  - action_features reports the correct concatenated shape (14 for 2
    arms x 7 joints).
  - get_action() concatenates every sub-arm's own joint-space action in
    config.arms' list order (left arm's 7 joints, then right arm's 7),
    not interleaved or reordered.
  - constructing a multi-leader with any arm NOT in "joint_space" mode
    raises at construction -- "ee_delta" has no multi-arm equivalent (see
    AnvilOpenArmMultiLeader's own docstring for why) and must fail loudly,
    not silently produce a wrong-shaped or wrong-meaning action.

Run: python3 -m lerobot_control.hilserl._bimanual2_composite_leader_test
"""

import sys

import numpy as np
from sensor_msgs.msg import JointState

from lerobot_control.hilserl.teleop_anvil_openarm_leader import AnvilOpenArmLeaderConfig
from lerobot_control.hilserl.teleop_anvil_openarm_multi_leader import (
    AnvilOpenArmMultiLeader,
    AnvilOpenArmMultiLeaderConfig,
)


def _check(label: str, cond: bool) -> bool:
    print(f"[{'OK' if cond else 'FAIL'}] {label}", flush=True)
    return cond


def _joint_state(leader, joint_deg: np.ndarray) -> JointState:
    msg = JointState()
    msg.name = list(leader.joint_names)
    msg.position = list(np.deg2rad(joint_deg))
    return msg


def main() -> int:
    ok = True

    print("[bimanual2] constructing a 2-arm multi-leader...", flush=True)
    cfg = AnvilOpenArmMultiLeaderConfig(
        arms=[
            AnvilOpenArmLeaderConfig(ros_prefix="leader_l", action_mode="joint_space"),
            AnvilOpenArmLeaderConfig(ros_prefix="leader_r", action_mode="joint_space"),
        ]
    )
    ml = AnvilOpenArmMultiLeader(cfg)

    ok &= _check(
        "both sub-leaders share the literal same KeyboardIntervention instance",
        ml._arms[0]._keyboard is ml._arms[1]._keyboard is ml._keyboard,
    )
    ok &= _check(
        "sub-leaders defer keyboard start()/stop() to the composite",
        not ml._arms[0]._owns_keyboard_lifecycle and not ml._arms[1]._owns_keyboard_lifecycle,
    )
    ok &= _check("action_features shape is (14,) for 2 arms x 7 joints", ml.action_features["shape"] == (14,))

    print("\n[bimanual2] get_action() concatenates in left-then-right arm order...", flush=True)
    ml._is_connected = True
    ml._arms[0]._is_connected = True
    ml._arms[1]._is_connected = True
    ml._arms[0]._on_joint_state(_joint_state(ml._arms[0], np.arange(1, 8, dtype=float)))
    ml._arms[1]._on_joint_state(_joint_state(ml._arms[1], np.arange(11, 18, dtype=float)))

    action = ml.get_action()
    ok &= _check("concatenated action is an ndarray of length 14", isinstance(action, np.ndarray) and action.shape == (14,))
    ok &= _check(
        "left arm's 7 values precede right arm's 7, each in its own joint order",
        np.allclose(action[:7], np.arange(1, 8)) and np.allclose(action[7:], np.arange(11, 18)),
    )

    print("\n[bimanual2] any arm using ee_delta (instead of joint_space) is rejected at construction...", flush=True)
    try:
        AnvilOpenArmMultiLeader(
            AnvilOpenArmMultiLeaderConfig(
                arms=[
                    AnvilOpenArmLeaderConfig(ros_prefix="leader_l", action_mode="ee_delta"),
                    AnvilOpenArmLeaderConfig(ros_prefix="leader_r", action_mode="joint_space"),
                ]
            )
        )
        ok &= _check("ee_delta arm raises ValueError", False)
    except ValueError:
        ok &= _check("ee_delta arm raises ValueError", True)

    print("\n[bimanual2] duplicate ros_prefix across arms raises at construction...", flush=True)
    try:
        AnvilOpenArmMultiLeader(
            AnvilOpenArmMultiLeaderConfig(
                arms=[
                    AnvilOpenArmLeaderConfig(ros_prefix="leader_r", action_mode="joint_space"),
                    AnvilOpenArmLeaderConfig(ros_prefix="leader_r", action_mode="joint_space"),
                ]
            )
        )
        ok &= _check("duplicate ros_prefix raises ValueError", False)
    except ValueError:
        ok &= _check("duplicate ros_prefix raises ValueError", True)

    if not ok:
        print("\n[bimanual2] FAILED", flush=True)
        return 1
    print("\n[bimanual2] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
