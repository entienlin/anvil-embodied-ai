"""Stage 4 test: AnvilOpenArmLeader's FK-based delta math, in isolation.

Deliberately does NOT call connect() -- that would require a live
controller_manager/switch_controller service (Stage 1's job, not this
one) and a real X11 DISPLAY for the keyboard listener. Instead this
directly injects synthetic JointState messages and a stub switch client,
so it can run as a pure computation test: no rclpy.init(), no ROS2 daemon,
no mock-robot, no network. Only real dependency is the resolved URDF
(RobotKinematics/placo), already proven correct in Stage 0.

Validates the things Stage 0 could NOT check (Stage 0 only exercised FK/IK
directly, not this class's own delta-tracking logic):
  - first intervention tick returns a zero delta and establishes the
    reference pose, rather than jumping based on an undefined prior ref
  - subsequent ticks compute delta = FK(current) - FK(previous tick),
    matching an independently hand-computed FK diff exactly
  - tracking is RELATIVE (frame-to-frame), not cumulative from the first
    intervention pose -- the one correctness property most likely to be
    silently wrong (this is exactly what the module docstring's "not
    leader-minus-follower" reasoning depends on)
  - give-back (intervention -> not intervening) resets the reference and
    returns to zero deltas

Run: python3 -m lerobot_control.hilserl._stage4_leader_test
"""

import sys

import numpy as np
from sensor_msgs.msg import JointState

from lerobot_control.hilserl.teleop_anvil_openarm_leader import AnvilOpenArmLeader, AnvilOpenArmLeaderConfig


class _StubSwitchClient:
    """Always succeeds -- this test covers get_action()'s delta math on the
    SUCCESS path. Failure/retry handling has its own dedicated test,
    _stage5e_switch_retry_test.py, added after a real switch failure was
    silently ignored on real hardware (get_action() used to not check the
    return value at all)."""

    def __init__(self):
        self.calls = []

    def switch_to_teleop(self):
        self.calls.append("teleop")
        return True

    def switch_to_policy(self):
        self.calls.append("policy")
        return True


def _make_joint_state(leader: AnvilOpenArmLeader, joint_deg: np.ndarray) -> JointState:
    msg = JointState()
    msg.name = list(leader.joint_names)
    msg.position = list(np.deg2rad(joint_deg))
    return msg


def _assert_close(label: str, got, expected, tol=1e-6) -> None:
    got_arr = np.array([got["delta_x"], got["delta_y"], got["delta_z"]])
    exp_arr = np.array(expected)
    err = np.abs(got_arr - exp_arr).max()
    status = "OK" if err < tol else "FAIL"
    print(f"[{status}] {label}: got={got_arr} expected={exp_arr} max_err={err:.2e}", flush=True)
    if err >= tol:
        raise AssertionError(f"{label} mismatch: got {got_arr}, expected {exp_arr}")


def main() -> int:
    config = AnvilOpenArmLeaderConfig(
        urdf_path="/workspace/configs/hilserl/urdf/openarm_teleop_kinematics.urdf",
        step_sizes={"x": 0.02, "y": 0.02, "z": 0.02},
    )
    leader = AnvilOpenArmLeader(config)
    stub_switch = _StubSwitchClient()
    leader._switch_client = stub_switch
    leader._is_connected = True

    pose_a = np.zeros(7)
    pose_b = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # joint1 +5deg
    pose_c = np.array([5.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # + joint2 -3deg

    # Independently hand-compute the expected FK positions (same kinematics
    # object, but a completely separate call path from get_action()'s
    # internal one -- this is what makes the comparison meaningful).
    T_a = leader._kinematics.forward_kinematics(pose_a)
    T_b = leader._kinematics.forward_kinematics(pose_b)
    T_c = leader._kinematics.forward_kinematics(pose_c)
    pos_a, pos_b, pos_c = T_a[:3, 3], T_b[:3, 3], T_c[:3, 3]
    steps = np.array([config.step_sizes["x"], config.step_sizes["y"], config.step_sizes["z"]])

    print("[stage4] intervention starts (edge transition)...", flush=True)
    leader._keyboard._is_intervention = True
    leader._on_joint_state(_make_joint_state(leader, pose_a))
    action1 = leader.get_action()
    _assert_close("tick1 (edge transition, no prior ref)", action1, [0.0, 0.0, 0.0])
    assert stub_switch.calls == ["teleop"], f"expected switch_to_teleop() on edge, got {stub_switch.calls}"
    print("[stage4] switch_to_teleop() called correctly on intervention start", flush=True)

    print("[stage4] tick2: leader moves A -> B...", flush=True)
    leader._on_joint_state(_make_joint_state(leader, pose_b))
    action2 = leader.get_action()
    _assert_close("tick2 (A->B delta)", action2, (pos_b - pos_a) / steps)

    print("[stage4] tick3: leader moves B -> C (must be relative to B, not A)...", flush=True)
    leader._on_joint_state(_make_joint_state(leader, pose_c))
    action3 = leader.get_action()
    _assert_close("tick3 (B->C delta, NOT C-A)", action3, (pos_c - pos_b) / steps)
    # Explicitly confirm it's not the (wrong) cumulative-from-origin answer.
    wrong_cumulative = (pos_c - pos_a) / steps
    got_arr = np.array([action3["delta_x"], action3["delta_y"], action3["delta_z"]])
    if np.abs(got_arr - wrong_cumulative).max() < 1e-6:
        raise AssertionError("tick3 matches the WRONG cumulative-from-origin delta, not frame-to-frame")
    print("[stage4] confirmed tick3 is NOT the cumulative-from-origin delta", flush=True)

    print("[stage4] give-back: intervention ends (edge transition)...", flush=True)
    leader._keyboard._is_intervention = False
    action4 = leader.get_action()
    _assert_close("tick4 (give-back)", action4, [0.0, 0.0, 0.0])
    assert stub_switch.calls == ["teleop", "policy"], f"expected switch_to_policy() on give-back, got {stub_switch.calls}"
    assert leader._leader_ee_ref is None, "reference must be cleared on give-back"
    print("[stage4] switch_to_policy() called correctly on give-back, reference cleared", flush=True)

    print("[stage4] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
