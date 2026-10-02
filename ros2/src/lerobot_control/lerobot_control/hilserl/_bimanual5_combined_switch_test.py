"""Offline test for AnvilOpenArmMultiLeader._combined_switch() -- the fix
for a real 2026-10-02 real-hardware failure: teleop_controller is ONE
shared controller instance spanning every configured arm slot, so
activating it for "just" leader_l while follower_r_forward_position_controller
is still active fails ("command interface 'follower_r_joint1/position' is
currently claimed by another controller"), even though the switch request
never mentions follower_r. The old code looped per-arm, each arm's own
SwitchServiceClient issuing an independent switch_controller call --
_combined_switch() replaces that with ONE call covering every arm's policy
controller together with the shared teleop_controller.

Pure-Python, no rclpy/ROS2 daemon needed: calls _combined_switch() as an
unbound method against a bare stub object (not a real AnvilOpenArmMultiLeader,
which needs real AnvilOpenArmLeader sub-instances to construct) -- same
"test the logic in isolation" approach as _stage5b_switch_logic_test.py
uses for the single-arm SwitchServiceClient._switch_to().
"""

import sys


def _check(label: str, cond: bool) -> bool:
    print(f"[{'OK' if cond else 'FAIL'}] {label}", flush=True)
    return cond


class _StubSwitchClient:
    def __init__(self, policy_controller: str, teleop_controller: str):
        self._policy_controller = policy_controller
        self._teleop_controller = teleop_controller
        self.calls: list[tuple[list[str], list[str]]] = []
        self._states: dict[str, str] = {}
        self._next_call_result = True

    def _get_controller_states(self):
        return dict(self._states)

    def _call_switch(self, activate, deactivate):
        self.calls.append((list(activate), list(deactivate)))
        return self._next_call_result


class _StubArm:
    def __init__(self, switch_client: _StubSwitchClient):
        self._switch_client = switch_client


def main() -> int:
    from lerobot_control.hilserl.teleop_anvil_openarm_multi_leader import AnvilOpenArmMultiLeader

    ok = True

    print("[bimanual5] real failure scenario: teleop inactive, follower_r_fpc still active...", flush=True)
    sc_l = _StubSwitchClient("follower_l_forward_position_controller", "teleop_controller")
    sc_r = _StubSwitchClient("follower_r_forward_position_controller", "teleop_controller")
    shared_states = {
        "teleop_controller": "inactive",
        "follower_l_forward_position_controller": "inactive",
        "follower_r_forward_position_controller": "active",
    }
    sc_l._states = shared_states
    sc_r._states = shared_states
    stub_self = type("Stub", (), {"_arms": [_StubArm(sc_l), _StubArm(sc_r)]})()

    result = AnvilOpenArmMultiLeader._combined_switch(stub_self, want_teleop=True)
    ok &= _check("combined switch to teleop succeeds", result is True)
    ok &= _check("exactly one switch_controller call made (not two)", len(sc_l.calls) == 1)
    if sc_l.calls:
        activate, deactivate = sc_l.calls[0]
        ok &= _check("activates teleop_controller", activate == ["teleop_controller"])
        ok &= _check(
            "deactivates ONLY follower_r_fpc (already-inactive follower_l_fpc omitted)",
            deactivate == ["follower_r_forward_position_controller"],
        )

    print("\n[bimanual5] already-correct state is a no-op (zero switch_controller calls)...", flush=True)
    sc_l2 = _StubSwitchClient("follower_l_forward_position_controller", "teleop_controller")
    sc_r2 = _StubSwitchClient("follower_r_forward_position_controller", "teleop_controller")
    already_teleop_states = {
        "teleop_controller": "active",
        "follower_l_forward_position_controller": "inactive",
        "follower_r_forward_position_controller": "inactive",
    }
    sc_l2._states = already_teleop_states
    sc_r2._states = already_teleop_states
    stub_self2 = type("Stub", (), {"_arms": [_StubArm(sc_l2), _StubArm(sc_r2)]})()
    result2 = AnvilOpenArmMultiLeader._combined_switch(stub_self2, want_teleop=True)
    ok &= _check("no-op switch still reports success", result2 is True)
    ok &= _check("no switch_controller call made", len(sc_l2.calls) == 0)

    print("\n[bimanual5] give-back direction: both policy controllers need activating...", flush=True)
    sc_l3 = _StubSwitchClient("follower_l_forward_position_controller", "teleop_controller")
    sc_r3 = _StubSwitchClient("follower_r_forward_position_controller", "teleop_controller")
    teleop_active_states = {
        "teleop_controller": "active",
        "follower_l_forward_position_controller": "inactive",
        "follower_r_forward_position_controller": "inactive",
    }
    sc_l3._states = teleop_active_states
    sc_r3._states = teleop_active_states
    stub_self3 = type("Stub", (), {"_arms": [_StubArm(sc_l3), _StubArm(sc_r3)]})()
    result3 = AnvilOpenArmMultiLeader._combined_switch(stub_self3, want_teleop=False)
    ok &= _check("combined switch to policy succeeds", result3 is True)
    ok &= _check("exactly one switch_controller call made", len(sc_l3.calls) == 1)
    if sc_l3.calls:
        activate, deactivate = sc_l3.calls[0]
        ok &= _check(
            "activates BOTH policy controllers together",
            sorted(activate) == ["follower_l_forward_position_controller", "follower_r_forward_position_controller"],
        )
        ok &= _check("deactivates the shared teleop_controller", deactivate == ["teleop_controller"])

    if not ok:
        print("\n[bimanual5] FAILED", flush=True)
        return 1
    print("\n[bimanual5] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
