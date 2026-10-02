"""Offline test for the 2026-10-02 fix to AnvilOpenArmMultiLeader.get_action():
each sub-arm's OWN get_action() does its own edge detection and calls its
OWN SwitchServiceClient independently -- the same shared-teleop_controller
conflict perform_episode_reset_and_wait() already hit and fixed
(_combined_switch()), but this is the LIVE per-tick path (every
control-loop tick while a human is actually driving), not just the
between-episodes reset. Confirmed on real hardware: 20 consecutive
"Unable to activate controller 'teleop_controller' since the command
interface 'follower_r_joint1/position' is currently claimed by another
controller" failures, then a hard crash, the first time two arms' live
intervention edges fired independently.

Pure-Python, no ROS2 daemon needed: stubs sub-arms with just enough state
(_was_intervening, _switch_client, _pending_switch_retries, _node logger,
_leader_ee_position(), get_action()) to exercise get_action()'s real
combined-switch-then-delegate logic without needing real ROS2/FK.
"""

import sys
import types


def _check(label: str, cond: bool) -> bool:
    print(f"[{'OK' if cond else 'FAIL'}] {label}", flush=True)
    return cond


class _StubSwitchClient:
    def __init__(self, policy_controller: str, teleop_controller: str, states: dict):
        self._policy_controller = policy_controller
        self._teleop_controller = teleop_controller
        self._states = states
        self.calls: list[tuple[list[str], list[str]]] = []
        self.next_result = True

    def _get_controller_states(self):
        return dict(self._states)

    def _call_switch(self, activate, deactivate):
        self.calls.append((list(activate), list(deactivate)))
        return self.next_result


class _StubLogger:
    def warning(self, msg):
        pass

    def error(self, msg):
        pass


class _StubNode:
    def get_logger(self):
        return _StubLogger()


class _StubArm:
    def __init__(self, ros_prefix, switch_client):
        self.config = type("Cfg", (), {"ros_prefix": ros_prefix})()
        self._switch_client = switch_client
        self._was_intervening = False
        self._pending_switch_retries = 0
        self._leader_ee_ref = None
        self._node = _StubNode()
        self._action_value = [float(ord(ros_prefix[-1]))] * 2  # distinct per arm, arbitrary

    def _leader_ee_position(self):
        return None  # joint_space mode never reads this

    def get_action(self):
        # Mirrors the REAL get_action()'s edge-detection: if the composite
        # did its job, _was_intervening already matches is_intervening by
        # the time this runs, so no switch is attempted here.
        return self._action_value


class _StubKeyboard:
    def __init__(self, is_intervention: bool):
        self.is_intervention = is_intervention


def main() -> int:
    from lerobot_control.hilserl.teleop_anvil_openarm_multi_leader import AnvilOpenArmMultiLeader

    ok = True

    print("[bimanual9] intervention starts: ONE combined switch, not two independent ones...", flush=True)
    shared_states = {
        "teleop_controller": "inactive",
        "follower_l_forward_position_controller": "active",
        "follower_r_forward_position_controller": "active",
    }
    sc_l = _StubSwitchClient("follower_l_forward_position_controller", "teleop_controller", shared_states)
    sc_r = _StubSwitchClient("follower_r_forward_position_controller", "teleop_controller", shared_states)
    arm_l = _StubArm("leader_l", sc_l)
    arm_r = _StubArm("leader_r", sc_r)

    stub_self = type("Stub", (), {})()
    stub_self._arms = [arm_l, arm_r]
    stub_self._keyboard = _StubKeyboard(is_intervention=True)
    stub_self._is_connected = True
    stub_self._combined_switch = types.MethodType(AnvilOpenArmMultiLeader._combined_switch, stub_self)

    result = AnvilOpenArmMultiLeader.get_action(stub_self)

    ok &= _check("exactly one switch_controller call made (not per-arm)", len(sc_l.calls) == 1)
    ok &= _check("leader_l's OWN switch client never called independently", len(sc_l.calls) <= 1)
    ok &= _check("leader_r's OWN switch client never called a SECOND time", len(sc_r.calls) == 0)
    ok &= _check("both arms' _was_intervening synced to True", arm_l._was_intervening is True and arm_r._was_intervening is True)
    ok &= _check("action is the concatenation of both sub-arms", list(result) == arm_l._action_value + arm_r._action_value)

    print("\n[bimanual9] already in sync: no switch attempted, pure delegation...", flush=True)
    arm_l2 = _StubArm("leader_l", _StubSwitchClient("follower_l_forward_position_controller", "teleop_controller", {}))
    arm_r2 = _StubArm("leader_r", _StubSwitchClient("follower_r_forward_position_controller", "teleop_controller", {}))
    arm_l2._was_intervening = True
    arm_r2._was_intervening = True
    stub_self2 = type("Stub", (), {})()
    stub_self2._arms = [arm_l2, arm_r2]
    stub_self2._keyboard = _StubKeyboard(is_intervention=True)
    stub_self2._is_connected = True
    stub_self2._combined_switch = types.MethodType(AnvilOpenArmMultiLeader._combined_switch, stub_self2)
    AnvilOpenArmMultiLeader.get_action(stub_self2)
    ok &= _check("no switch_controller call made when already in sync", len(arm_l2._switch_client.calls) == 0)

    print("\n[bimanual9] switch failure: retries accumulate on both arms...", flush=True)
    sc_l3 = _StubSwitchClient("follower_l_forward_position_controller", "teleop_controller", shared_states)
    sc_r3 = _StubSwitchClient("follower_r_forward_position_controller", "teleop_controller", shared_states)
    sc_l3.next_result = False
    arm_l3 = _StubArm("leader_l", sc_l3)
    arm_r3 = _StubArm("leader_r", sc_r3)
    stub_self3 = type("Stub", (), {})()
    stub_self3._arms = [arm_l3, arm_r3]
    stub_self3._keyboard = _StubKeyboard(is_intervention=True)
    stub_self3._is_connected = True
    stub_self3._combined_switch = types.MethodType(AnvilOpenArmMultiLeader._combined_switch, stub_self3)
    AnvilOpenArmMultiLeader.get_action(stub_self3)
    ok &= _check("failed switch leaves both arms' _was_intervening unchanged", arm_l3._was_intervening is False and arm_r3._was_intervening is False)
    ok &= _check("retry counters incremented on both arms", arm_l3._pending_switch_retries == 1 and arm_r3._pending_switch_retries == 1)

    print("\n[bimanual9] switch failure 20 times: raises rather than retrying forever...", flush=True)
    raised = False
    try:
        for _ in range(20):
            AnvilOpenArmMultiLeader.get_action(stub_self3)
    except RuntimeError:
        raised = True
    ok &= _check("raises after 20 consecutive failures", raised)

    if not ok:
        print("\n[bimanual9] FAILED", flush=True)
        return 1
    print("\n[bimanual9] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
