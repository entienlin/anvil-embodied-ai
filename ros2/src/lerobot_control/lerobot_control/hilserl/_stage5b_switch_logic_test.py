"""Isolated test for SwitchServiceClient._switch_to()'s state-aware request
building, added after the real Stage 5 attempt hit exactly the failure
mode this logic now avoids: "Controller with name 'teleop_controller' can
not be deactivated since it is not active."

Monkeypatches _get_controller_states() and _call_switch() so this needs no
real controller_manager, no mock-robot, no network -- just a bare rclpy
Node (required by SwitchServiceClient.__init__ for create_publisher/
create_subscription/create_client, but never actually used against a real
service here).

Run: python3 -m lerobot_control.hilserl._stage5b_switch_logic_test
"""

import sys

import rclpy
from rclpy.node import Node

from lerobot_control.hilserl.switch_service_client import SwitchServiceClient


def _make_client(node: Node) -> SwitchServiceClient:
    return SwitchServiceClient(
        node,
        policy_controller="follower_r_forward_position_controller",
        teleop_controller="teleop_controller",
    )


def _check(label: str, states: dict, call_fn, expect_call: bool, expect_activate=None, expect_deactivate=None):
    calls = []

    def fake_get_states():
        return states

    def fake_call_switch(activate, deactivate):
        calls.append((activate, deactivate))
        return True

    node = Node("stage5b_test_node")
    client = _make_client(node)
    client._get_controller_states = fake_get_states
    client._call_switch = fake_call_switch

    result = call_fn(client)

    status = "OK"
    if result is not True:
        status = "FAIL"
    if expect_call and not calls:
        status = "FAIL"
    if not expect_call and calls:
        status = "FAIL"
    if expect_call and calls:
        got_activate, got_deactivate = calls[0]
        if sorted(got_activate) != sorted(expect_activate) or sorted(got_deactivate) != sorted(expect_deactivate):
            status = "FAIL"

    print(f"[{status}] {label}: states={states} -> calls={calls}", flush=True)
    node.destroy_node()
    if status == "FAIL":
        raise AssertionError(f"{label} failed: calls={calls}, result={result}")


def main() -> int:
    rclpy.init(args=None)

    print("[stage5b] real failure scenario: teleop_controller already inactive "
          "(exactly what broke on real hardware)...", flush=True)
    _check(
        "policy already-inactive-teleop",
        {"teleop_controller": "inactive", "follower_r_forward_position_controller": "inactive"},
        lambda c: c.switch_to_policy(),
        expect_call=True,
        expect_activate=["follower_r_forward_position_controller"],
        expect_deactivate=[],  # must NOT include teleop_controller -- it's already inactive
    )

    print("[stage5b] normal expected case: teleop active, policy inactive...", flush=True)
    _check(
        "policy normal-case",
        {"teleop_controller": "active", "follower_r_forward_position_controller": "inactive"},
        lambda c: c.switch_to_policy(),
        expect_call=True,
        expect_activate=["follower_r_forward_position_controller"],
        expect_deactivate=["teleop_controller"],
    )

    print("[stage5b] already in the desired state entirely -- must be a no-op...", flush=True)
    _check(
        "policy no-op",
        {"teleop_controller": "inactive", "follower_r_forward_position_controller": "active"},
        lambda c: c.switch_to_policy(),
        expect_call=False,
    )

    print("[stage5b] reverse direction: switch_to_teleop, normal case...", flush=True)
    _check(
        "teleop normal-case",
        {"teleop_controller": "inactive", "follower_r_forward_position_controller": "active"},
        lambda c: c.switch_to_teleop(),
        expect_call=True,
        expect_activate=["teleop_controller"],
        expect_deactivate=["follower_r_forward_position_controller"],
    )

    print("[stage5b] PASSED", flush=True)
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
