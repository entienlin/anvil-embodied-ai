"""Isolated test for get_action()'s switch-failure retry handling, added
after a real switch_to_policy() failure during a real intervention
give-back was silently ignored: get_action() used to call
switch_to_teleop()/switch_to_policy() and never check their boolean return
value, so a failed real switch_controller call still flipped our internal
_was_intervening as if it had succeeded -- leaving the follower held by
neither controller while our own code believed the handoff was done
(reported as the arm going "dead" right after give-back).

No ROS2 service/hardware needed -- a stub switch client that fails N times
then succeeds (or always fails), plus a bare rclpy Node (only used for
_handle_switch_edge's warning/error log calls).

Run: python3 -m lerobot_control.hilserl._stage5e_switch_retry_test
"""

import sys

import rclpy
from rclpy.node import Node

from lerobot_control.hilserl.teleop_anvil_openarm_leader import AnvilOpenArmLeader, AnvilOpenArmLeaderConfig


class _FlakySwitchClient:
    def __init__(self, fail_count: int):
        """Fails the first `fail_count` calls (to EITHER method), then
        succeeds forever after. fail_count=-1 means always fail."""
        self.fail_count = fail_count
        self.calls = 0

    def _attempt(self) -> bool:
        self.calls += 1
        if self.fail_count < 0:
            return False
        return self.calls > self.fail_count

    def switch_to_teleop(self) -> bool:
        return self._attempt()

    def switch_to_policy(self) -> bool:
        return self._attempt()


def main() -> int:
    rclpy.init(args=None)
    node = Node("stage5e_test_node")

    config = AnvilOpenArmLeaderConfig(urdf_path="/workspace/configs/hilserl/urdf/openarm_teleop_kinematics.urdf")

    print("[stage5e] switch fails twice, then succeeds on the third attempt "
          "-- must retry, not silently accept the failed state...", flush=True)
    leader = AnvilOpenArmLeader(config)
    leader._node = node
    leader._switch_client = _FlakySwitchClient(fail_count=2)
    leader._is_connected = True

    leader._keyboard._is_intervention = True
    for i in range(2):
        action = leader.get_action()
        if leader._was_intervening is not False:
            print(f"[FAIL] tick {i}: _was_intervening should still be False while switch keeps failing", flush=True)
            return 1
        if action != {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0}:
            print(f"[FAIL] tick {i}: expected zero-delta action while unswitched, got {action}", flush=True)
            return 1
    print("[OK] stayed in the pre-switch state (zero-delta, _was_intervening=False) "
          "for both failed attempts", flush=True)

    action = leader.get_action()  # third attempt succeeds
    if leader._was_intervening is not True:
        print("[FAIL] expected _was_intervening=True after the successful 3rd attempt", flush=True)
        return 1
    if leader._pending_switch_retries != 0:
        print(f"[FAIL] expected retry counter reset to 0 after success, got {leader._pending_switch_retries}", flush=True)
        return 1
    print("[OK] switch succeeded on retry, state committed, retry counter reset", flush=True)

    print("[stage5e] switch always fails -- must raise after MAX_CONSECUTIVE_SWITCH_FAILURES, "
          "not retry forever...", flush=True)
    leader2 = AnvilOpenArmLeader(config)
    leader2._node = node
    leader2._switch_client = _FlakySwitchClient(fail_count=-1)
    leader2._is_connected = True
    leader2._keyboard._is_intervention = True

    try:
        for _ in range(100):
            leader2.get_action()
        print("[FAIL] expected a RuntimeError after enough consecutive failures, got none", flush=True)
        return 1
    except RuntimeError as e:
        print(f"[OK] correctly gave up and raised: {e}", flush=True)

    node.destroy_node()
    rclpy.shutdown()
    print("[stage5e] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
