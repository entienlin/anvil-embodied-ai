"""Isolated test for get_teleop_events()'s auto-give-back-on-label fix,
added for multi-episode collection: pressing 's'/'f' while still
intervening (without giving back via space first) used to leave the
follower held by teleop_controller through the episode's reset wait --
originally only fixed at the very end of a script (disconnect()), but a
multi-episode session hits this same boundary on EVERY episode.

No ROS2 service/hardware needed -- a stub switch client + a bare rclpy
Node (only used for warning/error log calls inside _handle_switch_edge).

Run: python3 -m lerobot_control.hilserl._stage5g_auto_giveback_on_label_test
"""

import sys

import rclpy
from lerobot.teleoperators.utils import TeleopEvents
from rclpy.node import Node

from lerobot_control.hilserl.teleop_anvil_openarm_leader import AnvilOpenArmLeader, AnvilOpenArmLeaderConfig


class _StubSwitchClient:
    def __init__(self):
        self.calls = []

    def switch_to_teleop(self):
        self.calls.append("teleop")
        return True

    def switch_to_policy(self):
        self.calls.append("policy")
        return True


def main() -> int:
    rclpy.init(args=None)
    node = Node("stage5g_test_node")

    config = AnvilOpenArmLeaderConfig(urdf_path="/workspace/configs/hilserl/urdf/openarm_teleop_kinematics.urdf")
    leader = AnvilOpenArmLeader(config)
    leader._node = node
    stub = _StubSwitchClient()
    leader._switch_client = stub
    leader._is_connected = True

    print("[stage5g] simulate: user pressed space (intervention starts)...", flush=True)
    leader._keyboard._is_intervention = True
    leader.get_action()
    if stub.calls != ["teleop"]:
        print(f"[FAIL] expected switch_to_teleop() after space, got {stub.calls}", flush=True)
        return 1

    print("[stage5g] simulate: user presses 's' WITHOUT pressing space to give back first...", flush=True)
    leader._keyboard._success = True
    events = leader.get_teleop_events()

    if stub.calls != ["teleop", "policy"]:
        print(f"[FAIL] expected switch_to_policy() to fire automatically on 's', got {stub.calls}", flush=True)
        return 1
    if leader._was_intervening is not False:
        print("[FAIL] expected _was_intervening to be forced False after auto-give-back", flush=True)
        return 1
    if not events[TeleopEvents.SUCCESS]:
        print("[FAIL] success label should still be reported even though we auto-gave-back", flush=True)
        return 1
    print("[OK] switch_to_policy() fired automatically on 's', state corrected, "
          "success label still reported", flush=True)

    print("[stage5g] normal case: give back via space BEFORE pressing 's' -- "
          "must not double-switch...", flush=True)
    leader2 = AnvilOpenArmLeader(config)
    leader2._node = node
    stub2 = _StubSwitchClient()
    leader2._switch_client = stub2
    leader2._is_connected = True

    leader2._keyboard._is_intervention = True
    leader2.get_action()  # -> teleop
    leader2._keyboard._is_intervention = False
    leader2.get_action()  # -> policy (proper give-back via space)
    leader2._keyboard._success = True
    leader2.get_teleop_events()

    if stub2.calls != ["teleop", "policy"]:
        print(f"[FAIL] expected exactly one give-back (via space), got extra calls: {stub2.calls}", flush=True)
        return 1
    print("[OK] no redundant switch when give-back already happened properly via space", flush=True)

    node.destroy_node()
    rclpy.shutdown()
    print("[stage5g] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
