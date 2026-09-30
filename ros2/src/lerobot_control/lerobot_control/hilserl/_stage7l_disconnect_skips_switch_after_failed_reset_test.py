"""Real-hardware finding (2026-09-30): AnvilOpenArmLeader.disconnect() (run
via atexit on ANY process exit, including a crash) used to unconditionally
call switch_to_policy() -- even when perform_episode_reset_and_wait() had
just deliberately raised to leave the arm safely on teleop_controller after
a failed/timed-out reset. That silently overrode the safety decision and
handed an unverified, un-homed arm to policy control. Confirmed live: an
arms_resetter timeout left the arm exactly this way.

This test confirms the fix: _reset_in_progress_unsafe gates disconnect()'s
give-back, set the moment a reset switches to teleop and cleared only once
that SAME reset confirms success. No ROS2/hardware needed -- a bare Node +
stub switch/arms_resetter clients, mirroring _stage7j's pattern.

Run: python3 -m lerobot_control.hilserl._stage7l_disconnect_skips_switch_after_failed_reset_test
"""

import sys

import rclpy
from rclpy.node import Node

from lerobot_control.hilserl.teleop_anvil_openarm_leader import AnvilOpenArmLeader, AnvilOpenArmLeaderConfig


class _RecordingSwitchClient:
    def __init__(self):
        self.calls = []

    def switch_to_teleop(self):
        self.calls.append("switch_to_teleop")
        return True

    def switch_to_policy(self):
        self.calls.append("switch_to_policy")
        return True


class _RecordingArmsResetter:
    def __init__(self, ok: bool):
        self.ok = ok

    def reset_and_wait(self, timeout_s=15.0):
        return self.ok


def main() -> int:
    rclpy.init(args=None)
    node = Node("stage7l_test_node")
    config = AnvilOpenArmLeaderConfig(urdf_path="/workspace/configs/hilserl/urdf/openarm_teleop_kinematics.urdf")

    # --- failed reset: disconnect() must NOT hand control back to policy ---
    leader = AnvilOpenArmLeader(config)
    leader._node = node
    leader._is_connected = True
    leader._switch_client = _RecordingSwitchClient()
    leader._arms_resetter = _RecordingArmsResetter(ok=False)

    try:
        leader.perform_episode_reset_and_wait()
        print("[FAIL] a failed arms_resetter should raise", flush=True)
        return 1
    except RuntimeError:
        pass

    if not leader._reset_in_progress_unsafe:
        print("[FAIL] _reset_in_progress_unsafe must be True after a failed reset", flush=True)
        return 1
    print("[OK] failed reset leaves _reset_in_progress_unsafe=True", flush=True)

    leader._is_connected = True  # disconnect() flips this; reset for a clean call
    leader.disconnect()
    if "switch_to_policy" in leader._switch_client.calls:
        print(f"[FAIL] disconnect() must not switch back to policy after a failed reset, got {leader._switch_client.calls}", flush=True)
        return 1
    print("[OK] disconnect() skips switch_to_policy() after a failed reset -- arm stays on teleop_controller", flush=True)

    # --- successful reset: disconnect() SHOULD behave exactly as before ---
    leader2 = AnvilOpenArmLeader(config)
    leader2._node = node
    leader2._is_connected = True
    leader2._switch_client = _RecordingSwitchClient()
    leader2._arms_resetter = _RecordingArmsResetter(ok=True)

    leader2.perform_episode_reset_and_wait()
    if leader2._reset_in_progress_unsafe:
        print("[FAIL] _reset_in_progress_unsafe must be False after a successful reset", flush=True)
        return 1
    print("[OK] successful reset clears _reset_in_progress_unsafe", flush=True)

    leader2._switch_client.calls.clear()
    leader2.disconnect()
    if leader2._switch_client.calls != ["switch_to_policy"]:
        print(f"[FAIL] disconnect() must still give back to policy normally after a successful reset, got {leader2._switch_client.calls}", flush=True)
        return 1
    print("[OK] disconnect() still gives back to policy normally after a successful reset (unchanged behavior)", flush=True)

    print("[stage7l] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
