"""patch_leader_skip_reset_when_stopping(): once an auto-stop has been
requested (_request_stop() -- HILSERL_MAX_EPISODES reached), the actor's
act_with_policy() loop still calls online_env.reset() once more for an
episode that will never run, and that reset used to make real
switch_controller/arms_resetter service calls that raced the shutdown
teardown and crashed (round-3 DAgger, 2026-09-29: "arms_resetter did not
report 'homed'" right after the 10th/last episode saved). This patch must
make perform_episode_reset_and_wait() a pure no-op once stopping, and leave
it fully working beforehand. No ROS2 service/hardware needed -- a stub
switch client/arms_resetter whose methods must NOT be called once stopping.

Run: python3 -m lerobot_control.hilserl._stage7j_skip_reset_after_stop_requested_test
"""

import sys

import rclpy
from rclpy.node import Node

from lerobot_control.hilserl import patches
from lerobot_control.hilserl.patches import patch_leader_skip_reset_when_stopping
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
        self.calls = 0

    def reset_and_wait(self, timeout_s=15.0):
        self.calls += 1
        return self.ok


def main() -> int:
    rclpy.init(args=None)
    node = Node("stage7j_test_node")

    config = AnvilOpenArmLeaderConfig(urdf_path="/workspace/configs/hilserl/urdf/openarm_teleop_kinematics.urdf")
    leader = AnvilOpenArmLeader(config)
    leader._node = node
    leader._is_connected = True

    patch_leader_skip_reset_when_stopping()
    patch_leader_skip_reset_when_stopping()  # idempotent

    before_calls, after_calls = [], []

    # --- not stopping: real reset must still run exactly as before ---
    patches._auto_stop_state["stop_requested"] = False
    leader._switch_client = _RecordingSwitchClient()
    leader._arms_resetter = _RecordingArmsResetter(ok=True)
    leader.perform_episode_reset_and_wait(
        before_homing=lambda: before_calls.append(1), after_homed=lambda: after_calls.append(1)
    )
    if leader._switch_client.calls != ["switch_to_teleop", "switch_to_policy"] or leader._arms_resetter.calls != 1:
        print(f"[FAIL] not-stopping: expected a real reset, got switch={leader._switch_client.calls} "
              f"arms_resetter_calls={leader._arms_resetter.calls}", flush=True)
        return 1
    if before_calls != [1]:
        print("[FAIL] not-stopping: before_homing must still fire", flush=True)
        return 1
    print("[OK] not stopping: perform_episode_reset_and_wait() still does the real switch+home", flush=True)

    # --- stopping: must be a pure no-op, no ROS calls of any kind ---
    patches._auto_stop_state["stop_requested"] = True
    leader._switch_client = _RecordingSwitchClient()
    leader._arms_resetter = _RecordingArmsResetter(ok=True)
    before_calls.clear()
    after_calls.clear()
    leader.perform_episode_reset_and_wait(
        before_homing=lambda: before_calls.append(1), after_homed=lambda: after_calls.append(1)
    )
    if leader._switch_client.calls or leader._arms_resetter.calls:
        print(f"[FAIL] stopping: expected zero ROS calls, got switch={leader._switch_client.calls} "
              f"arms_resetter_calls={leader._arms_resetter.calls}", flush=True)
        return 1
    if before_calls or after_calls:
        print("[FAIL] stopping: before_homing/after_homed must not fire either -- nothing should touch the arm", flush=True)
        return 1
    print("[OK] stopping: perform_episode_reset_and_wait() is a pure no-op -- no switch/arms_resetter/callback calls", flush=True)

    # --- also confirm a real arms_resetter failure still raises when NOT stopping ---
    patches._auto_stop_state["stop_requested"] = False
    leader._switch_client = _RecordingSwitchClient()
    leader._arms_resetter = _RecordingArmsResetter(ok=False)
    try:
        leader.perform_episode_reset_and_wait()
        print("[FAIL] not-stopping: a real arms_resetter failure must still raise", flush=True)
        return 1
    except RuntimeError:
        print("[OK] not stopping: a real arms_resetter failure still raises (this patch didn't swallow it)", flush=True)

    print("[stage7j] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
