"""Isolated test for the stale-heartbeat bug found in the first full session
with homing: the follower's heartbeat kept republishing the last PRE-reset
command through homing and right through re-activation of
follower_r_forward_position_controller, overwriting the bumpless home-pose
message and commanding a large step from home back toward the old pose. The
arm stopped responding within ~1.5s of re-activation.

Pure logic, no ROS2/hardware: bare instances with injected state, same
bypass-connect() pattern as _stage6b/_stage6f.

Run: python3 -m lerobot_control.hilserl._stage6g_reset_heartbeat_test
"""

import sys

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from lerobot_control.hilserl.robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig
from lerobot_control.hilserl.teleop_anvil_openarm_leader import AnvilOpenArmLeader, AnvilOpenArmLeaderConfig


def _joint_state(pos: float) -> JointState:
    msg = JointState()
    msg.name = [f"follower_r_joint{i}" for i in range(1, 8)] + ["follower_r_finger_joint1"]
    msg.position = [pos] * 8
    return msg


def main() -> int:
    cfg = AnvilOpenArmFollowerConfig(
        ros_prefix="follower_r", camera_topics={"wrist_r": "/cam_wrist_r/image_raw/compressed"}
    )

    print("[stage6g] reseed replaces a stale pre-reset command with the real post-homing pose...", flush=True)
    robot = AnvilOpenArmFollower(cfg)
    stale = Float64MultiArray()
    stale.data = [0.9] * 8  # pose at the end of the previous episode
    robot._last_sent_msg = stale
    robot._obs_manager.update_joint_state(_joint_state(0.1))  # arm is now at home
    robot.reseed_heartbeat_after_reset()
    if list(robot._last_sent_msg.data) != [0.1] * 8:
        print(f"[FAIL] heartbeat still holds a stale/incorrect command: {list(robot._last_sent_msg.data)}", flush=True)
        return 1
    print("[OK] heartbeat now holds the actual home pose, not the stale pre-reset command", flush=True)

    print("[stage6g] if the pose can't be read, the heartbeat must go SILENT, never stay stale...", flush=True)
    robot2 = AnvilOpenArmFollower(cfg)
    robot2._last_sent_msg = stale  # stale pre-reset command, and no joint state available
    robot2.reseed_heartbeat_after_reset()
    if robot2._last_sent_msg is not None:
        print("[FAIL] a stale command survived a failed reseed", flush=True)
        return 1
    print("[OK] failed reseed leaves the heartbeat empty (silent), not stale", flush=True)

    print("[stage6g] clear_heartbeat() empties it (used before homing)...", flush=True)
    robot3 = AnvilOpenArmFollower(cfg)
    robot3._last_sent_msg = stale
    robot3.clear_heartbeat()
    if robot3._last_sent_msg is not None:
        print("[FAIL] clear_heartbeat() did not clear", flush=True)
        return 1
    print("[OK] cleared", flush=True)

    print("[stage6g] leader orchestration: before_homing first, after_homed BEFORE the policy "
          "controller is re-activated...", flush=True)
    events: list[str] = []

    class _Switch:
        def switch_to_teleop(self):
            events.append("switch_to_teleop")
            return True

        def switch_to_policy(self):
            events.append("switch_to_policy")
            return True

    class _Resetter:
        def reset_and_wait(self, timeout_s):
            events.append("homing")
            return True

    leader = AnvilOpenArmLeader(
        AnvilOpenArmLeaderConfig(urdf_path="/workspace/configs/hilserl/urdf/openarm_teleop_kinematics.urdf")
    )
    leader._is_connected = True
    leader._switch_client = _Switch()
    leader._arms_resetter = _Resetter()
    leader.perform_episode_reset_and_wait(
        timeout_s=5.0,
        before_homing=lambda: events.append("before_homing"),
        after_homed=lambda: events.append("after_homed"),
    )
    expected = ["before_homing", "switch_to_teleop", "homing", "after_homed", "switch_to_policy"]
    if events != expected:
        print(f"[FAIL] wrong order: {events}, expected {expected}", flush=True)
        return 1
    print(f"[OK] order correct: {events}", flush=True)

    print("[stage6g] after_homed must NOT run if homing fails...", flush=True)
    events.clear()

    class _FailingResetter:
        def reset_and_wait(self, timeout_s):
            events.append("homing")
            return False

    leader._arms_resetter = _FailingResetter()
    try:
        leader.perform_episode_reset_and_wait(
            timeout_s=5.0,
            before_homing=lambda: events.append("before_homing"),
            after_homed=lambda: events.append("after_homed"),
        )
        print("[FAIL] expected a RuntimeError", flush=True)
        return 1
    except RuntimeError:
        pass
    if "after_homed" in events or "switch_to_policy" in events:
        print(f"[FAIL] reseed/re-activation ran after a failed homing: {events}", flush=True)
        return 1
    print(f"[OK] failed homing stops before any reseed or policy re-activation: {events}", flush=True)

    print("[stage6g] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
