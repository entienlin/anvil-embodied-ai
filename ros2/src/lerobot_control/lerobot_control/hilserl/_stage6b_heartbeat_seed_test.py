"""Isolated test for AnvilOpenArmFollower's connect()-time heartbeat seed,
added after the 6th real "follower dead" occurrence -- localized (via the
devbox's motor-status log, cross-referenced with the actor's own "Policy
FPS 1.1 below required 10 at step 0" warning at the exact same timestamp)
to the gap between connect() completing and the actor's first real
policy-inference-driven send_action() call, which took ~6s in practice --
a cold start (model/GPU warmup) the pure-recording path never hits, since
teleop-driven send_action() calls start immediately. The heartbeat thread
added earlier only republishes self._last_sent_msg, which stayed None
until the first successful send_action() -- zero protection during this
specific window.

No ROS2 service/real hardware needed -- constructs a bare
AnvilOpenArmFollower instance (bypassing connect()) with a fake
ObservationManager-like joint state already present, and calls
_seed_heartbeat_from_current_position() directly.

Run: python3 -m lerobot_control.hilserl._stage6b_heartbeat_seed_test
"""

import sys

from sensor_msgs.msg import JointState

from lerobot_control.hilserl.robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig


def main() -> int:
    config = AnvilOpenArmFollowerConfig(
        ros_prefix="follower_r",
        camera_topics={"wrist_r": "/cam_wrist_r/image_raw/compressed"},
    )
    robot = AnvilOpenArmFollower(config)

    print("[stage6b] before connect(): heartbeat must have nothing to send yet...", flush=True)
    if robot._last_sent_msg is not None:
        print(f"[FAIL] expected _last_sent_msg to start None, got {robot._last_sent_msg}", flush=True)
        return 1
    print("[OK] _last_sent_msg starts None, matching pre-fix behavior before any seeding", flush=True)

    print("[stage6b] simulating a fresh joint_states message arriving (as connect() would see)...", flush=True)
    msg = JointState()
    msg.name = [f"follower_r_joint{i}" for i in range(1, 8)] + ["follower_r_finger_joint1"]
    msg.position = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05]
    robot._obs_manager.update_joint_state(msg)

    robot._seed_heartbeat_from_current_position()

    if robot._last_sent_msg is None:
        print("[FAIL] expected _last_sent_msg to be seeded after "
              "_seed_heartbeat_from_current_position(), got None", flush=True)
        return 1
    expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.05]
    got = list(robot._last_sent_msg.data)
    if got != expected:
        print(f"[FAIL] expected seeded hold command {expected} (7 joints + gripper), got {got}", flush=True)
        return 1
    if robot._last_publish_time is None:
        print("[FAIL] expected _last_publish_time to be set alongside the seed", flush=True)
        return 1
    print(f"[OK] heartbeat seeded with the real current position {got} "
          "(7 real joint values + gripper) -- protected from the very first tick, "
          "before send_action() is ever called", flush=True)

    print("[stage6b] missing joint data must not crash, just skip seeding...", flush=True)
    robot2 = AnvilOpenArmFollower(config)
    robot2._seed_heartbeat_from_current_position()  # no joint state at all yet
    if robot2._last_sent_msg is not None:
        print("[FAIL] expected no seed when no joint state is available yet", flush=True)
        return 1
    print("[OK] no crash, no seed, when joint state isn't available yet", flush=True)

    print("[stage6b] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
