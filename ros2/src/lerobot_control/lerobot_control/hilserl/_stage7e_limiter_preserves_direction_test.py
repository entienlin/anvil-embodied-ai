"""The follower's per-tick joint limits must scale the WHOLE joint-delta
vector (direction-preserving), not clip each joint independently -- see the
comment in AnvilOpenArmFollower.send_action for the real-hardware incident
(2026-09-24: independent clipping bent the IK's intended descent into a
rise). Pure logic, no ROS2/hardware.

Run: python3 -m lerobot_control.hilserl._stage7e_limiter_preserves_direction_test
"""

import sys

import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from lerobot_control.hilserl.robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> int:
    scale = AnvilOpenArmFollower._scale_to_cap

    print("[stage7e] _scale_to_cap keeps direction and caps the largest joint...", flush=True)
    delta = np.array([0.16, 0.0, 0.0, -0.33, 0.0, 0.0, 0.17])  # the real descent demand: joints 1, 4, 7
    out = scale(delta, 0.03)
    if abs(float(np.max(np.abs(out))) - 0.03) > 1e-12:
        print(f"[FAIL] largest joint not capped to 0.03: {out}", flush=True)
        return 1
    if _cos(out, delta) < 1 - 1e-12:
        print(f"[FAIL] direction changed (cos={_cos(out, delta)})", flush=True)
        return 1
    clipped = np.clip(delta, -0.03, 0.03)  # what independent clipping produces
    if _cos(clipped, delta) > 1 - 1e-3:
        print("[FAIL] test is vacuous: independent clipping should change direction here", flush=True)
        return 1
    print(f"[OK] capped at 0.03, cos(scaled, original)=1.0 (independent clip would give {_cos(clipped, delta):.3f})", flush=True)

    if not np.array_equal(scale(np.array([0.01, -0.02]), 0.03), np.array([0.01, -0.02])) or not np.array_equal(
        scale(np.zeros(3), 0.03), np.zeros(3)
    ):
        print("[FAIL] small/zero deltas must pass through unchanged", flush=True)
        return 1
    print("[OK] small and zero deltas pass through unchanged", flush=True)

    print("[stage7e] send_action(): both limits preserve the commanded direction...", flush=True)
    cfg = AnvilOpenArmFollowerConfig(
        ros_prefix="follower_r",
        camera_topics={"wrist_r": "/x"},
        max_position_delta=0.05,
        max_command_step_rad=0.03,
    )
    r = AnvilOpenArmFollower(cfg)
    js = JointState()
    js.name = [f"follower_r_joint{i}" for i in range(1, 8)] + ["follower_r_finger_joint1"]
    js.position = [0.0] * 8
    r._obs_manager.update_joint_state(js)
    r._is_connected = True

    class _Pub:
        def publish(self, m):
            pass

    r._pub = _Pub()
    start = Float64MultiArray()
    start.data = [0.0] * 8
    r._last_sent_msg = start
    target = np.array([0.16, 0.0, 0.0, -0.33, 0.0, 0.0, 0.17])
    out = r.send_action({f"follower_r_joint{i + 1}.pos": np.rad2deg(target[i]) for i in range(7)})
    got = np.deg2rad([out[f"follower_r_joint{i}.pos"] for i in range(1, 8)])
    if _cos(got, target) < 1 - 1e-9:
        print(f"[FAIL] published command bent away from the target direction (cos={_cos(got, target)})", flush=True)
        return 1
    if float(np.max(np.abs(got))) > 0.03 + 1e-9:
        print(f"[FAIL] step cap exceeded: {np.max(np.abs(got))}", flush=True)
        return 1
    print(f"[OK] published command is parallel to the target and capped ({np.max(np.abs(got)):.3f} rad max)", flush=True)

    print("[stage7e] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
