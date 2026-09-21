"""Consecutive published commands must differ by <= max_command_step_rad.
Run: python3 -m lerobot_control.hilserl._stage6i_command_step_limit_test"""
import sys

import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from lerobot_control.hilserl.robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig


def main() -> int:
    cfg = AnvilOpenArmFollowerConfig(ros_prefix="follower_r", camera_topics={"wrist_r": "/x"}, max_position_delta=0.5)
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
    prev = np.zeros(7)
    for t in (0.4, -0.4, 0.4, -0.4):  # alternating wild targets (deg)
        out = r.send_action({f"follower_r_joint{i}.pos": np.rad2deg(t) for i in range(1, 8)})
        cur = np.deg2rad([out[f"follower_r_joint{i}.pos"] for i in range(1, 8)])
        if np.max(np.abs(cur - prev)) > 0.03 + 1e-9:
            print(f"[FAIL] step {np.max(np.abs(cur-prev))}", flush=True)
            return 1
        prev = cur
    print("[OK] alternating 0.8 rad targets produce <=0.03 rad steps", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
