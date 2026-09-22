"""A single raising callback (e.g. a corrupted camera frame) must NOT kill
the whole background spin thread -- see the docstring on
AnvilOpenArmFollower._spin_loop for the real-hardware incident this
guards against (2026-09-22: one bad JPEG frame silently killed the spin
thread, starving observations, which several seconds later tripped the
liveness watchdog on what looked like an unresponsive arm).

Pure logic, no ROS2 daemon needed -- constructs a bare instance the same
bypass-connect() way _stage6g/_stage5c do, injects a raising subscription
callback directly, and confirms the thread survives and keeps spinning.

Run: python3 -m lerobot_control.hilserl._stage7b_spin_loop_survives_callback_exception_test
"""

import sys
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String

from lerobot_control.hilserl.robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig


def main() -> int:
    rclpy.init()
    try:
        cfg = AnvilOpenArmFollowerConfig(
            ros_prefix="follower_r", camera_topics={"wrist_r": "/cam_wrist_r/image_raw/compressed"}
        )
        robot = AnvilOpenArmFollower(cfg)
        robot._node = Node("stage7b_test_node")
        robot._executor = SingleThreadedExecutor()
        robot._executor.add_node(robot._node)
        robot._spin_stop = threading.Event()

        calls = {"n": 0}

        def _raising_cb(_msg):
            calls["n"] += 1
            raise RuntimeError("simulated cv2.imdecode() failure -- corrupted frame")

        pub = robot._node.create_publisher(String, "/stage7b_test_topic", 10)
        robot._node.create_subscription(String, "/stage7b_test_topic", _raising_cb, 10)

        thread = threading.Thread(target=robot._spin_loop, daemon=True)
        thread.start()

        print("[stage7b] publishing 5 messages, each raising in its callback...", flush=True)
        for _ in range(5):
            pub.publish(String(data="boom"))
            time.sleep(0.15)

        if not thread.is_alive():
            print("[FAIL] spin thread died after a raising callback", flush=True)
            return 1
        if calls["n"] < 3:
            print(f"[FAIL] only {calls['n']}/5 callbacks fired -- thread likely died partway", flush=True)
            return 1
        print(f"[OK] spin thread survived {calls['n']} raising callbacks, still alive", flush=True)

        robot._spin_stop.set()
        thread.join(timeout=2.0)
        if thread.is_alive():
            print("[FAIL] spin thread did not stop after _spin_stop.set()", flush=True)
            return 1
        print("[OK] spin thread stops cleanly once told to", flush=True)

        print("[stage7b] PASSED", flush=True)
        return 0
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
