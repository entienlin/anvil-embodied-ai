"""Regression test for the real Stage 5 crash: "RuntimeError: Executor is
already spinning".

Root cause (confirmed via rclpy source, rclpy/utilities.py's spin_once()
docstring): "If no executor is provided (ie. None), then the global
executor is used." AnvilOpenArmFollower and AnvilOpenArmLeader each create
their own Node and spin it via the bare rclpy.spin_once(node) convenience
call in an independent background thread -- with no explicit executor,
both silently share ONE process-wide global executor, and two threads
calling .spin_once() on the same Executor instance concurrently trip its
reentrancy guard. Reproduced for real on real hardware the first time both
classes ever ran together in one process (Stage 2 only had the follower,
Stage 4 bypasses the leader's connect() entirely).

This test doesn't need either full class or real hardware -- the bug is
purely about executor-sharing between two Nodes' spin threads, reproducible
with two bare Nodes. Confirms the actual fix pattern now used in both
classes (a private SingleThreadedExecutor per node) lets two nodes spin
concurrently in one process without colliding.

Run: python3 -m lerobot_control.hilserl._stage5c_dual_node_spin_test
"""

import sys
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node


def main() -> int:
    rclpy.init(args=None)

    node_a = Node("stage5c_node_a")
    node_b = Node("stage5c_node_b")

    executor_a = SingleThreadedExecutor()
    executor_a.add_node(node_a)
    executor_b = SingleThreadedExecutor()
    executor_b.add_node(node_b)

    errors: list[Exception] = []
    stop = threading.Event()

    def spin(executor):
        try:
            while not stop.is_set():
                executor.spin_once(timeout_sec=0.05)
        except Exception as e:  # noqa: BLE001 -- want to see anything, not just RuntimeError
            errors.append(e)

    print("[stage5c] spinning two independent nodes concurrently for 1s "
          "(each with its own dedicated executor)...", flush=True)
    t_a = threading.Thread(target=spin, args=(executor_a,), daemon=True)
    t_b = threading.Thread(target=spin, args=(executor_b,), daemon=True)
    t_a.start()
    t_b.start()
    time.sleep(1.0)
    stop.set()
    t_a.join(timeout=2.0)
    t_b.join(timeout=2.0)

    executor_a.shutdown()
    executor_b.shutdown()
    node_a.destroy_node()
    node_b.destroy_node()
    rclpy.shutdown()

    if errors:
        print(f"[FAIL] {len(errors)} exception(s) during concurrent spin: {errors}", flush=True)
        return 1

    print("[OK] two nodes spun concurrently for 1s with dedicated executors, zero exceptions", flush=True)
    print("[stage5c] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
