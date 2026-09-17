"""Live end-effector position readout, for measuring real
end_effector_bounds against the physical rig.

Read-only: subscribes to /joint_states and continuously prints
follower_r_hand_tcp's live (x, y, z) position via forward kinematics. No
publishing, no switch_controller calls, no dataset -- safe to run at any
time regardless of which controller is currently driving the arm (teleop
or policy), since it never commands anything.

Procedure for measuring end_effector_bounds:
  1. Put the arm in teleop mode (either via the devbox's own manual
     controls, or by running hilserl_gym_manipulator and pressing space)
     so it can be physically jogged by hand via the leader arm.
  2. Run this script alongside it.
  3. Jog the follower to each physical extreme of the region the task
     needs (or the full safe envelope, if measuring that first) and watch
     the printed (x, y, z). Note the min/max you see on each axis as you
     go.
  4. The resulting {min: [x_min, y_min, z_min], max: [x_max, y_max, z_max]}
     is what env_push_bottle.yaml's end_effector_bounds expects.

Run: python3 -m lerobot_control.hilserl.live_ee_position
"""

import sys
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from lerobot.model.kinematics import RobotKinematics

URDF_PATH = "/workspace/configs/hilserl/urdf/openarm_teleop_kinematics.urdf"
JOINT_NAMES = [f"follower_r_joint{i}" for i in range(1, 8)]


def main() -> int:
    rclpy.init(args=None)
    node = Node("live_ee_position")
    executor = SingleThreadedExecutor()
    executor.add_node(node)

    kinematics = RobotKinematics(
        urdf_path=URDF_PATH, target_frame_name="follower_r_hand_tcp", joint_names=JOINT_NAMES
    )

    latest: dict[str, JointState | None] = {"msg": None}

    def on_joint_state(msg: JointState) -> None:
        latest["msg"] = msg

    node.create_subscription(JointState, "/joint_states", on_joint_state, 10)

    print("[live_ee_position] waiting for /joint_states...", flush=True)
    deadline = time.monotonic() + 10.0
    while latest["msg"] is None and time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)
    if latest["msg"] is None:
        print("[live_ee_position] no /joint_states received within 10s -- check "
              "ROS_DOMAIN_ID/CYCLONEDDS_URI match the devbox", flush=True)
        return 1

    print("[live_ee_position] streaming follower_r_hand_tcp position -- "
          "jog the arm and watch the values. Ctrl+C to stop.", flush=True)
    print(f"{'x':>10} {'y':>10} {'z':>10}", flush=True)

    running_min = np.full(3, np.inf)
    running_max = np.full(3, -np.inf)

    try:
        while True:
            executor.spin_once(timeout_sec=0.1)
            msg = latest["msg"]
            name_to_pos = dict(zip(msg.name, msg.position, strict=False))
            if not all(j in name_to_pos for j in JOINT_NAMES):
                continue
            joint_deg = np.rad2deg(np.array([name_to_pos[j] for j in JOINT_NAMES], dtype=float))
            pos = kinematics.forward_kinematics(joint_deg)[:3, 3]
            running_min = np.minimum(running_min, pos)
            running_max = np.maximum(running_max, pos)
            print(f"{pos[0]:10.4f} {pos[1]:10.4f} {pos[2]:10.4f}   "
                  f"(seen so far -- min: [{running_min[0]:.4f}, {running_min[1]:.4f}, {running_min[2]:.4f}]  "
                  f"max: [{running_max[0]:.4f}, {running_max[1]:.4f}, {running_max[2]:.4f}])",
                  end="\r", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print(flush=True)
        print("[live_ee_position] final observed range:", flush=True)
        print(f"  min: [{running_min[0]:.4f}, {running_min[1]:.4f}, {running_min[2]:.4f}]", flush=True)
        print(f"  max: [{running_max[0]:.4f}, {running_max[1]:.4f}, {running_max[2]:.4f}]", flush=True)
        print("[live_ee_position] this is only the range you actually swept through -- "
              "not a safety margin. Widen slightly if you want headroom.", flush=True)

    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
