"""Isolated test for patch_inverse_kinematics_to_always_reseed_from_real_joints(),
added after the follower snapped to a bizarre "shutdown"-looking joint
configuration immediately after a real intervention give-back, with no
error logged at all.

Diagnosis: InverseKinematicsRLStep (constructed with
initial_guess_current_joints=False by gym_manipulator.py, not configurable)
seeds its IK solve from the real robot's joints only on its very first
call ever -- every call after that reseeds from its OWN previous output,
never the real robot's actual joints again. During real intervention the
robot's real joints are driven independently (native leader-mirroring,
bypassing this step entirely), so the solver's internal seed and the
robot's real state diverge -- and because this is a 7-DOF arm (redundant
for a 6D pose), that divergence can be a completely different elbow
configuration even when the resulting EE pose looks fine.

This test doesn't need real divergence to demonstrate the bug -- it
directly inspects WHICH q_curr value kinematics.inverse_kinematics() was
actually called with, via a recording wrapper around the real
RobotKinematics instance (real URDF, no ROS2/hardware needed).

Run: python3 -m lerobot_control.hilserl._stage5f_ik_reseed_patch_test
"""

import sys

import numpy as np
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.converters import create_transition
from lerobot.robots.so_follower.robot_kinematic_processor import InverseKinematicsRLStep
from lerobot.types import TransitionKey

from lerobot_control.hilserl.patches import patch_inverse_kinematics_to_always_reseed_from_real_joints

URDF = "/workspace/configs/hilserl/urdf/openarm_teleop_kinematics.urdf"
MOTOR_NAMES = [f"follower_r_joint{i}" for i in range(1, 8)]


class _RecordingKinematics:
    """Delegates to a real RobotKinematics, recording the q_curr each
    inverse_kinematics() call was actually seeded with."""

    def __init__(self, real: RobotKinematics):
        self._real = real
        self.seed_log: list[np.ndarray] = []

    def forward_kinematics(self, *a, **kw):
        return self._real.forward_kinematics(*a, **kw)

    def inverse_kinematics(self, q_curr, t_des, **kw):
        self.seed_log.append(np.array(q_curr, dtype=float).copy())
        return self._real.inverse_kinematics(q_curr, t_des, **kw)


def _make_transition(joint_deg: np.ndarray, target_pos: np.ndarray):
    observation = {f"{j}.pos": float(v) for j, v in zip(MOTOR_NAMES, joint_deg, strict=False)}
    action = {
        "ee.x": float(target_pos[0]),
        "ee.y": float(target_pos[1]),
        "ee.z": float(target_pos[2]),
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "ee.gripper_pos": 0.0,
    }
    return create_transition(observation=observation, action=action)


def main() -> int:
    real_kinematics = RobotKinematics(urdf_path=URDF, target_frame_name="follower_r_hand_tcp", joint_names=MOTOR_NAMES)

    pose_a = np.zeros(7)
    pose_b = np.array([20.0, 10.0, 0.0, 30.0, 0.0, 0.0, 0.0])  # a real robot pose far from A
    target_a = real_kinematics.forward_kinematics(pose_a)[:3, 3]

    print("[stage5f] UNPATCHED: confirming the seed really does go stale "
          "across calls (sanity-checks the diagnosis)...", flush=True)
    rec = _RecordingKinematics(real_kinematics)
    step = InverseKinematicsRLStep(kinematics=rec, motor_names=MOTOR_NAMES, initial_guess_current_joints=False)

    step(_make_transition(pose_a, target_a))  # first call: seeds from real q_raw (pose_a)
    step(_make_transition(pose_b, target_a))  # second call: robot is now "really" at pose_b

    seed1, seed2 = rec.seed_log
    if not np.allclose(seed1, pose_a, atol=1e-6):
        print(f"[FAIL] first call should seed from pose_a, got {seed1}", flush=True)
        return 1
    if np.allclose(seed2, pose_b, atol=1e-3):
        print(f"[FAIL] unpatched second call should NOT seed from the real pose_b "
              f"(that would mean the bug doesn't exist) -- got {seed2}", flush=True)
        return 1
    print(f"[OK] unpatched: second call seeded from stale solver state {seed2}, "
          f"NOT the real current pose {pose_b} -- confirms the diagnosis", flush=True)

    print("[stage5f] applying patch...", flush=True)
    patch_inverse_kinematics_to_always_reseed_from_real_joints()

    print("[stage5f] PATCHED: every call must seed from the REAL current joints, "
          "regardless of prior calls...", flush=True)
    rec2 = _RecordingKinematics(real_kinematics)
    step2 = InverseKinematicsRLStep(kinematics=rec2, motor_names=MOTOR_NAMES, initial_guess_current_joints=False)

    step2(_make_transition(pose_a, target_a))
    step2(_make_transition(pose_b, target_a))  # robot "really" moved to pose_b via intervention
    step2(_make_transition(pose_a, target_a))  # and back -- must reseed from pose_a again, not drift

    seeds = rec2.seed_log
    expected = [pose_a, pose_b, pose_a]
    for i, (got, exp) in enumerate(zip(seeds, expected, strict=False)):
        if not np.allclose(got, exp, atol=1e-6):
            print(f"[FAIL] call {i}: expected seed {exp}, got {got}", flush=True)
            return 1
    print("[OK] patched: every call seeded exactly from that call's real current "
          "joints, no drift across calls", flush=True)

    print("[stage5f] re-applying patch must be a no-op (idempotent)...", flush=True)
    before = InverseKinematicsRLStep.__call__
    patch_inverse_kinematics_to_always_reseed_from_real_joints()
    if InverseKinematicsRLStep.__call__ is not before:
        print("[FAIL] re-applying the patch replaced the method again", flush=True)
        return 1
    print("[OK] idempotent", flush=True)

    print("[stage5f] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
