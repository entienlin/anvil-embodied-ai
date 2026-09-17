"""Isolated test for patch_ee_bounds_and_safety_to_clamp_not_crash(), added
after the real crash: "EE jump 0.275m > 0.05m" during a real intervention
give-back, which killed the script before it could switch back to policy
control or save the episode.

No ROS2, no hardware -- EEBoundsAndSafety is a plain dataclass processor
step, testable directly.

Run: python3 -m lerobot_control.hilserl._stage5d_ee_bounds_patch_test
"""

import sys

from lerobot.robots.so_follower.robot_kinematic_processor import EEBoundsAndSafety

from lerobot_control.hilserl.patches import patch_ee_bounds_and_safety_to_clamp_not_crash


def _make_action(x, y, z):
    return {"ee.x": x, "ee.y": y, "ee.z": z, "ee.wx": 0.0, "ee.wy": 0.0, "ee.wz": 0.0}


def main() -> int:
    bounds = {"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]}

    print("[stage5d] confirming the ORIGINAL (unpatched) behavior really does "
          "raise on an oversized jump...", flush=True)
    step_before = EEBoundsAndSafety(end_effector_bounds=bounds)
    step_before.action(_make_action(0.0, 0.0, 0.0))  # establishes _last_pos
    try:
        step_before.action(_make_action(0.3, 0.0, 0.0))  # 0.3m jump > 0.05m default
        print("[FAIL] expected a ValueError before patching, got none", flush=True)
        return 1
    except ValueError as e:
        print(f"[OK] unpatched step correctly raised: {e}", flush=True)

    print("[stage5d] applying patch...", flush=True)
    patch_ee_bounds_and_safety_to_clamp_not_crash()

    print("[stage5d] same oversized jump must now CLAMP, not raise...", flush=True)
    step_after = EEBoundsAndSafety(end_effector_bounds=bounds)
    step_after.action(_make_action(0.0, 0.0, 0.0))
    result = step_after.action(_make_action(0.3, 0.0, 0.0))
    moved = result["ee.x"]
    if not (0.0 < moved <= 0.05 + 1e-9):
        print(f"[FAIL] expected ee.x clamped to (0, 0.05], got {moved}", flush=True)
        return 1
    print(f"[OK] clamped to ee.x={moved:.4f} (max_ee_step_m=0.05), no exception raised", flush=True)

    print("[stage5d] a SMALL jump (within max_ee_step_m) must pass through unclamped...", flush=True)
    result2 = step_after.action(_make_action(moved + 0.01, 0.0, 0.0))
    if abs(result2["ee.x"] - (moved + 0.01)) > 1e-9:
        print(f"[FAIL] small jump was altered: expected {moved + 0.01}, got {result2['ee.x']}", flush=True)
        return 1
    print("[OK] small jump passed through unmodified", flush=True)

    print("[stage5d] bounds clipping still applies (unrelated to the jump check)...", flush=True)
    result3 = step_after.action(_make_action(50.0, 0.0, 0.0))
    # Clamped first by max_ee_step_m (since 50.0 is a huge jump from wherever
    # we are), so this just confirms it stays within [-1, 1] eventually --
    # run it a few times to let the clamp walk it to the bound.
    for _ in range(200):
        result3 = step_after.action(_make_action(50.0, 0.0, 0.0))
    if not (0.99 <= result3["ee.x"] <= 1.0 + 1e-9):
        print(f"[FAIL] expected ee.x to converge to bounds max 1.0, got {result3['ee.x']}", flush=True)
        return 1
    print(f"[OK] converged to bounds max: ee.x={result3['ee.x']:.4f}", flush=True)

    print("[stage5d] re-applying patch must be a no-op (idempotent)...", flush=True)
    before = EEBoundsAndSafety.action
    patch_ee_bounds_and_safety_to_clamp_not_crash()
    if EEBoundsAndSafety.action is not before:
        print("[FAIL] re-applying the patch replaced the method again", flush=True)
        return 1
    print("[OK] idempotent", flush=True)

    print("[stage5d] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
