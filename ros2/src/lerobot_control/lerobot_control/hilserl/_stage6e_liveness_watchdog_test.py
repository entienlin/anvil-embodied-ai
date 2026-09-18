"""Isolated test for AnvilOpenArmFollower's liveness watchdog, added after
a real-hardware run confirmed the pipeline had no way to detect a dead
arm: the follower faulted 14s into an episode that still logged a clean
"Episode reward: 0.0" completion, because nothing ever checked whether
commanded actions produced any observed motion.

Pure computation test -- bypasses connect() entirely (no ROS2, no
hardware needed), calling _check_liveness() directly with synthetic
commanded/observed joint vectors, matching the pattern already
established in _stage4_leader_test.py / _stage5f_ik_reseed_patch_test.py.

Run: python3 -m lerobot_control.hilserl._stage6e_liveness_watchdog_test
"""

import sys

import numpy as np

from lerobot_control.hilserl.robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig


def _make_robot() -> AnvilOpenArmFollower:
    config = AnvilOpenArmFollowerConfig(
        ros_prefix="follower_r",
        camera_topics={"wrist_r": "/cam_wrist_r/image_raw/compressed"},
    )
    return AnvilOpenArmFollower(config)


def main() -> int:
    n = 7
    window = AnvilOpenArmFollower._LIVENESS_WINDOW_SIZE
    cmd_thresh = AnvilOpenArmFollower._LIVENESS_COMMAND_THRESHOLD_RAD
    still_thresh = AnvilOpenArmFollower._LIVENESS_STILL_THRESHOLD_RAD

    print("[stage6e] holding still (commanded ~= observed) never trips, any duration...", flush=True)
    robot = _make_robot()
    observed = np.zeros(n)
    try:
        for _ in range(window * 3):
            commanded = observed + (cmd_thresh * 0.1)  # well under the "real move" threshold
            robot._check_liveness(commanded, observed)
    except RuntimeError as e:
        print(f"[FAIL] holding still tripped the watchdog unexpectedly: {e}", flush=True)
        return 1
    print("[OK] no trip while holding still", flush=True)

    print("[stage6e] healthy arm (observed tracks commanded) never trips...", flush=True)
    robot = _make_robot()
    observed = np.zeros(n)
    try:
        for _ in range(window * 3):
            commanded = observed + 0.05  # a real move attempt, well above cmd_thresh
            robot._check_liveness(commanded, observed)
            observed = commanded  # arm actually got there -- healthy tracking
    except RuntimeError as e:
        print(f"[FAIL] healthy tracking tripped the watchdog unexpectedly: {e}", flush=True)
        return 1
    print("[OK] no trip while the arm actually tracks commands", flush=True)

    print("[stage6e] dead arm (commanded moves, observed frozen) trips after exactly window ticks...", flush=True)
    robot = _make_robot()
    observed = np.zeros(n)
    tripped_at = None
    try:
        for i in range(window * 2):
            commanded = np.zeros(n)
            commanded[0] = 0.05 * (i + 1)  # keeps demanding real motion every tick
            robot._check_liveness(commanded, observed)  # observed never changes -- "dead"
    except RuntimeError as e:
        tripped_at = i  # noqa: F821 -- i is bound from the loop that raised
        print(f"[OK] watchdog tripped at tick {tripped_at}: {e}", flush=True)
    if tripped_at is None:
        print("[FAIL] expected the watchdog to trip for a dead arm, it never did", flush=True)
        return 1
    if tripped_at != window - 1:
        print(f"[FAIL] expected trip at tick {window - 1} (0-indexed), got {tripped_at}", flush=True)
        return 1
    print(f"[OK] tripped at exactly the expected tick ({window - 1})", flush=True)

    print("[stage6e] an interspersed still tick resets the streak, delaying a trip...", flush=True)
    robot = _make_robot()
    observed = np.zeros(n)
    tripped = False
    try:
        for i in range(window - 1):
            commanded = np.zeros(n)
            commanded[0] = 0.05
            robot._check_liveness(commanded, observed)
        # One "holding still" tick partway through -- should reset the streak.
        robot._check_liveness(observed.copy(), observed)
        for _ in range(window - 1):
            commanded = np.zeros(n)
            commanded[0] = 0.05
            robot._check_liveness(commanded, observed)
    except RuntimeError:
        tripped = True
    if tripped:
        print("[FAIL] expected the reset tick to delay the trip past this many ticks", flush=True)
        return 1
    print("[OK] streak correctly reset by a legitimate still tick", flush=True)

    print("[stage6e] missing observation (current_rad=None) doesn't crash or reset progress...", flush=True)
    robot = _make_robot()
    observed = np.zeros(n)
    try:
        for i in range(window - 1):
            commanded = np.zeros(n)
            commanded[0] = 0.05
            robot._check_liveness(commanded, observed)
        robot._check_liveness(np.zeros(n), None)  # transient gap -- must not reset the streak
        tripped = False
        try:
            commanded = np.zeros(n)
            commanded[0] = 0.05
            robot._check_liveness(commanded, observed)  # this is the window-th real tick
        except RuntimeError:
            tripped = True
        if not tripped:
            print("[FAIL] expected the streak to have survived the None tick and trip here", flush=True)
            return 1
    except RuntimeError as e:
        print(f"[FAIL] unexpected early trip: {e}", flush=True)
        return 1
    print("[OK] None observation handled gracefully, streak preserved", flush=True)

    print("[stage6e] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
