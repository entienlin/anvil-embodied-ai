"""Liveness watchdog must NOT trip while the human is intervening (forward
position controller inactive => our commands legitimately ignored), but must
still trip when the policy is driving and the arm doesn't respond.

Run: python3 -m lerobot_control.hilserl._stage6h_liveness_skip_when_intervening_test
"""

import sys

import numpy as np

from lerobot_control.hilserl import teleop_anvil_openarm_leader as leader_mod
from lerobot_control.hilserl.robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig


class _Kb:
    is_intervention = False


class _Leader:
    def __init__(self, intervening):
        self._was_intervening = intervening
        self._keyboard = _Kb()


def _run(robot, ticks=30):
    cmd, obs = np.full(8, 0.2), np.zeros(8)
    for _ in range(ticks):
        robot._check_liveness(cmd, obs)


def main() -> int:
    cfg = AnvilOpenArmFollowerConfig(ros_prefix="follower_r", camera_topics={"wrist_r": "/x"})
    robot = AnvilOpenArmFollower(cfg)

    leader_mod._registered_leader = _Leader(intervening=True)
    try:
        _run(robot)
    except RuntimeError:
        print("[FAIL] watchdog tripped while intervening", flush=True)
        return 1
    print("[OK] no trip while intervening", flush=True)

    leader_mod._registered_leader = _Leader(intervening=False)
    try:
        _run(robot)
    except RuntimeError:
        print("[OK] still trips when policy drives and arm is unresponsive", flush=True)
        return 0
    print("[FAIL] watchdog no longer trips for a genuinely dead arm", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
