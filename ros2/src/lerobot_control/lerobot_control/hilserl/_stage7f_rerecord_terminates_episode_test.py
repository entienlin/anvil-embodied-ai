"""`r` (rerecord) must terminate the episode AND carry RERECORD_EPISODE on the
same step, so lerobot's record loop discards it (it only checks the flag at
the terminating step). s/f must be unchanged. Pure logic.

Run: python3 -m lerobot_control.hilserl._stage7f_rerecord_terminates_episode_test
"""

import sys

from lerobot.teleoperators.utils import TeleopEvents

from lerobot_control.hilserl.teleop_anvil_openarm_leader import AnvilOpenArmLeader, AnvilOpenArmLeaderConfig


class _Kb:
    is_intervention = False

    def __init__(self, **labels):
        self._labels = {"success": False, "failure": False, "rerecord": False, **labels}

    def consume_episode_labels(self):
        return dict(self._labels)


def main() -> int:
    leader = AnvilOpenArmLeader(
        AnvilOpenArmLeaderConfig(urdf_path="/workspace/configs/hilserl/urdf/openarm_teleop_kinematics.urdf")
    )
    cases = [
        ("nothing pressed", {}, False, False),
        ("s", {"success": True}, True, False),
        ("f", {"failure": True}, True, False),
        ("r", {"rerecord": True}, True, True),
    ]
    for name, labels, want_term, want_rerec in cases:
        leader._keyboard = _Kb(**labels)
        ev = leader.get_teleop_events()
        if ev[TeleopEvents.TERMINATE_EPISODE] != want_term or ev[TeleopEvents.RERECORD_EPISODE] != want_rerec:
            print(f"[FAIL] {name}: terminate={ev[TeleopEvents.TERMINATE_EPISODE]} rerecord={ev[TeleopEvents.RERECORD_EPISODE]}", flush=True)
            return 1
        print(f"[OK] {name}: terminate={want_term} rerecord={want_rerec}", flush=True)
    print("[stage7f] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
