"""One-shot, isolated verification of ArmsResetterClient against the real
devbox service -- connects AnvilOpenArmLeader (establishing the normal
policy-controlled baseline, same as any session start), calls
perform_episode_reset_and_wait() exactly once, and reports the result plus
the resulting controller state. No policy, no learner, no dataset -- pure
motion-only check that this project's first real call to
/arms_resetter/reset actually works, before wiring it into anything bigger.

Run: python3 -m lerobot_control.hilserl.verify_arms_resetter
Needs a real TTY (KeyboardIntervention.start() requires one) -- run with
`docker run -it`, matching every other real-hardware script in this
project.
"""

import sys
import time

from lerobot_control.hilserl.teleop_anvil_openarm_leader import AnvilOpenArmLeader, AnvilOpenArmLeaderConfig

URDF_PATH = "/workspace/configs/hilserl/urdf/openarm_teleop_kinematics.urdf"


def main() -> int:
    config = AnvilOpenArmLeaderConfig(urdf_path=URDF_PATH)
    leader = AnvilOpenArmLeader(config)

    print("[verify_arms_resetter] connecting (establishes the normal "
          "policy-controlled baseline, same as any session start)...", flush=True)
    leader.connect()
    print("[verify_arms_resetter] connected OK", flush=True)

    print("[verify_arms_resetter] calling perform_episode_reset_and_wait() -- "
          "the arm should visibly move to its home position now...", flush=True)
    start = time.monotonic()
    try:
        leader.perform_episode_reset_and_wait(timeout_s=18.0)
        elapsed = time.monotonic() - start
        print(f"[verify_arms_resetter] SUCCESS -- homed in {elapsed:.1f}s", flush=True)
    except RuntimeError as e:
        elapsed = time.monotonic() - start
        print(f"[verify_arms_resetter] FAILED after {elapsed:.1f}s: {e}", flush=True)
        leader.disconnect()
        return 1

    states = leader._switch_client._get_controller_states()
    print(f"[verify_arms_resetter] controller states after homing: {states}", flush=True)
    policy_ctrl = leader.config.policy_controller
    if states is None or states.get(policy_ctrl) != "active":
        print(
            f"[verify_arms_resetter] WARNING: expected '{policy_ctrl}' active after "
            f"homing, got {states}",
            flush=True,
        )
        leader.disconnect()
        return 1

    print(f"[verify_arms_resetter] confirmed: '{policy_ctrl}' is active again after "
          "homing, matching the expected post-reset baseline.", flush=True)

    leader.disconnect()
    print("[verify_arms_resetter] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
