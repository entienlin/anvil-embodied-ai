"""RobotEnv.step() hard-codes info={IS_INTERVENTION: False}, which
step_env_and_process_transition() then merges OVER the True set by the
intervention processor -- so every episode's intervention rate read 0.0.
The patch must drop that key from step()'s info so the merge can't clobber
it, without changing anything else step() returns.

Run: python3 -m lerobot_control.hilserl._stage7d_intervention_flag_not_clobbered_test
"""

import sys

from lerobot.rl.gym_manipulator import RobotEnv
from lerobot.teleoperators.utils import TeleopEvents

from lerobot_control.hilserl.patches import patch_robot_env_step_keeps_intervention_flag


def main() -> int:
    def fake_step(self, action):  # mimics stock RobotEnv.step's return shape
        return "obs", 0.0, False, False, {TeleopEvents.IS_INTERVENTION: False, "other": 1}

    RobotEnv.step = fake_step
    patch_robot_env_step_keeps_intervention_flag()
    patch_robot_env_step_keeps_intervention_flag()  # idempotent

    env = RobotEnv.__new__(RobotEnv)
    obs, reward, terminated, truncated, info = env.step(None)

    if TeleopEvents.IS_INTERVENTION in info:
        print("[FAIL] step() still reports IS_INTERVENTION and would clobber the processor's value", flush=True)
        return 1
    print("[OK] step() no longer returns IS_INTERVENTION", flush=True)

    if (obs, reward, terminated, truncated) != ("obs", 0.0, False, False) or info.get("other") != 1:
        print("[FAIL] patch altered other return values", flush=True)
        return 1
    print("[OK] every other return value and info key untouched", flush=True)

    # The exact merge step_env_and_process_transition() performs:
    new_info = {TeleopEvents.IS_INTERVENTION: True}  # what the intervention processor recorded
    new_info.update(info)
    if new_info.get(TeleopEvents.IS_INTERVENTION, False) is not True:
        print("[FAIL] intervention flag was still overwritten by the merge", flush=True)
        return 1
    print("[OK] merged info keeps IS_INTERVENTION=True when the operator intervened", flush=True)

    new_info = {TeleopEvents.IS_INTERVENTION: False}
    new_info.update(info)
    if new_info.get(TeleopEvents.IS_INTERVENTION, False) is not False:
        print("[FAIL] flag reads True when not intervening", flush=True)
        return 1
    print("[OK] still False when not intervening", flush=True)

    print("[stage7d] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
