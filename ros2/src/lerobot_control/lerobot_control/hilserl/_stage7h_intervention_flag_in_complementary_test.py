"""InterventionActionProcessorStep must leave its normal output unchanged and
additionally put float(is_intervention) in complementary_data["discrete_penalty"]
(the slot the actor copies into every Transition). Pure logic.

Run: python3 -m lerobot_control.hilserl._stage7h_intervention_flag_in_complementary_test
"""

import sys

import torch
from lerobot.processor.hil_processor import InterventionActionProcessorStep
from lerobot.processor.converters import create_transition
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.types import TransitionKey

from lerobot_control.hilserl.patches import patch_intervention_flag_in_complementary_info


def _run(step, intervening: bool):
    tr = create_transition(
        action=torch.tensor([0.5, 0.5, 0.5]),
        info={TeleopEvents.IS_INTERVENTION: intervening},
        complementary_data={"teleop_action": {"delta_x": 0.1, "delta_y": -0.2, "delta_z": 0.3}},
    )
    return step(tr)


def main() -> int:
    step = InterventionActionProcessorStep(use_gripper=False, terminate_on_success=True)
    patch_intervention_flag_in_complementary_info()
    patch_intervention_flag_in_complementary_info()  # idempotent

    out = _run(step, True)
    flag = out[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty")
    if flag != 1.0:
        print(f"[FAIL] intervening step flag = {flag}", flush=True)
        return 1
    if not torch.allclose(out[TransitionKey.ACTION], torch.tensor([0.1, -0.2, 0.3])):
        print(f"[FAIL] action override changed: {out[TransitionKey.ACTION]}", flush=True)
        return 1
    print("[OK] intervening: flag=1.0 and the human's action still overrides the policy's", flush=True)

    out = _run(step, False)
    if out[TransitionKey.COMPLEMENTARY_DATA].get("discrete_penalty") != 0.0:
        print("[FAIL] non-intervening step should carry 0.0", flush=True)
        return 1
    if not torch.allclose(out[TransitionKey.ACTION], torch.tensor([0.5, 0.5, 0.5])):
        print("[FAIL] policy action must pass through when not intervening", flush=True)
        return 1
    if "teleop_action" not in out[TransitionKey.COMPLEMENTARY_DATA]:
        print("[FAIL] teleop_action bookkeeping lost", flush=True)
        return 1
    print("[OK] not intervening: flag=0.0, policy action and teleop_action bookkeeping intact", flush=True)
    print("[stage7h] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
