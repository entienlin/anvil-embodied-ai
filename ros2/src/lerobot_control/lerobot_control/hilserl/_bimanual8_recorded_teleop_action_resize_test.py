"""Offline test for the 2026-10-02 fix inside patch_step_env_squeezes_batched_action():
complementary_data["teleop_action"] (what control_loop() actually records
into the dataset frame) stays at the undersized 3-element neutral_action
placeholder on any NON-intervening tick -- InterventionActionProcessorStep
only overwrites it with the real action while IS_INTERVENTION is True.
This is separate from (and downstream of) the neutral_action fix tested in
_bimanual6_neutral_action_resize_test.py, which only affects what's SENT
to the robot, not what gets RECORDED. Confirmed on real hardware:
`ValueError: The feature 'action' of shape '(3,)' does not have the
expected shape '(14,)'` the first time a non-intervening tick's frame was
actually saved to a real bimanual dataset.

Pure-Python, no ROS2 daemon needed: same stub env/robot approach as
_bimanual6/_bimanual7, with action_processor simulating
InterventionActionProcessorStep's real behavior -- complementary_data's
teleop_action only gets replaced with the real action while intervening.
"""

import sys


def _check(label: str, cond: bool) -> bool:
    print(f"[{'OK' if cond else 'FAIL'}] {label}", flush=True)
    return cond


class _StubBus:
    def __init__(self, joint_names):
        self.motors = dict.fromkeys(joint_names)


class _StubRobot:
    def __init__(self, joint_names, current_positions):
        self.bus = _StubBus(joint_names)
        self._current_positions = current_positions

    def get_observation(self):
        return dict(self._current_positions)


class _StubEnv:
    def __init__(self, robot):
        self.robot = robot

    def step(self, action):
        return {"observation.state": list(action)}, 0.0, False, False, {}


def main() -> int:
    from lerobot_control.hilserl.patches import patch_step_env_squeezes_batched_action

    patch_step_env_squeezes_batched_action()

    import torch

    from lerobot.rl import gym_manipulator
    from lerobot.rl.gym_manipulator import create_transition
    from lerobot.types import TransitionKey

    ok = True
    joint_names = [f"j{i}" for i in range(14)]
    current_positions = {f"{j}.pos": float(100 + i) for i, j in enumerate(joint_names)}
    robot = _StubRobot(joint_names, current_positions)

    print("[bimanual8] non-intervening tick: complementary_data's teleop_action stays undersized...", flush=True)
    neutral_action = torch.tensor([0.0, 0.0, 0.0])

    def _action_processor_not_intervening(transition):
        # Mirrors InterventionActionProcessorStep's real behavior exactly:
        # NOT intervening -> complementary_data["teleop_action"] is left as
        # whatever action came in (the undersized neutral_action), never
        # overwritten with the real 14-dim teleop action.
        return {
            TransitionKey.ACTION: transition[TransitionKey.ACTION],
            TransitionKey.REWARD: 0.0,
            TransitionKey.DONE: False,
            TransitionKey.TRUNCATED: False,
            TransitionKey.COMPLEMENTARY_DATA: {"teleop_action": transition[TransitionKey.ACTION]},
            TransitionKey.INFO: {},
        }

    captured = {}

    def _env_processor(transition):
        captured["complementary_data"] = transition[TransitionKey.COMPLEMENTARY_DATA]
        return transition

    env = _StubEnv(robot)
    initial_transition = create_transition(action=neutral_action)
    gym_manipulator.step_env_and_process_transition(
        env=env,
        transition=initial_transition,
        action=neutral_action,
        env_processor=_env_processor,
        action_processor=_action_processor_not_intervening,
    )

    recorded = captured["complementary_data"]["teleop_action"]
    ok &= _check("recorded teleop_action is now 14-dim, not 3", len(recorded) == 14)
    expected = [current_positions[f"{j}.pos"] for j in joint_names]
    ok &= _check("recorded teleop_action matches the robot's real current position", recorded.tolist() == expected)

    print("\n[bimanual8] intervening tick: real 14-dim teleop_action passes through unchanged...", flush=True)
    real_action = torch.tensor([float(i) for i in range(14)])

    def _action_processor_intervening(transition):
        return {
            TransitionKey.ACTION: real_action,
            TransitionKey.REWARD: 0.0,
            TransitionKey.DONE: False,
            TransitionKey.TRUNCATED: False,
            TransitionKey.COMPLEMENTARY_DATA: {"teleop_action": real_action},
            TransitionKey.INFO: {},
        }

    captured2 = {}

    def _env_processor2(transition):
        captured2["complementary_data"] = transition[TransitionKey.COMPLEMENTARY_DATA]
        return transition

    env2 = _StubEnv(robot)
    initial_transition2 = create_transition(action=real_action)
    gym_manipulator.step_env_and_process_transition(
        env=env2,
        transition=initial_transition2,
        action=real_action,
        env_processor=_env_processor2,
        action_processor=_action_processor_intervening,
    )
    recorded2 = captured2["complementary_data"]["teleop_action"]
    ok &= _check("already-correctly-sized teleop_action is untouched", recorded2 is real_action)

    if not ok:
        print("\n[bimanual8] FAILED", flush=True)
        return 1
    print("\n[bimanual8] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
