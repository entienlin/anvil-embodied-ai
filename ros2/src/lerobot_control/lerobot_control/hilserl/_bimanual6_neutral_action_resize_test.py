"""Offline test for the 2026-10-02 fix inside patch_step_env_squeezes_batched_action():
stock lerobot's control_loop() always sends a hardcoded 3-element (4 with
gripper) "neutral_action" on any non-intervening tick -- harmless for
ee_delta configs (the IK chain always re-sizes it before env.step()), but
for joint-space configs (processor.inverse_kinematics: null) that chain is
skipped entirely, so the 3-element placeholder reached env.step()
unmodified and crashed with "IndexError: index 3 is out of bounds for
dimension 0 with size 3" the moment a robot has more than 3 joints. Hit on
the very first bimanual (14-joint) recording attempt; every single-arm
(7-joint) session so far happened to have the human intervening from the
very first tick, which is why it was never triggered there either.

Pure-Python, no ROS2 daemon needed: stubs env.robot (bus.motors +
get_observation()) and env.step() directly, calls the real patched
gym_manipulator.step_env_and_process_transition().
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
        self.last_step_action = None

    def step(self, action):
        self.last_step_action = list(action)
        return {"observation.state": [0.0]}, 0.0, False, False, {}


def main() -> int:
    from lerobot_control.hilserl.patches import patch_step_env_squeezes_batched_action

    patch_step_env_squeezes_batched_action()

    from lerobot.rl import gym_manipulator
    from lerobot.rl.gym_manipulator import create_transition
    from lerobot.types import TransitionKey

    ok = True

    print("[bimanual6] non-intervening tick: 3-element placeholder action, 7-joint robot...", flush=True)
    joint_names = [f"follower_r_joint{i}" for i in range(1, 8)]
    current_positions = {f"{j}.pos": float(10 + i) for i, j in enumerate(joint_names)}
    robot = _StubRobot(joint_names, current_positions)
    env = _StubEnv(robot)

    def _identity_action_processor(transition):
        # Simulates InterventionActionProcessorStep when NOT intervening:
        # the short placeholder action passes through unchanged.
        return {
            TransitionKey.ACTION: transition[TransitionKey.ACTION],
            TransitionKey.REWARD: 0.0,
            TransitionKey.DONE: False,
            TransitionKey.TRUNCATED: False,
            TransitionKey.COMPLEMENTARY_DATA: {},
            TransitionKey.INFO: {},
        }

    def _identity_env_processor(transition):
        return transition

    neutral_action = [0.0, 0.0, 0.0]
    initial_transition = create_transition(action=neutral_action)
    gym_manipulator.step_env_and_process_transition(
        env=env,
        transition=initial_transition,
        action=neutral_action,
        env_processor=_identity_env_processor,
        action_processor=_identity_action_processor,
    )

    ok &= _check("env.step() received 7 values, not 3", len(env.last_step_action) == 7)
    expected = [current_positions[f"{j}.pos"] for j in joint_names]
    ok &= _check("env.step() received the robot's REAL CURRENT positions (hold still)", env.last_step_action == expected)
    ok &= _check("NOT all-zero (would be a real, unsafe commanded motion)", env.last_step_action != [0.0] * 7)

    print("\n[bimanual6] intervening tick: correctly-sized 14-element action passes through unchanged...", flush=True)
    joint_names_14 = [f"j{i}" for i in range(14)]
    current_positions_14 = {f"{j}.pos": 0.0 for j in joint_names_14}
    robot2 = _StubRobot(joint_names_14, current_positions_14)
    env2 = _StubEnv(robot2)
    real_action = [float(i) for i in range(14)]
    initial_transition2 = create_transition(action=real_action)
    gym_manipulator.step_env_and_process_transition(
        env=env2,
        transition=initial_transition2,
        action=real_action,
        env_processor=_identity_env_processor,
        action_processor=_identity_action_processor,
    )
    ok &= _check("correctly-sized action is untouched", env2.last_step_action == real_action)

    if not ok:
        print("\n[bimanual6] FAILED", flush=True)
        return 1
    print("\n[bimanual6] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
