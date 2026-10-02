"""Offline test for the 2026-10-02 fix inside patch_step_env_squeezes_batched_action():
env_processor's own batch-action step (to_batch_action_processor) requires
the transition's action to be a torch.Tensor ("Action should be a
PolicyAction type (tensor), but got <class 'list'>"). For joint-space
configs, InterventionActionProcessorStep's ndarray branch (used whenever
actually intervening) produces action.tolist() -- a plain Python list --
and nothing downstream re-wraps it, unlike the ee_delta/IK chain which
happens to already hand back a tensor. Hit on the first real bimanual
recording session that actually intervened and stepped the real robot --
never caught before because hilserl_gym_manipulator had never recorded in
joint-space mode before (single-arm's own joint-space training data was
relabeled from an old ee_delta recording, not freshly recorded this way).

Pure-Python, no ROS2 daemon needed: same stub env/robot approach as
_bimanual6_neutral_action_resize_test.py, with an env_processor stub that
returns its input transition unchanged so the final action type can be
inspected directly.
"""

import sys


def _check(label: str, cond: bool) -> bool:
    print(f"[{'OK' if cond else 'FAIL'}] {label}", flush=True)
    return cond


class _StubBus:
    def __init__(self, joint_names):
        self.motors = dict.fromkeys(joint_names)


class _StubRobot:
    def __init__(self, joint_names):
        self.bus = _StubBus(joint_names)

    def get_observation(self):
        return {f"{j}.pos": 0.0 for j in self.bus.motors}


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

    print("[bimanual7] intervening joint-space tick: action.tolist() (plain list) action...", flush=True)
    joint_names = [f"j{i}" for i in range(14)]
    robot = _StubRobot(joint_names)
    env = _StubEnv(robot)

    # Mirrors InterventionActionProcessorStep's real ndarray branch: a plain
    # Python list, exactly what action.tolist() produces -- NOT a tensor.
    real_intervening_action = [float(i) for i in range(14)]

    captured = {}

    def _action_processor(transition):
        return {
            TransitionKey.ACTION: transition[TransitionKey.ACTION],
            TransitionKey.REWARD: 0.0,
            TransitionKey.DONE: False,
            TransitionKey.TRUNCATED: False,
            TransitionKey.COMPLEMENTARY_DATA: {},
            TransitionKey.INFO: {},
        }

    def _env_processor(transition):
        # Simulates to_batch_action_processor's own real requirement: raise
        # exactly like the real pipeline step does if given a non-tensor.
        action = transition[TransitionKey.ACTION]
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"Action should be a PolicyAction type (tensor), but got {type(action)}")
        captured["final_action"] = action
        return transition

    initial_transition = create_transition(action=real_intervening_action)
    gym_manipulator.step_env_and_process_transition(
        env=env,
        transition=initial_transition,
        action=real_intervening_action,
        env_processor=_env_processor,
        action_processor=_action_processor,
    )

    final_action = captured.get("final_action")
    ok &= _check("env_processor received a torch.Tensor, not a list", isinstance(final_action, torch.Tensor))
    ok &= _check(
        "tensor values match the original list exactly",
        final_action is not None and final_action.tolist() == real_intervening_action,
    )

    print("\n[bimanual7] action already a tensor (e.g. SAC path) passes through correctly...", flush=True)
    captured2 = {}

    def _env_processor2(transition):
        action = transition[TransitionKey.ACTION]
        if not isinstance(action, torch.Tensor):
            raise ValueError(f"Action should be a PolicyAction type (tensor), but got {type(action)}")
        captured2["final_action"] = action
        return transition

    tensor_action = torch.tensor([1.0, 2.0, 3.0])
    env3 = _StubEnv(_StubRobot(["a", "b", "c"]))
    initial_transition2 = create_transition(action=tensor_action)
    gym_manipulator.step_env_and_process_transition(
        env=env3,
        transition=initial_transition2,
        action=tensor_action,
        env_processor=_env_processor2,
        action_processor=_action_processor,
    )
    ok &= _check("already-a-tensor action still a tensor", isinstance(captured2.get("final_action"), torch.Tensor))

    if not ok:
        print("\n[bimanual7] FAILED", flush=True)
        return 1
    print("\n[bimanual7] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
