"""Hardware-free check for the ACT proof-of-concept eval path
(run_policy_eval.py): does make_policy()+select_action() actually work for
ACT against this task's exact declared env features (the same features/
features_map as eval_act_push_bottle.yaml's env: block)? No robot/ROS2
needed -- this only exercises the policy-construction and inference call
chain run_policy_eval.py's loop uses, with an untrained ACT policy (shapes/
plumbing only, not quality).

Run: python3 -m lerobot_control.hilserl._act_poc_policy_plumbing_test
"""

import sys

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy


def main() -> int:
    env_cfg = HILSerlRobotEnvConfig(
        features={
            "state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "wrist_r_image": PolicyFeature(type=FeatureType.VISUAL, shape=(128, 128, 3)),
            "ee_delta_action": PolicyFeature(type=FeatureType.ACTION, shape=(3,)),
        },
        features_map={
            "state": "observation.state",
            "wrist_r_image": "observation.images.wrist_r",
            "ee_delta_action": "action",
        },
    )
    policy_cfg = ACTConfig(
        chunk_size=50,
        n_action_steps=30,
        normalization_mapping={"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg).eval()

    expected_in = {"observation.state", "observation.images.wrist_r"}
    if set(policy_cfg.input_features) != expected_in or list(policy_cfg.output_features) != ["action"]:
        print(f"[FAIL] wrong features: in={list(policy_cfg.input_features)} out={list(policy_cfg.output_features)}", flush=True)
        return 1
    print("[OK] policy constructed with the task's real feature names (observation.state, "
          "observation.images.wrist_r -> action)", flush=True)

    device = torch.device(policy_cfg.device)
    batch = {
        "observation.state": torch.randn(1, 7, device=device),
        "observation.images.wrist_r": torch.rand(1, 3, 128, 128, device=device),
    }
    with torch.no_grad():
        action = policy.select_action(batch=batch)
    if tuple(action.shape) != (1, 3):
        print(f"[FAIL] expected action shape (1, 3), got {tuple(action.shape)}", flush=True)
        return 1
    print(f"[OK] select_action() returns the right shape: {tuple(action.shape)}", flush=True)

    # run_policy_eval.py calls select_action() every real tick, same as actor.py does for
    # SAC -- confirm ACT's internal chunk queue survives many consecutive calls, including
    # crossing the n_action_steps=30 re-query boundary, without crashing or changing shape.
    policy.reset()
    for _ in range(35):
        with torch.no_grad():
            a = policy.select_action(batch=batch)
        if tuple(a.shape) != (1, 3):
            print(f"[FAIL] shape changed across repeated calls: {tuple(a.shape)}", flush=True)
            return 1
    print("[OK] 35 consecutive select_action() calls (crossing the n_action_steps=30 "
          "re-query boundary) succeeded with a stable action shape", flush=True)

    print("[act_poc_policy_plumbing] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
