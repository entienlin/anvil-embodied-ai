"""Hardware-free check for the joint-space ACT eval path
(run_joint_policy_eval.py): does make_policy()+select_action() work for a
7-dim joint-space action (matching eval_act_jointspace_push_bottle.yaml's
env: block and the demos_v2_jointspace dataset), and does the output shape
match what env.step() expects (a flat 7-element tensor, motor_names order,
after squeezing ACT's batch dimension)? No robot/ROS2 needed -- untrained
ACT policy, shapes/plumbing only.

Run: python3 -m lerobot_control.hilserl._act_jointspace_policy_plumbing_test
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
            "joint_action": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
        },
        features_map={
            "state": "observation.state",
            "wrist_r_image": "observation.images.wrist_r",
            "joint_action": "action",
        },
    )
    policy_cfg = ACTConfig(
        chunk_size=50,
        n_action_steps=30,
        normalization_mapping={"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"},
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg).eval()

    if list(policy_cfg.output_features) != ["action"] or policy_cfg.output_features["action"].shape != (7,):
        print(f"[FAIL] wrong action feature: {policy_cfg.output_features}", flush=True)
        return 1
    print("[OK] policy constructed with a 7-dim joint-space action (not the old 3-dim EE-delta)", flush=True)

    device = torch.device(policy_cfg.device)
    batch = {
        "observation.state": torch.randn(1, 7, device=device),
        "observation.images.wrist_r": torch.rand(1, 3, 128, 128, device=device),
    }
    with torch.no_grad():
        action = policy.select_action(batch=batch)
    if tuple(action.shape) != (1, 7):
        print(f"[FAIL] expected action shape (1, 7), got {tuple(action.shape)}", flush=True)
        return 1
    print(f"[OK] select_action() returns the right shape: {tuple(action.shape)}", flush=True)

    # run_joint_policy_eval.py does action.squeeze(0) before handing this to env.step(),
    # which needs a flat, indexable (7,) sequence matching motor_names order.
    raw_action = action.squeeze(0) if action.dim() > 1 else action
    if tuple(raw_action.shape) != (7,):
        print(f"[FAIL] squeezed action should be flat (7,), got {tuple(raw_action.shape)}", flush=True)
        return 1
    print("[OK] squeezed action is a flat 7-element tensor, ready for env.step() with no EE-delta/IK involved", flush=True)

    print("[act_jointspace_policy_plumbing] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
