"""Evaluate a SAC checkpoint against a FIXED reference dataset, so results
are comparable across checkpoints/sessions -- see the plan's "Strategic
question" section: past evaluations this session accidentally compared
different checkpoints against different reference sets (or forgot to apply
patch_sac_deterministic_inference(), measuring the unpatched stochastic
select_action() instead of what the real robot actually runs). This script
fixes both: one reference dataset by default, and the deterministic-
inference patch always applied before evaluating.

No ROS2/hardware needed -- pure offline scoring against recorded frames.

Usage:
    ros2 run lerobot_control hilserl_eval_checkpoint \
        --checkpoint outputs/hilserl_push_bottle/checkpoints/001000/pretrained_model \
        --reference-repo-id SSTC-AITeam/hilserl-push-bottle \
        --reference-root data/datasets/push_bottle_demos
"""

import argparse
import sys

import torch


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.sac.modeling_sac import SACPolicy

    from lerobot_control.hilserl.patches import patch_sac_deterministic_inference

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a pretrained_model dir")
    parser.add_argument("--reference-repo-id", default="SSTC-AITeam/hilserl-push-bottle")
    parser.add_argument("--reference-root", default="data/datasets/push_bottle_demos")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--stride", type=int, default=25, help="Evaluate every Nth frame")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Same patch the real actor process applies -- without this, select_action()
    # samples stochastically and the numbers here don't reflect real robot
    # behavior (confirmed bug in an earlier ad-hoc version of this check).
    patch_sac_deterministic_inference()

    policy = SACPolicy.from_pretrained(args.checkpoint).to(args.device).eval()
    ds = LeRobotDataset(
        args.reference_repo_id, root=args.reference_root, video_backend=args.video_backend
    )

    errors = []
    for i in range(0, len(ds), args.stride):
        sample = ds[i]
        batch = {k: v.unsqueeze(0).to(args.device) for k, v in sample.items() if k.startswith("observation")}
        with torch.no_grad():
            action = policy.select_action(batch).squeeze(0).cpu()
        errors.append((action - sample["action"]).abs().mean().item())

    mean_error = sum(errors) / len(errors)
    print(f"EVAL checkpoint={args.checkpoint} n_frames={len(errors)} mean_abs_error_vs_reference={mean_error:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
