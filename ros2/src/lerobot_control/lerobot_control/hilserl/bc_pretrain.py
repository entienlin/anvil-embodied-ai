"""Behavior-cloning pretrain for the HIL-SERL SAC actor.

lerobot 0.5.1 only loads SAC weights via `policy.pretrained_path`, and the
learner's `dataset:` only feeds RLPD-style batch mixing -- it never
pretrains. An untrained SAC actor samples wild, jittery actions (fault #11),
so this script fits the actor to the recorded human demos first and saves a
checkpoint that `policy.pretrained_path` can point at.

The SAC pipeline feeds RAW (unnormalized) observations to the policy, so
this trains on raw dataset tensors. Actor loss = MSE(tanh(mean), demo action)
plus a penalty on the unclamped std so the resulting policy acts near
deterministically. No robot / ROS2 needed at runtime.

Usage (learner-style config, same yaml as the real run):
    BC_STEPS=1500 BC_OUT=outputs/bc_pretrained \
    ros2 run lerobot_control hilserl_bc_pretrain --config_path <yaml>
"""

import logging
import os

import torch
import torch.nn.functional as F  # noqa: N812

from lerobot_control.hilserl import robot_anvil_openarm  # noqa: F401 -- registration side effect
from lerobot_control.hilserl import teleop_anvil_openarm_leader  # noqa: F401 -- registration side effect

import lerobot.policies.sac.configuration_sac  # noqa: F401,E402 -- registers the "sac" policy choice for draccus
import lerobot.policies.sac.modeling_sac  # noqa: F401,E402

STD_PENALTY_WEIGHT = 0.1


def _bc_loss(policy, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actor = policy.actor
    obs_enc = actor.encoder(batch, cache=None, detach=False)
    hidden = actor.network(obs_enc)
    mean = actor.mean_layer(hidden)
    std = torch.exp(actor.std_layer(hidden))  # unclamped so the penalty always has gradient
    target = batch["action"].clamp(-0.99, 0.99)
    mse = F.mse_loss(torch.tanh(mean), target)
    return mse + STD_PENALTY_WEIGHT * std.mean(), mse, std.mean()


def bc_pretrain(cfg) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy

    steps = int(os.environ.get("BC_STEPS", "1500"))
    out_dir = os.environ.get("BC_OUT", "outputs/bc_pretrained")
    batch_size = int(os.environ.get("BC_BATCH", "64"))
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} already exists; pick a new BC_OUT")

    device = torch.device(cfg.policy.device)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.train()

    datasets = [LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, video_backend=cfg.dataset.video_backend)]
    # Optional extra human-driven data (e.g. the learner's saved replay-buffer
    # datasets, outputs/<run>/dataset), comma-separated container paths.
    for i, root in enumerate(p for p in os.environ.get("BC_EXTRA_DATASETS", "").split(",") if p):
        extra = LeRobotDataset(f"local/bc_extra_{i}", root=root, video_backend=cfg.dataset.video_backend)
        print(f"[BC] extra dataset {root}: {extra.num_episodes} episodes, {len(extra)} frames", flush=True)
        datasets.append(extra)

    class _Keys(torch.utils.data.Dataset):
        """Only the tensors BC uses -- the saved replay-buffer datasets carry
        different extra columns, which breaks default_collate on a mixed batch."""

        def __init__(self, inner):
            self.inner = inner

        def __len__(self):
            return len(self.inner)

        def __getitem__(self, i):
            item = self.inner[i]
            return {k: item[k] for k in ("observation.state", "observation.images.wrist_r", "action")}

    ds = torch.utils.data.ConcatDataset([_Keys(d) for d in datasets])
    print(f"[BC] training on {len(ds)} frames total", flush=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)

    params = list(policy.actor.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    step = 0
    while step < steps:
        for raw in loader:
            batch = {k: v.to(device) for k, v in raw.items() if isinstance(v, torch.Tensor)}
            loss, mse, std = _bc_loss(policy, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"[BC] step {step}/{steps} loss {loss.item():.4f} mse {mse.item():.4f} mean_std {std.item():.4f}", flush=True)
            if step >= steps:
                break

    policy.save_pretrained(out_dir)
    logging.info("[BC] saved pretrained policy to %s", out_dir)


def main() -> None:
    from lerobot.configs import parser
    from lerobot.configs.train import TrainRLServerPipelineConfig

    from lerobot_control.hilserl.patches import apply_all_patches

    logging.basicConfig(level=logging.INFO)
    apply_all_patches()

    @parser.wrap()
    def cli(cfg: TrainRLServerPipelineConfig):
        bc_pretrain(cfg)

    cli()


if __name__ == "__main__":
    main()
