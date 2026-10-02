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

Data prep (all on by default, see env vars) -- added 2026-09-24 after the
offline learning-health analysis showed earlier BC checkpoints memorized
their few training episodes (train loss 0.001) yet correlated only
+0.1..+0.3 with the human's actions on anything new:
  * successes only   -- failed/timed-out demos are not something to imitate
  * idle lead-in trimmed -- every recorded episode starts with ~2-3 s of
    zero action while the operator presses space; those identical
    home-pose frames teach "at home: do nothing", which is exactly the
    stationary behaviour seen on the robot
  * held-out episodes -- every BC_HOLDOUT_EVERY-th kept episode is never
    trained on; held-out correlation with the human's action (per axis) is
    printed every BC_EVAL_EVERY steps and the best checkpoint is kept.
    This is the go/no-go gate before any RL.

Usage (learner-style config, same yaml as the real run):
    BC_SKIP_BASE=1 BC_EXTRA_DATASETS=/demos BC_STEPS=3000 BC_OUT=outputs/bc_v5 \
    ros2 run lerobot_control hilserl_bc_pretrain --config_path <yaml>

DAgger-style corrections (BC_TRAJ_DIRS): the actor saves every episode of a
policy run to <dir>/ep_*.npz (state, image, executed action, per-step
is_intervention flag). Behaviour cloning only sees states the human's own
demos visited, so a policy that drifts off them has no answer there. The
steps where the HUMAN took over during a policy run are labelled corrections
at exactly the states the policy reached; each contiguous human segment is
added to the training set (BC_TRAJ_WEIGHT times, default 3, since there are
few of them next to the demos).

Env vars: BC_TRAJ_DIRS, BC_TRAJ_WEIGHT, BC_STEPS, BC_OUT, BC_BATCH (64), BC_SKIP_BASE, BC_EXTRA_DATASETS
(comma-separated dataset roots), BC_SUCCESS_ONLY (1), BC_TRIM_IDLE (1),
BC_IDLE_THRESH (0.02), BC_SMOOTH (1 = off; k = centered moving average of the
training targets over k ticks), BC_HOLDOUT_EVERY (8; 0 = no hold-out),
BC_EVAL_EVERY (250).

BC_ACTION_BOUNDS (2026-09-30, default "none" -- exactly reproduces every
prior BC run's behavior unchanged): SACPolicy's actor always squashes its
output through tanh, i.e. its native action range is (-1, 1) by
construction (see policy_kwargs.std_max's comment in the yaml). The
original 3-dim EE-delta action convention is a normalized per-axis
multiplier already close to that range, so the original clamp(-0.99, 0.99)
"normalization" worked by coincidence. It does NOT work for a joint-space
action in real degrees (e.g. joint4 spans ~65 degrees in demos_v2_jointspace)
-- clamping would crush every target to +-0.99, destroying the signal.
BC_ACTION_BOUNDS=auto computes real per-joint (min, max) from the assembled
training set itself (same "measure real bounds from data" approach already
used for end_effector_bounds elsewhere in this project), pads it by 10% on
each side so real deployment values slightly outside the training range
don't hard-saturate tanh, and normalizes to/from that range in _bc_loss /
_policy_action instead of the old fixed clamp. Set it whenever training on
a joint-space-relabeled dataset (e.g. demos_v2_jointspace); leave unset
("none") for the original EE-delta convention.
"""

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from lerobot_control.hilserl import robot_anvil_openarm  # noqa: F401 -- registration side effect
from lerobot_control.hilserl import robot_anvil_openarm_multi  # noqa: F401 -- registration side effect
from lerobot_control.hilserl import teleop_anvil_openarm_leader  # noqa: F401 -- registration side effect
from lerobot_control.hilserl import teleop_anvil_openarm_multi_leader  # noqa: F401 -- registration side effect

import lerobot.policies.sac.configuration_sac  # noqa: F401,E402 -- registers the "sac" policy choice for draccus
import lerobot.policies.sac.modeling_sac  # noqa: F401,E402

STD_PENALTY_WEIGHT = 0.1
_KEYS = ("observation.state", "observation.images.wrist_r")


def _policy_action(policy, batch: dict, action_mid=None, action_half=None) -> torch.Tensor:
    """The deterministic action the real robot runs (tanh of the mean),
    de-normalized back to the dataset's own action units when
    action_mid/action_half are given (see BC_ACTION_BOUNDS)."""
    actor = policy.actor
    hidden = actor.network(actor.encoder(batch, cache=None, detach=False))
    out = torch.tanh(actor.mean_layer(hidden))
    if action_mid is not None:
        out = out * action_half.to(out.device) + action_mid.to(out.device)
    return out


def _bc_loss(policy, batch: dict, action_mid=None, action_half=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    actor = policy.actor
    obs_enc = actor.encoder(batch, cache=None, detach=False)
    hidden = actor.network(obs_enc)
    mean = actor.mean_layer(hidden)
    std = torch.exp(actor.std_layer(hidden))  # unclamped so the penalty always has gradient
    if action_mid is not None:
        target = ((batch["action"] - action_mid.to(batch["action"].device)) / action_half.to(batch["action"].device)).clamp(-1.0, 1.0)
    else:
        target = batch["action"].clamp(-0.99, 0.99)
    mse = F.mse_loss(torch.tanh(mean), target)
    return mse + STD_PENALTY_WEIGHT * std.mean(), mse, std.mean()


class _Frames(torch.utils.data.Dataset):
    """Selected frames of one LeRobotDataset, with (optionally smoothed)
    training targets. Returns only the tensors BC uses -- the saved
    replay-buffer datasets carry different extra columns, which breaks
    default_collate on a mixed batch."""

    def __init__(self, inner, indices, targets, raw):
        self.inner, self.indices, self.targets, self.raw = inner, indices, targets, raw

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        item = self.inner[int(self.indices[i])]
        out = {k: item[k] for k in _KEYS}
        out["action"] = torch.as_tensor(self.targets[i], dtype=torch.float32)
        out["action_raw"] = torch.as_tensor(self.raw[i], dtype=torch.float32)
        return out


def _episodes(ds, success_only: bool, trim_idle: bool, idle_thresh: float, smooth: int):
    """-> list of (frame_indices, training_targets, raw_actions), one per kept episode."""
    ep = np.array([int(x) for x in ds.hf_dataset["episode_index"]])
    # Datasets relabeled purely for action-space conversion (e.g.
    # relabel_joint_space.py's demos_v2_jointspace) don't carry a
    # next.reward column -- they were never given one, since their source
    # (demos_v2) is already curated to be all-successful episodes by
    # construction, with no reward-based filtering ever needed. Skip the
    # success filter gracefully in that case rather than crash on a
    # missing column.
    rew = (
        np.array([float(np.array(x).reshape(-1)[0]) for x in ds.hf_dataset["next.reward"]])
        if "next.reward" in ds.hf_dataset.column_names
        else None
    )
    act = np.stack([np.array(x) for x in ds.hf_dataset["action"]]).astype(np.float32)
    kept = []
    for e in sorted(set(ep)):
        idx = np.where(ep == e)[0]
        if success_only and rew is not None and rew[idx].max() <= 0:
            continue
        a = act[idx]
        if trim_idle:
            moving = np.where(np.abs(a).max(1) > idle_thresh)[0]
            first = int(moving[0]) if len(moving) else 0
            idx, a = idx[first:], a[first:]
        kept.append((idx, _smooth(a, smooth), a))
    return kept


def _smooth(a: np.ndarray, k: int) -> np.ndarray:
    """Centered moving average over k ticks (edge-padded); k <= 1 = unchanged."""
    if k <= 1:
        return a
    pad = k // 2
    padded = np.pad(a, ((pad, k - 1 - pad), (0, 0)), mode="edge")
    return np.stack([padded[j : j + len(a)] for j in range(k)]).mean(0)


class _Arrays(torch.utils.data.Dataset):
    """In-memory frames (from actor trajectory files).

    NOT yet generalized to N cameras like _Frames above was (2026-10-02) --
    this path (BC_TRAJ_DIRS, DAgger-style corrections from saved actor
    trajectories) is single-camera only (hardcodes "observation.images.wrist_r"
    below), since save_actor_trajectory() itself only ever saved one
    camera's frames. Not a blocker for the first bimanual BC pretrain (no
    saved trajectories exist yet to correct from), but will need the same
    treatment as _Frames before a bimanual DAgger round is possible.
    """

    def __init__(self, state, image, targets, raw):
        self.state, self.image, self.targets, self.raw = state, image, targets, raw

    def __len__(self):
        return len(self.state)

    def __getitem__(self, i):
        return {
            "observation.state": torch.as_tensor(self.state[i], dtype=torch.float32),
            "observation.images.wrist_r": torch.as_tensor(self.image[i], dtype=torch.float32),
            "action": torch.as_tensor(self.targets[i], dtype=torch.float32),
            "action_raw": torch.as_tensor(self.raw[i], dtype=torch.float32),
        }


def _human_segments(traj_dirs: list[str], smooth: int):
    """Contiguous runs of human-driven steps from actor trajectory files -> _Arrays."""
    import glob

    states, images, targets, raws, n_files, n_segments = [], [], [], [], 0, 0
    for d in traj_dirs:
        for path in sorted(glob.glob(os.path.join(d, "ep_*.npz"))):
            z = np.load(path)
            if "image" not in z.files:
                print(f"[BC] skipping {path}: no images saved", flush=True)
                continue
            n_files += 1
            human = z["is_intervention"] > 0.5
            i = 0
            while i < len(human):
                if not human[i]:
                    i += 1
                    continue
                j = i
                while j < len(human) and human[j]:
                    j += 1
                seg = slice(i, j)
                a = z["action"][seg].astype(np.float32)
                states.append(z["state_deg"][seg])
                images.append(z["image"][seg])
                targets.append(_smooth(a, smooth))
                raws.append(a)
                n_segments += 1
                i = j
    if not states:
        return None, n_files, 0
    ds = _Arrays(np.concatenate(states), np.concatenate(images), np.concatenate(targets), np.concatenate(raws))
    return ds, n_files, n_segments


@torch.no_grad()
def _evaluate(policy, loader, device, action_mid=None, action_half=None) -> tuple[np.ndarray, float]:
    policy.eval()
    pa, ra = [], []
    for raw in loader:
        batch = {k: v.to(device) for k, v in raw.items() if k in _KEYS}
        pa.append(_policy_action(policy, batch, action_mid, action_half).cpu().numpy())
        ra.append(raw["action_raw"].numpy())
    policy.train()
    pa, ra = np.concatenate(pa), np.concatenate(ra)
    n_dims = pa.shape[1]
    corr = np.array([np.corrcoef(pa[:, k], ra[:, k])[0, 1] if pa[:, k].std() > 1e-9 else 0.0 for k in range(n_dims)])
    # pa is already de-normalized back to the dataset's own action units when
    # action_mid/action_half were passed to _policy_action, so ra needs no
    # clipping in that case -- both sides are in the same real units already.
    ra_cmp = ra if action_mid is not None else np.clip(ra, -0.99, 0.99)
    mse = float(((pa - ra_cmp) ** 2).mean())
    return corr, mse


def bc_pretrain(cfg) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy

    env = os.environ.get
    steps = int(env("BC_STEPS", "1500"))
    out_dir = env("BC_OUT", "outputs/bc_pretrained")
    batch_size = int(env("BC_BATCH", "64"))
    success_only = env("BC_SUCCESS_ONLY", "1") == "1"
    trim_idle = env("BC_TRIM_IDLE", "1") == "1"
    idle_thresh = float(env("BC_IDLE_THRESH", "0.02"))
    smooth = int(env("BC_SMOOTH", "1"))
    holdout_every = int(env("BC_HOLDOUT_EVERY", "8"))
    eval_every = int(env("BC_EVAL_EVERY", "250"))
    action_bounds_mode = env("BC_ACTION_BOUNDS", "none")
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} already exists; pick a new BC_OUT")

    device = torch.device(cfg.policy.device)
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
    policy.train()

    # Real bimanual finding (2026-10-02): _KEYS was hardcoded to
    # single-arm's one camera ("observation.images.wrist_r") -- this script
    # was never generalized to N cameras. Crashed with
    # `KeyError: 'observation.images.wrist_l'` inside SACPolicy's own
    # get_cached_image_features() the first time this ran against a
    # 4-camera bimanual config: _Frames.__getitem__ only ever selected the
    # one hardcoded key out of the real dataset item (which already has
    # every camera's decoded frame -- LeRobotDataset.__getitem__ is
    # generic), so the batch handed to the policy's encoder was missing 3
    # of its 4 expected image keys. Derive the real camera keys from the
    # already-constructed policy itself (policy.actor.encoder.image_keys,
    # the exact same list get_cached_image_features() iterates) rather than
    # guessing -- generalizes correctly to 1 camera (single-arm, unchanged
    # behavior) or N (bimanual).
    global _KEYS
    _KEYS = ("observation.state", *policy.actor.encoder.image_keys)

    sources = []
    if env("BC_SKIP_BASE") != "1":  # BC_SKIP_BASE=1: train ONLY on BC_EXTRA_DATASETS
        sources.append(
            LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, video_backend=cfg.dataset.video_backend)
        )
    for i, root in enumerate(p for p in env("BC_EXTRA_DATASETS", "").split(",") if p):
        extra = LeRobotDataset(f"local/bc_extra_{i}", root=root, video_backend=cfg.dataset.video_backend)
        print(f"[BC] extra dataset {root}: {extra.num_episodes} episodes, {len(extra)} frames", flush=True)
        sources.append(extra)

    train_parts, hold_parts, all_raw, n_kept, n_total = [], [], [], 0, 0
    for ds in sources:
        n_total += ds.num_episodes
        for idx, tgt, raw in _episodes(ds, success_only, trim_idle, idle_thresh, smooth):
            is_holdout = holdout_every > 0 and n_kept % holdout_every == holdout_every - 1
            (hold_parts if is_holdout else train_parts).append(_Frames(ds, idx, tgt, raw))
            all_raw.append(raw)
            n_kept += 1
    traj_dirs = [d for d in env("BC_TRAJ_DIRS", "").split(",") if d]
    if traj_dirs:
        corr_ds, n_files, n_seg = _human_segments(traj_dirs, smooth)
        weight = int(env("BC_TRAJ_WEIGHT", "3"))
        if corr_ds is None:
            print(f"[BC] no human-driven steps found in {n_files} trajectory files under {traj_dirs}", flush=True)
        else:
            print(
                f"[BC] corrections: {len(corr_ds)} human-driven steps in {n_seg} segments from {n_files} episodes "
                f"(each repeated x{weight})",
                flush=True,
            )
            train_parts.extend([corr_ds] * weight)
    train_ds = torch.utils.data.ConcatDataset(train_parts)
    print(
        f"[BC] {n_kept}/{n_total} episodes kept (success_only={success_only}, trim_idle={trim_idle}, smooth={smooth}); "
        f"train {len(train_parts)} eps / {len(train_ds)} frames, held-out {len(hold_parts)} eps",
        flush=True,
    )

    action_mid = action_half = None
    if action_bounds_mode == "auto":
        raw_all = np.concatenate(all_raw)
        lo, hi = raw_all.min(0), raw_all.max(0)
        pad = 0.10 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        action_mid = torch.as_tensor((hi + lo) / 2.0, dtype=torch.float32)
        action_half = torch.as_tensor((hi - lo) / 2.0, dtype=torch.float32)
        print(
            f"[BC] BC_ACTION_BOUNDS=auto -- normalizing {len(action_mid)}-dim action to (-1,1) via per-dim "
            f"bounds measured from {len(raw_all)} training frames (10% padded): "
            f"lo={np.round(lo, 2).tolist()} hi={np.round(hi, 2).tolist()}",
            flush=True,
        )
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    hold_loader = (
        torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(hold_parts), batch_size=128, num_workers=2)
        if hold_parts
        else None
    )
    # A fixed slice of TRAIN frames scored the same way, to make over-fitting visible.
    train_probe = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_ds, list(range(0, len(train_ds), max(1, len(train_ds) // 300)))),
        batch_size=128,
        num_workers=2,
    )

    opt = torch.optim.Adam(list(policy.actor.parameters()), lr=3e-4)
    best_score, best_state, step = -9.0, None, 0
    while step < steps:
        for raw in loader:
            batch = {k: v.to(device) for k, v in raw.items() if k in _KEYS + ("action",)}
            loss, mse, std = _bc_loss(policy, batch, action_mid, action_half)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if step % 50 == 0 or step == 1:
                print(f"[BC] step {step}/{steps} loss {loss.item():.4f} mse {mse.item():.4f} mean_std {std.item():.4f}", flush=True)
            if step % eval_every == 0 or step == steps:
                tc, _ = _evaluate(policy, train_probe, device, action_mid, action_half)
                if hold_loader is not None:
                    hc, hm = _evaluate(policy, hold_loader, device, action_mid, action_half)
                    score = float(hc.mean())
                    tag = ""
                    if score > best_score:
                        best_score, tag = score, "  <- best so far"
                        best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                    print(f"[BC-EVAL] step {step}: HELD-OUT corr per-dim = {np.round(hc, 3)} (mean {score:+.3f}) mse {hm:.4f} | train corr = {np.round(tc, 3)}{tag}", flush=True)
                else:
                    print(f"[BC-EVAL] step {step}: train corr = {np.round(tc, 3)} (no held-out set)", flush=True)
            if step >= steps:
                break

    if best_state is not None:
        policy.load_state_dict(best_state)
        print(f"[BC] restored best held-out checkpoint (mean corr {best_score:+.3f})", flush=True)
    policy.save_pretrained(out_dir)
    if action_mid is not None:
        # patch_sac_deterministic_inference() (patches.py) is the ONLY real
        # inference call site (real-robot stepping) -- it returns raw
        # tanh(means) in (-1,1) with no de-normalization, since it's a
        # single global patch shared by every SACPolicy checkpoint, most of
        # which (the original EE-delta convention) need no de-normalization
        # at all. Save the bounds alongside the checkpoint so a loader at
        # eval time (see run_policy_eval.py's load_sac_action_bounds()) can
        # attach them to THIS policy instance specifically -- without this
        # file, deploying this checkpoint would send every joint to ~0+-1
        # degree, the same missing-normalization bug already found and
        # fixed for ACT earlier tonight.
        import json

        with open(os.path.join(out_dir, "action_bounds.json"), "w") as f:
            json.dump({"mid": action_mid.tolist(), "half": action_half.tolist()}, f)
        print(f"[BC] saved action_bounds.json ({len(action_mid)}-dim) alongside the checkpoint", flush=True)
    print(f"[BC] saved pretrained policy to {out_dir}", flush=True)


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
