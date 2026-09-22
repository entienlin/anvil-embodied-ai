"""Isolated test for the SAC "shaky motion / BC differences washed out"
fixes (patch_sac_deterministic_inference, patch_sac_freeze_temperature):

1. select_action() is deterministic (bit-identical across repeated calls
   on the same batch) and equals torch.tanh(means) computed independently.
2. compute_loss_temperature() returns 0.0 and its .backward() leaves
   log_alpha.grad as a real all-zero tensor (not None, doesn't raise) --
   the exact crash this patch must avoid (log_alpha.requires_grad stays
   True the whole time).
3. compute_loss_actor()/compute_loss_critic() are UNCHANGED -- still
   produce nonzero, properly-connected gradients through the actor/critic
   networks (training itself must not be touched by either patch).

No ROS2/hardware needed -- constructs a bare SACPolicy directly from a
minimal SACConfig matching this task's real feature shapes.

Run: python3 -m lerobot_control.hilserl._stage7a_sac_deterministic_and_frozen_temp_test
"""

import sys

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy

from lerobot_control.hilserl.patches import patch_sac_deterministic_inference, patch_sac_freeze_temperature


def _make_policy() -> SACPolicy:
    cfg = SACConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.wrist_r": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        device="cpu",
    )
    return SACPolicy(cfg)


def _batch() -> dict:
    return {
        "observation.state": torch.rand(2, 7),
        "observation.images.wrist_r": torch.rand(2, 3, 128, 128),
    }


def main() -> int:
    patch_sac_deterministic_inference()
    patch_sac_freeze_temperature()
    policy = _make_policy()
    policy.eval()

    print("[stage7a] select_action() is deterministic and equals tanh(means)...", flush=True)
    batch = _batch()
    a1 = policy.select_action(batch)
    a2 = policy.select_action(batch)
    if not torch.equal(a1, a2):
        print(f"[FAIL] select_action() not deterministic: {a1} vs {a2}", flush=True)
        return 1
    with torch.no_grad():
        obs_features = None
        if policy.shared_encoder and policy.actor.encoder.has_images:
            obs_features = policy.actor.encoder.get_cached_image_features(batch)
        _, _, means = policy.actor(batch, obs_features)
        expected = torch.tanh(means)
    if not torch.allclose(a1, expected):
        print(f"[FAIL] select_action() != tanh(means): {a1} vs {expected}", flush=True)
        return 1
    print("[OK] select_action() is deterministic and equals tanh(means)", flush=True)

    print("[stage7a] compute_loss_temperature() is frozen (0.0, real zero grad, no crash)...", flush=True)
    ACTION = "action"
    train_batch = {"state": _batch()}
    train_batch[ACTION] = torch.rand(2, 3) * 2 - 1
    train_batch["reward"] = torch.rand(2)
    train_batch["next_state"] = _batch()
    train_batch["done"] = torch.zeros(2)

    if not policy.log_alpha.requires_grad:
        print("[FAIL] log_alpha.requires_grad is False -- would crash .backward() elsewhere", flush=True)
        return 1

    policy.train()
    loss_temperature = policy.compute_loss_temperature(observations=train_batch["state"])
    if loss_temperature.item() != 0.0:
        print(f"[FAIL] loss_temperature != 0.0: {loss_temperature.item()}", flush=True)
        return 1
    policy.log_alpha.grad = None
    loss_temperature.backward()
    if policy.log_alpha.grad is None:
        print("[FAIL] log_alpha.grad is None after backward() -- optimizer.step() would silently skip it", flush=True)
        return 1
    if policy.log_alpha.grad.item() != 0.0:
        print(f"[FAIL] log_alpha.grad != 0.0: {policy.log_alpha.grad.item()}", flush=True)
        return 1
    print("[OK] loss_temperature == 0.0, backward() succeeds, log_alpha.grad is a real zero tensor", flush=True)

    print("[stage7a] compute_loss_actor()/compute_loss_critic() are unaffected...", flush=True)
    actor_loss = policy.compute_loss_actor(observations=train_batch["state"])
    if not actor_loss.requires_grad:
        print("[FAIL] actor_loss.requires_grad is False -- training would be broken", flush=True)
        return 1
    actor_loss.backward()
    actor_params = [p for n, p in policy.actor.named_parameters() if not n.startswith("encoder")]
    if not any(p.grad is not None and torch.any(p.grad != 0) for p in actor_params):
        print("[FAIL] no nonzero gradient reached the actor's own parameters", flush=True)
        return 1

    critic_loss = policy.compute_loss_critic(
        observations=train_batch["state"],
        actions=train_batch[ACTION],
        rewards=train_batch["reward"],
        next_observations=train_batch["next_state"],
        done=train_batch["done"],
    )
    if not critic_loss.requires_grad:
        print("[FAIL] critic_loss.requires_grad is False -- training would be broken", flush=True)
        return 1
    critic_loss.backward()
    if not any(
        p.grad is not None and torch.any(p.grad != 0) for p in policy.critic_ensemble.parameters()
    ):
        print("[FAIL] no nonzero gradient reached the critic's parameters", flush=True)
        return 1
    print("[OK] actor/critic losses still produce real, connected gradients", flush=True)

    print("[stage7a] re-applying both patches is a no-op...", flush=True)
    patch_sac_deterministic_inference()
    patch_sac_freeze_temperature()
    print("[OK] idempotent", flush=True)

    print("[stage7a] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
