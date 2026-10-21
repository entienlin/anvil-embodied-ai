"""patch_sac_actor_bc_anchor(): adds an opt-in
HILSERL_ACTOR_BC_ANCHOR_WEIGHT * MSE(actor_output, executed_action) term
to the actor loss, to directly oppose the real collapse measured from
actual episodes (actor output losing fine, local state-reactivity,
converging toward an almost state-invariant output).

1. Default (env var unset/0): loss_actor must be byte-identical to the
   ORIGINAL, unpatched computation -- no side effect when off.
2. Nonzero weight + a demo_actions batch deliberately far from the
   actor's current output: loss_actor must be measurably HIGHER than the
   weight=0 case, and the difference must match weight * mse(...) exactly.
3. Backward pass succeeds, actor parameters receive nonzero gradients.
4. Normalization: with _anvil_action_mid/_anvil_action_half set (joint-
   space-style checkpoint), raw-scale demo actions get correctly mapped
   into the actor's own (-1,1) tanh scale before the MSE; without those
   attributes (ee_delta-style checkpoint), demo_actions pass through
   unnormalized.
5. Re-applying the patch is a no-op (idempotent).

compute_loss_actor() samples stochastically (self.actor(...).rsample()),
so every comparison below reseeds torch immediately before each top-level
call that triggers it, with the SAME seed, so the two sides being compared
draw identical noise and are directly comparable.

No ROS2/hardware needed -- bare SACPolicy from a minimal SACConfig.

Run: python3 -m lerobot_control.hilserl._stage15_actor_bc_anchor_test
"""

import os
import sys

import torch
import torch.nn.functional as F

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import ACTION, SACPolicy

from lerobot_control.hilserl import patches

SEED = 42


def _make_config() -> SACConfig:
    return SACConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.wrist_r": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        device="cpu",
    )


def _batch(n=4) -> dict:
    return {
        "observation.state": torch.rand(n, 7),
        "observation.images.wrist_r": torch.rand(n, 3, 128, 128),
    }


def main() -> int:
    os.environ.pop("HILSERL_ACTOR_BC_ANCHOR_WEIGHT", None)
    patches.patch_sac_actor_bc_anchor()

    policy = SACPolicy(_make_config())
    torch.manual_seed(0)
    obs = _batch()
    forward_batch = {ACTION: torch.full((4, 3), 5.0), "state": obs, "observation_feature": None}

    # --- default (weight unset -> 0.0): must match the ORIGINAL computation exactly ---
    torch.manual_seed(SEED)
    baseline_loss = policy.compute_loss_actor(observations=obs, observation_features=None, demo_actions=None)
    torch.manual_seed(SEED)
    out = policy.forward(forward_batch, model="actor")
    if not torch.allclose(out["loss_actor"], baseline_loss):
        print(f"[FAIL] weight=0: forward()'s loss_actor ({out['loss_actor'].item()}) should exactly match "
              f"the original no-anchor computation ({baseline_loss.item()})", flush=True)
        return 1
    print("[OK] default (HILSERL_ACTOR_BC_ANCHOR_WEIGHT unset) -- byte-identical to unpatched behavior", flush=True)

    # --- nonzero weight, no action_bounds attached (ee_delta-style passthrough) ---
    os.environ["HILSERL_ACTOR_BC_ANCHOR_WEIGHT"] = "2.0"
    torch.manual_seed(SEED)
    with torch.no_grad():
        actions_pi, _, _ = policy.actor(obs, None)
    demo_far = actions_pi.detach() + 10.0  # deliberately far from the actor's own output
    forward_batch_far = {ACTION: demo_far, "state": obs, "observation_feature": None}
    torch.manual_seed(SEED)
    out_far = policy.forward(forward_batch_far, model="actor")
    expected_bc_term = 2.0 * F.mse_loss(actions_pi, demo_far)
    expected_total = baseline_loss + expected_bc_term
    if not torch.allclose(out_far["loss_actor"], expected_total, atol=1e-5):
        print(f"[FAIL] weight=2.0: got {out_far['loss_actor'].item()}, "
              f"expected baseline+bc_term={expected_total.item()}", flush=True)
        return 1
    print("[OK] nonzero weight: loss_actor == baseline + weight * mse(actor_output, demo_actions), "
          "no action_bounds attached (passthrough, no normalization)", flush=True)

    # --- backward pass must succeed and actually move the actor's parameters ---
    policy.zero_grad()
    out_far["loss_actor"].backward()
    grad_norms = [p.grad.norm().item() for p in policy.actor.parameters() if p.grad is not None]
    if not grad_norms or sum(grad_norms) == 0.0:
        print("[FAIL] backward() produced no gradient on any actor parameter", flush=True)
        return 1
    print(f"[OK] backward() succeeds, {len(grad_norms)} actor param tensors received nonzero gradient", flush=True)

    # --- with action_bounds attached, demo_actions must be normalized before the MSE ---
    policy._anvil_action_mid = torch.zeros(3)
    policy._anvil_action_half = torch.full((3,), 50.0)
    raw_demo = torch.full((4, 3), 25.0)  # raw-scale -> normalized == 25/50 == 0.5
    forward_batch_bounded = {ACTION: raw_demo, "state": obs, "observation_feature": None}

    torch.manual_seed(SEED + 1)
    baseline_loss_2 = policy.compute_loss_actor(observations=obs, observation_features=None, demo_actions=None)
    torch.manual_seed(SEED + 1)
    with torch.no_grad():
        actions_pi2, _, _ = policy.actor(obs, None)
    normalized_demo = (raw_demo - policy._anvil_action_mid) / policy._anvil_action_half
    expected_bounded = baseline_loss_2 + 2.0 * F.mse_loss(actions_pi2, normalized_demo)

    torch.manual_seed(SEED + 1)
    out_bounded = policy.forward(forward_batch_bounded, model="actor")
    if not torch.allclose(out_bounded["loss_actor"], expected_bounded, atol=1e-5):
        print(f"[FAIL] with action_bounds attached: got {out_bounded['loss_actor'].item()}, "
              f"expected {expected_bounded.item()} (raw demo action normalized via mid/half first)", flush=True)
        return 1
    print("[OK] with _anvil_action_mid/_anvil_action_half attached, raw-scale demo actions are "
          "correctly normalized into the actor's (-1,1) scale before the MSE term", flush=True)

    # --- idempotent re-apply ---
    patches.patch_sac_actor_bc_anchor()
    torch.manual_seed(SEED + 1)
    out_again = policy.forward(forward_batch_bounded, model="actor")
    if not torch.allclose(out_again["loss_actor"], out_bounded["loss_actor"]):
        print("[FAIL] re-applying the patch changed behavior -- not idempotent", flush=True)
        return 1
    print("[OK] re-applying the patch is a no-op (idempotent)", flush=True)

    print("\n[stage15] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
