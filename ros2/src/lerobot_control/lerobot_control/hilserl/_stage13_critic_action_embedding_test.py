"""patch_sac_critic_action_embedding() + patch_sac_tolerates_shape_mismatch_on_load():

1. Unset/0 -> architecture byte-for-byte unchanged (no action_embed
   attribute, critic's first layer sized for raw action_dim).
2. Set -> critic_ensemble/critic_target share the SAME action_embed
   object (not a copy), critic_forward() runs with the new shape,
   action_embed's params are included in critic_ensemble.parameters()
   (so the real optimizer_critic = Adam(critic_ensemble.parameters())
   picks them up with zero extra wiring), and update_target_networks()
   is an exact no-op on action_embed's own weights (Polyak-blending a
   tensor with itself).
3. A checkpoint saved WITHOUT the embedding loads cleanly into a policy
   constructed WITH it enabled -- shape-mismatched critic keys are
   skipped (fresh/random), everything else (actor) loads exactly.

No ROS2/hardware needed -- bare SACPolicy from a minimal SACConfig.

Run: python3 -m lerobot_control.hilserl._stage13_critic_action_embedding_test
"""

import os
import sys
import tempfile

import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.modeling_sac import SACPolicy

from lerobot_control.hilserl import patches


def _make_config() -> SACConfig:
    return SACConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            "observation.images.wrist_r": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 128, 128)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))},
        device="cpu",
    )


def _batch(n=2) -> dict:
    return {
        "observation.state": torch.rand(n, 7),
        "observation.images.wrist_r": torch.rand(n, 3, 128, 128),
    }


def main() -> int:
    os.environ.pop("HILSERL_CRITIC_ACTION_EMBED_DIM", None)
    patches.patch_sac_critic_action_embedding()
    policy_off = SACPolicy(_make_config())
    if hasattr(policy_off.critic_ensemble, "action_embed"):
        print("[FAIL] action_embed present with the feature unset -- not a no-op", flush=True)
        return 1
    orig_first_layer = policy_off.critic_ensemble.critics[0].net.net[0]
    expected_raw_input = policy_off.encoder_critic.output_dim + 3
    if orig_first_layer.weight.shape[1] != expected_raw_input:
        print(f"[FAIL] unset case: critic first layer input_dim {orig_first_layer.weight.shape[1]} "
              f"!= expected {expected_raw_input}", flush=True)
        return 1
    print("[OK] HILSERL_CRITIC_ACTION_EMBED_DIM unset -- architecture unchanged", flush=True)

    os.environ["HILSERL_CRITIC_ACTION_EMBED_DIM"] = "16"
    import lerobot.policies.sac.modeling_sac as sac_mod

    sac_mod.SACPolicy._anvil_critic_action_embed_init_patched = False
    patches.patch_sac_critic_action_embedding()

    policy_on = SACPolicy(_make_config())
    ce, ct = policy_on.critic_ensemble, policy_on.critic_target
    if not hasattr(ce, "action_embed") or not hasattr(ct, "action_embed"):
        print("[FAIL] action_embed missing with the feature enabled", flush=True)
        return 1
    if ce.action_embed is not ct.action_embed:
        print("[FAIL] critic_ensemble and critic_target should share the SAME action_embed object", flush=True)
        return 1
    print("[OK] action_embed present, shared (same object) between ensemble and target", flush=True)

    new_first_layer = ce.critics[0].net.net[0]
    expected_embed_input = policy_on.encoder_critic.output_dim + 16
    if new_first_layer.weight.shape[1] != expected_embed_input:
        print(f"[FAIL] enabled case: critic first layer input_dim {new_first_layer.weight.shape[1]} "
              f"!= expected {expected_embed_input}", flush=True)
        return 1
    print("[OK] critic's first layer sized for embed_dim, not raw action_dim", flush=True)

    with torch.no_grad():
        q = policy_on.critic_forward(_batch(), torch.rand(2, 3), use_target=False)
    if q.shape != (2, 2):  # (num_critics, batch)
        print(f"[FAIL] critic_forward() with embedding enabled returned shape {q.shape}, expected (2, 2)", flush=True)
        return 1
    print("[OK] critic_forward() runs end-to-end with the embedding enabled", flush=True)

    embed_param_ids = {id(p) for p in ce.action_embed.parameters()}
    ensemble_param_ids = {id(p) for p in ce.parameters()}
    if not embed_param_ids.issubset(ensemble_param_ids):
        print("[FAIL] action_embed's parameters are not included in critic_ensemble.parameters() -- "
              "the real critic optimizer would never train them", flush=True)
        return 1
    print("[OK] action_embed's parameters are included in critic_ensemble.parameters() "
          "(optimizer_critic picks them up with zero extra wiring)", flush=True)

    before = [p.detach().clone() for p in ce.action_embed.parameters()]
    policy_on.config.critic_target_update_weight = 0.3  # arbitrary, must not matter for a self-blend
    policy_on.update_target_networks()
    after = list(ce.action_embed.parameters())
    # allclose, not equal: blending a tensor with itself (x*tau + x*(1-tau))
    # is an exact no-op ALGEBRAICALLY, but each multiply/add still rounds
    # independently in float32 -- confirmed directly, ~1e-8 max abs diff,
    # pure rounding noise, not a real pairing/ordering bug.
    if not all(torch.allclose(b, a, atol=1e-6) for b, a in zip(before, after, strict=True)):
        print("[FAIL] update_target_networks() meaningfully changed action_embed's own weights -- "
              "should be a no-op (Polyak-blending a tensor with itself), not a real pairing bug", flush=True)
        return 1
    print("[OK] update_target_networks() is a no-op (up to float32 rounding) on action_embed's shared weights", flush=True)

    # --- shape-mismatch-tolerant loading: save WITHOUT embedding, load WITH it ---
    with tempfile.TemporaryDirectory() as d:
        policy_off.save_pretrained(d)
        try:
            loaded = SACPolicy.from_pretrained(d, config=_make_config())
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] loading a pre-embedding checkpoint under the new architecture raised: {e!r}", flush=True)
            return 1
        if not hasattr(loaded.critic_ensemble, "action_embed"):
            print("[FAIL] loaded policy should still have the new architecture (action_embed present)", flush=True)
            return 1
        # Actor shapes are untouched by this change -- confirm it actually loaded
        # the OLD weights exactly, not just "didn't crash".
        for (name_old, p_old), (name_new, p_new) in zip(
            policy_off.actor.named_parameters(), loaded.actor.named_parameters(), strict=True
        ):
            if not torch.equal(p_old, p_new):
                print(f"[FAIL] actor parameter {name_old} did not load from the old checkpoint "
                      f"(unaffected by this change, should match exactly)", flush=True)
                return 1
        print("[OK] a pre-embedding checkpoint loads cleanly under the new architecture -- "
              "actor weights match exactly, critic's reshaped layers started fresh instead of crashing", flush=True)

    print("\n[stage13] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
