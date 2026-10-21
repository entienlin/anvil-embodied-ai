"""patch_sac_actor_held_exploration_noise(): adds a small, TEMPORALLY-HELD
random perturbation to select_action()'s deployed output -- resampled only
every HILSERL_EXPLORE_HOLD_TICKS calls, not freshly every tick -- so the
critic gets real (state, action) diversity in the replay buffer without
reproducing the per-tick jitter patch_sac_deterministic_inference() exists
to prevent.

1. Default (env unset/0): byte-identical to the deterministic (no-noise)
   select_action() output.
2. Nonzero std, no action_bounds attached: perturbation is added and held
   CONSTANT across consecutive calls within the hold window, then
   resampled to a new value once the window elapses. Result stays
   clamped to [-1, 1].
3. With _anvil_action_mid/_anvil_action_half attached (joint-space-style
   checkpoint): perturbation magnitude scales with the dimension's own
   half-range, and the result is clamped to [mid-half, mid+half].
4. num_discrete_actions is not None: perturbation is skipped entirely
   (never touches a discrete action slot).
5. Re-applying the patch is a no-op (idempotent).

No ROS2/hardware needed -- bare SACPolicy from a minimal SACConfig.

Run: python3 -m lerobot_control.hilserl._stage16_held_exploration_noise_test
"""

import os
import sys

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


def _batch(n=1) -> dict:
    return {
        "observation.state": torch.rand(n, 7),
        "observation.images.wrist_r": torch.rand(n, 3, 128, 128),
    }


def main() -> int:
    os.environ.pop("HILSERL_EXPLORE_NOISE_STD", None)
    os.environ.pop("HILSERL_EXPLORE_HOLD_TICKS", None)
    patches.patch_sac_deterministic_inference()
    patches.patch_sac_actor_held_exploration_noise()

    policy = SACPolicy(_make_config())
    policy.eval()  # dropout in the image post-encoder is otherwise stochastic even "deterministic" inference
    torch.manual_seed(0)
    obs = _batch()

    # --- default (env unset -> 0.0): must match the deterministic baseline exactly ---
    base1 = policy.select_action(obs)
    base2 = policy.select_action(obs)
    if not torch.equal(base1, base2):
        print("[FAIL] default: deterministic select_action() should be exactly repeatable", flush=True)
        return 1
    print("[OK] default (HILSERL_EXPLORE_NOISE_STD unset) -- deterministic, unaffected", flush=True)

    # --- nonzero std, no action_bounds: held across the window, clamped to [-1,1] ---
    os.environ["HILSERL_EXPLORE_NOISE_STD"] = "0.5"
    os.environ["HILSERL_EXPLORE_HOLD_TICKS"] = "3"
    for attr in ("_anvil_explore_perturbation", "_anvil_explore_ticks_left"):
        if hasattr(policy, attr):
            delattr(policy, attr)

    out1 = policy.select_action(obs)
    out2 = policy.select_action(obs)
    out3 = policy.select_action(obs)
    if not (torch.equal(out1, out2) and torch.equal(out2, out3)):
        print(f"[FAIL] perturbation should stay HELD (identical) across the 3-tick window, "
              f"got out1={out1}, out2={out2}, out3={out3}", flush=True)
        return 1
    if torch.equal(out1, base1):
        print("[FAIL] perturbed output should differ from the unperturbed deterministic baseline "
              "(std=0.5 should essentially never land on exactly zero noise)", flush=True)
        return 1
    if out1.min() < -1.0 - 1e-5 or out1.max() > 1.0 + 1e-5:
        print(f"[FAIL] perturbed output should stay clamped to [-1,1], got {out1}", flush=True)
        return 1
    print("[OK] nonzero std: perturbation held constant across the hold window, differs from "
          "baseline, clamped to [-1,1]", flush=True)

    out4 = policy.select_action(obs)  # 4th call -- window of 3 elapsed, must resample
    if torch.equal(out3, out4):
        print("[WARN] 4th call (past the 3-tick hold window) returned the identical perturbation -- "
              "possible but astronomically unlikely by chance with fresh random noise; treating as fail", flush=True)
        return 1
    print("[OK] perturbation resamples to a new value once the hold window elapses", flush=True)

    # --- with action_bounds attached: scale by half, clamp to [mid-half, mid+half] ---
    policy._anvil_action_mid = torch.tensor([10.0, -5.0, 20.0])
    policy._anvil_action_half = torch.tensor([40.0, 15.0, 30.0])
    for attr in ("_anvil_explore_perturbation", "_anvil_explore_ticks_left"):
        delattr(policy, attr)
    os.environ["HILSERL_EXPLORE_NOISE_STD"] = "0.1"
    bounded_out = policy.select_action(obs)
    low = policy._anvil_action_mid - policy._anvil_action_half
    high = policy._anvil_action_mid + policy._anvil_action_half
    if bounded_out.min() < low.min() - 1e-4 or (bounded_out.max(dim=0).values > high).any():
        print(f"[FAIL] with action_bounds attached, output {bounded_out} exceeds "
              f"[{low}, {high}]", flush=True)
        return 1
    print("[OK] with _anvil_action_mid/_anvil_action_half attached, perturbed output stays "
          "within [mid-half, mid+half]", flush=True)
    del policy._anvil_action_mid
    del policy._anvil_action_half

    # --- discrete actions: perturbation must be skipped entirely ---
    # (needs its own policy instance -- discrete_critic is only built in
    # __init__ when num_discrete_actions is set at CONSTRUCTION time, not
    # something that can be flipped on an existing instance's config)
    discrete_cfg = _make_config()
    discrete_cfg.num_discrete_actions = 4
    discrete_policy = SACPolicy(discrete_cfg)
    discrete_policy.eval()
    try:
        discrete_out = discrete_policy.select_action(obs)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] select_action() with num_discrete_actions set raised: {e!r}", flush=True)
        return 1
    # discrete branch appends a 4th column (the argmax discrete action) -- shape must reflect that,
    # and must be untouched by any exploration perturbation logic (no crash, no extra noise column).
    if discrete_out.shape[-1] != 4:
        print(f"[FAIL] expected a 4-column output (3 continuous + 1 discrete), got shape {discrete_out.shape}",
              flush=True)
        return 1
    if hasattr(discrete_policy, "_anvil_explore_perturbation"):
        print("[FAIL] exploration perturbation state was set despite num_discrete_actions being set -- "
              "should have returned early, untouched", flush=True)
        return 1
    print("[OK] num_discrete_actions set: select_action() still runs end-to-end, untouched by "
          "the continuous-only exploration perturbation", flush=True)

    # --- idempotent re-apply ---
    patches.patch_sac_actor_held_exploration_noise()
    for attr in ("_anvil_explore_perturbation", "_anvil_explore_ticks_left"):
        if hasattr(policy, attr):
            delattr(policy, attr)
    os.environ["HILSERL_EXPLORE_NOISE_STD"] = "0.0"
    reapplied_out = policy.select_action(obs)
    if not torch.equal(reapplied_out, base1):
        print("[FAIL] re-applying the patch changed behavior with noise back off -- not idempotent", flush=True)
        return 1
    print("[OK] re-applying the patch is a no-op (idempotent)", flush=True)

    print("\n[stage16] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
