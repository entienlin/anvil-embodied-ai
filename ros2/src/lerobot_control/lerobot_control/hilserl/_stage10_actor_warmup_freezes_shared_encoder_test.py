"""patch_sac_actor_warmup(): confirms both real gaps found and fixed in
this patch's history.

(1) 2026-10-08: when SACConfig.shared_encoder is True (the default,
unoverridden by this task), the shared encoder's parameters must ALSO be
frozen (requires_grad False) for the warm-up window, not just
optimizers["actor"].step(). Without this, optimizers["critic"].step()
(never warmup-gated) keeps updating the shared representation the
actor's forward pass depends on, letting the actor's effective OUTPUT
drift even while its own optimizer never takes a step -- confirmed
directly on real hardware: episode 2 collapsed to near-zero commanded
deltas after only ~100 optimization steps, well under a 200-step warmup
budget.

(2) 2026-10-12: skipping optimizers["actor"].step() entirely (rather
than calling it with a near-zero LR) pins Adam's own internal step
counter at 0 for the whole warm-up window regardless of its length, so
the first unskipped call is always Adam's maximally-aggressive
cold-start step (step_size = lr / (1 - beta1**step), largest at
step=1). Confirmed directly on real hardware: with shared-encoder
freezing already active and HILSERL_ACTOR_WARMUP_STEPS=200, the policy
still collapsed almost exactly at the warm-up boundary. Fixed via a
linear LR ramp (near-zero -> base_lr), calling the real optimizer step
on every call instead of skipping it.

Pure logic, no hardware/torch-Adam-internals needed -- a fake optimizer
with real param_groups is enough to verify the LR-ramp math and the
encoder freeze/unfreeze timing.

Loads patches.py directly by file path (not via the lerobot_control.hilserl
package __init__, which eagerly imports rclpy-dependent modules this test
doesn't need and the host venv doesn't have installed).

Run: python3 _stage10_actor_warmup_freezes_shared_encoder_test.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import torch


def _load_patches_module():
    path = Path(__file__).resolve().parent / "patches.py"
    spec = importlib.util.spec_from_file_location("_anvil_patches_standalone10", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeOptimizer:
    def __init__(self, base_lr=1e-3):
        self.param_groups = [{"lr": base_lr}]
        self.step_calls = 0
        self.lr_at_each_step_call = []

    def step(self, *a, **kw):
        self.step_calls += 1
        self.lr_at_each_step_call.append(self.param_groups[0]["lr"])


class _FakeEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class _FakePolicy:
    def __init__(self):
        self.shared_encoder = True
        self.encoder_critic = _FakeEncoder()


def main() -> None:
    os.environ["HILSERL_ACTOR_WARMUP_STEPS"] = "5"

    import lerobot.rl.learner as learner_mod

    patches = _load_patches_module()

    policy = _FakePolicy()
    encoder_params = list(policy.encoder_critic.parameters())
    assert all(p.requires_grad for p in encoder_params), "sanity: starts trainable"

    fake_actor_opt = _FakeOptimizer(base_lr=1e-3)

    def _fake_orig_make(cfg, policy):
        return {"actor": fake_actor_opt}, None

    learner_mod.make_optimizers_and_scheduler = _fake_orig_make

    patches.patch_sac_actor_warmup()

    optimizers, _ = learner_mod.make_optimizers_and_scheduler(cfg=None, policy=policy)
    actor_opt = optimizers["actor"]

    assert all(not p.requires_grad for p in encoder_params), (
        "shared-encoder params must be frozen as soon as warm-up starts"
    )
    print("[OK] shared-encoder parameters frozen at warm-up start")

    # All 5 warm-up calls must be REAL .step() calls (never skipped) at a
    # linearly ramping, strictly-below-base LR -- this is the actual
    # 2026-10-12 fix: Adam's own counter must advance normally.
    for i in range(1, 5):
        actor_opt.step()
        assert fake_actor_opt.step_calls == i, f"call {i} must be a REAL step, not skipped"
        lr = fake_actor_opt.lr_at_each_step_call[-1]
        assert 0 < lr < 1e-3, f"call {i}: lr {lr} must be a small positive fraction of base_lr, not 0 or full"
        assert all(not p.requires_grad for p in encoder_params), f"encoder must stay frozen during ramp (call {i})"
    print("[OK] first 4 of 5 warm-up calls: real (non-skipped) steps at a ramping sub-base LR, encoder frozen")

    # 5th call: ramp completes -- LR restored to exactly base_lr, encoder unfreezes.
    actor_opt.step()
    assert fake_actor_opt.step_calls == 5
    assert fake_actor_opt.lr_at_each_step_call[-1] == 1e-3, "ramp must land exactly on base_lr at the boundary"
    assert all(p.requires_grad for p in encoder_params), "shared-encoder must unfreeze exactly when ramp completes"
    print("[OK] ramp reaches exactly base_lr and encoder unfreezes on the 5th (final warm-up) call")

    # 6th call onward: full LR, no further ramp logic touches lr.
    actor_opt.step()
    assert fake_actor_opt.step_calls == 6
    assert fake_actor_opt.lr_at_each_step_call[-1] == 1e-3
    assert all(p.requires_grad for p in encoder_params)
    print("[OK] post-warm-up calls stay at full base_lr, encoder remains unfrozen")

    print("\nAll checks passed.")


if __name__ == "__main__":
    sys.exit(main())
