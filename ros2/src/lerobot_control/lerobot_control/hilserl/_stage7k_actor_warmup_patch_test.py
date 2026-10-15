"""patch_sac_actor_warmup(): HILSERL_ACTOR_WARMUP_STEPS=N must linearly ramp
optimizers["actor"]'s LR from near-zero up to its base value over the first
N real .step() calls (never skipping the call itself -- redesigned
2026-10-12 from a hard .step()-skip to this ramp, specifically because
skipping pinned Adam's internal step counter at 0 for the whole window,
making the first unskipped call always a maximal cold-start step
regardless of N). unset/0 must leave step() untouched entirely. Pure
logic, no ROS2/hardware/real optimizer.

Run: python3 -m lerobot_control.hilserl._stage7k_actor_warmup_patch_test
"""

import os
import sys

import lerobot.rl.learner as learner_mod

from lerobot_control.hilserl import patches


class _FakeOptimizer:
    def __init__(self, base_lr=0.001):
        self.step_calls = 0
        self.param_groups = [{"lr": base_lr}]
        self.lr_at_last_step = []

    def step(self, *a, **kw):
        self.step_calls += 1
        self.lr_at_last_step.append(self.param_groups[0]["lr"])


def _fresh_optimizers():
    return {"actor": _FakeOptimizer(), "critic": _FakeOptimizer(), "temperature": _FakeOptimizer()}


def main() -> int:
    # --- unset (default): make_optimizers_and_scheduler must be left untouched ---
    os.environ.pop("HILSERL_ACTOR_WARMUP_STEPS", None)
    orig_make = learner_mod.make_optimizers_and_scheduler
    patches.patch_sac_actor_warmup()
    if learner_mod.make_optimizers_and_scheduler is not orig_make:
        print("[FAIL] should not wrap make_optimizers_and_scheduler when unset/0", flush=True)
        return 1
    print("[OK] HILSERL_ACTOR_WARMUP_STEPS unset -- patch declined to install (no-op, matches prior behavior)", flush=True)

    # --- set: install fresh, since the first call above returned early and never marked the patch applied ---
    os.environ["HILSERL_ACTOR_WARMUP_STEPS"] = "3"
    learner_mod._anvil_actor_warmup_patched = False
    fake_opts = _fresh_optimizers()
    learner_mod.make_optimizers_and_scheduler = lambda cfg, policy: (fake_opts, None)
    patches.patch_sac_actor_warmup()

    optimizers, _ = learner_mod.make_optimizers_and_scheduler(cfg=None, policy=None)
    actor_opt = optimizers["actor"]
    base_lr = 0.001

    for i in range(3):
        actor_opt.step()
        if actor_opt.step_calls != i + 1:
            print(f"[FAIL] call {i + 1}/3 during warm-up must still reach the real optimizer "
                  f"(never skipped, only LR-scaled) -- step_calls={actor_opt.step_calls}", flush=True)
            return 1
        expected_ramp_lr = base_lr * (i + 1) / 3
        if abs(actor_opt.lr_at_last_step[-1] - expected_ramp_lr) > 1e-9:
            print(f"[FAIL] call {i + 1}/3 LR should be ramped to {expected_ramp_lr}, "
                  f"got {actor_opt.lr_at_last_step[-1]}", flush=True)
            return 1
    print("[OK] first 3 calls (the warm-up window) all reach the real optimizer with linearly ramped LR", flush=True)

    if abs(actor_opt.param_groups[0]["lr"] - base_lr) > 1e-9:
        print(f"[FAIL] LR should be restored to exactly base_lr={base_lr} once the ramp completes, "
              f"got {actor_opt.param_groups[0]['lr']}", flush=True)
        return 1
    print("[OK] LR restored to exactly base_lr once the ramp completes", flush=True)

    actor_opt.step()
    if actor_opt.step_calls != 4 or abs(actor_opt.lr_at_last_step[-1] - base_lr) > 1e-9:
        print("[FAIL] calls after warm-up should reach the real optimizer at full base_lr", flush=True)
        return 1
    actor_opt.step()
    if actor_opt.step_calls != 5 or abs(actor_opt.lr_at_last_step[-1] - base_lr) > 1e-9:
        print("[FAIL] calls after warm-up should all reach the real optimizer at full base_lr", flush=True)
        return 1
    print("[OK] every call after the warm-up window reaches the real actor optimizer at full base_lr", flush=True)

    if optimizers["critic"].step_calls != 0 or optimizers["temperature"].step_calls != 0:
        print("[FAIL] critic/temperature optimizers must be completely untouched by this patch", flush=True)
        return 1
    print("[OK] critic/temperature optimizers are untouched (never wrapped)", flush=True)

    print("[stage7k] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
