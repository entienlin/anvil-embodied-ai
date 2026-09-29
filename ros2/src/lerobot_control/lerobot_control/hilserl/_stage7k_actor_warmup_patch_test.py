"""patch_sac_actor_warmup(): HILSERL_ACTOR_WARMUP_STEPS=N must skip the
first N calls to optimizers["actor"].step() (critic-only warm-up), then let
every call afterward through unchanged; unset/0 must leave step() untouched
entirely (today's prior behaviour). Pure logic, no ROS2/hardware/real optimizer.

Run: python3 -m lerobot_control.hilserl._stage7k_actor_warmup_patch_test
"""

import os
import sys

import lerobot.rl.learner as learner_mod

from lerobot_control.hilserl import patches


class _FakeOptimizer:
    def __init__(self):
        self.step_calls = 0

    def step(self, *a, **kw):
        self.step_calls += 1


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

    for i in range(3):
        actor_opt.step()
        if actor_opt.step_calls != 0:
            print(f"[FAIL] call {i + 1}/3 during warm-up should not reach the real optimizer (step_calls={actor_opt.step_calls})", flush=True)
            return 1
    print("[OK] first 3 calls (the warm-up window) never reach the real actor optimizer", flush=True)

    actor_opt.step()
    if actor_opt.step_calls != 1:
        print(f"[FAIL] call 4 should be the first real step (step_calls={actor_opt.step_calls})", flush=True)
        return 1
    actor_opt.step()
    if actor_opt.step_calls != 2:
        print("[FAIL] calls after warm-up should all reach the real optimizer", flush=True)
        return 1
    print("[OK] every call after the warm-up window reaches the real actor optimizer", flush=True)

    if optimizers["critic"].step_calls != 0 or optimizers["temperature"].step_calls != 0:
        print("[FAIL] critic/temperature optimizers must be completely untouched by this patch", flush=True)
        return 1
    print("[OK] critic/temperature optimizers are untouched (never wrapped)", flush=True)

    print("[stage7k] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
