"""patch_learner_resume_reloads_trained_weights(): handle_resume_logic()
swaps in the checkpoint's own saved train_config.json on every --resume,
but that config's policy.pretrained_path was never rewritten by
save_checkpoint() -- it stays frozen at whatever the very first, non-resumed
launch passed (the BC baseline), so every resume silently reconstructed the
actor/critic from BC instead of from wherever training actually left off.
No ROS2/hardware needed -- a bare stand-in for handle_resume_logic() plays
the role of the real (buggy) lerobot function.

Run: python3 -m lerobot_control.hilserl._stage14_resume_reloads_trained_weights_test
"""

import sys
from types import SimpleNamespace

from lerobot_control.hilserl import patches


def _make_cfg(resume: bool, pretrained_path: str, output_dir: str = "/workspace/outputs/learner"):
    return SimpleNamespace(
        resume=resume,
        output_dir=output_dir,
        policy=SimpleNamespace(pretrained_path=pretrained_path),
    )


def main() -> int:
    import lerobot.rl.learner as learner_mod

    STALE_BC_PATH = "/workspace/configs/hilserl/bc_grasp_place_poc"

    def _fake_orig_handle_resume_logic(cfg):
        # Mirrors the REAL bug: whether resuming or not, the ORIGINAL
        # lerobot function returns a cfg whose policy.pretrained_path is
        # whatever was baked into the checkpoint's own saved
        # train_config.json -- the stale BC path, never a self-reference to
        # the checkpoint's own weights.
        if not cfg.resume:
            return cfg
        return _make_cfg(resume=True, pretrained_path=STALE_BC_PATH, output_dir=cfg.output_dir)

    # Stand in for the real lerobot.rl.learner.handle_resume_logic -- our
    # patch captures whatever this currently is as `orig_handle_resume_logic`
    # at patch-apply time, exactly like the real patch does against the real
    # function.
    learner_mod.handle_resume_logic = _fake_orig_handle_resume_logic
    learner_mod.CHECKPOINTS_DIR = "checkpoints"
    learner_mod.LAST_CHECKPOINT_LINK = "last"
    learner_mod.PRETRAINED_MODEL_DIR = "pretrained_model"
    learner_mod._anvil_resume_weights_patched = False

    patches.patch_learner_resume_reloads_trained_weights()

    # --- fresh (non-resumed) run: pretrained_path must be left alone ---
    fresh_cfg = _make_cfg(resume=False, pretrained_path=STALE_BC_PATH)
    out = learner_mod.handle_resume_logic(fresh_cfg)
    if out.policy.pretrained_path != STALE_BC_PATH:
        print(
            f"[FAIL] non-resume case: pretrained_path should be untouched, "
            f"got {out.policy.pretrained_path!r}",
            flush=True,
        )
        return 1
    print("[OK] non-resume case: pretrained_path left untouched", flush=True)

    # --- resumed run: pretrained_path must point at the checkpoint's OWN
    # trained weights, not the stale BC baseline the bug would otherwise
    # silently keep reloading forever ---
    resumed_cfg = _make_cfg(resume=True, pretrained_path=STALE_BC_PATH)
    out = learner_mod.handle_resume_logic(resumed_cfg)
    expected = "/workspace/outputs/learner/checkpoints/last/pretrained_model"
    if out.policy.pretrained_path != expected:
        print(
            f"[FAIL] resume case: expected pretrained_path={expected!r}, "
            f"got {out.policy.pretrained_path!r} -- "
            f"still the stale BC path == the exact bug this patch fixes",
            flush=True,
        )
        return 1
    print(
        f"[OK] resume case: pretrained_path correctly repointed to {expected!r}, "
        f"not left at the stale {STALE_BC_PATH!r}",
        flush=True,
    )

    # --- a second resume (different output_dir) must re-derive its own
    # checkpoint path, not reuse the first call's -- guards against any
    # accidental closure-captured state ---
    resumed_cfg_2 = _make_cfg(resume=True, pretrained_path=STALE_BC_PATH, output_dir="/workspace/outputs/other_run")
    out2 = learner_mod.handle_resume_logic(resumed_cfg_2)
    expected_2 = "/workspace/outputs/other_run/checkpoints/last/pretrained_model"
    if out2.policy.pretrained_path != expected_2:
        print(
            f"[FAIL] second resume (different output_dir): expected {expected_2!r}, "
            f"got {out2.policy.pretrained_path!r}",
            flush=True,
        )
        return 1
    print("[OK] a second resume with a different output_dir re-derives its own checkpoint path", flush=True)

    print("\n[stage14] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
