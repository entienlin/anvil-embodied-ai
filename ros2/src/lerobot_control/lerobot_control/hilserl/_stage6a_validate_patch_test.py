"""Isolated test for patch_train_pipeline_config_validate_for_optional_dataset(),
added after discovering TrainPipelineConfig.validate() crashes with
AttributeError the moment dataset: is omitted from a TrainRLServerPipelineConfig
-- even though that field is explicitly typed Optional specifically to
support pure-online RL (no offline dataset). Both actor.py and learner.py
call cfg.validate() unconditionally at startup, so this blocked the
simplest smoke-test path entirely.

No ROS2/hardware needed -- constructs a bare TrainRLServerPipelineConfig
directly (not via draccus parsing) with a real SAC policy config attached
(validate() requires self.policy is not None) and dataset left at its
default None.

Run: python3 -m lerobot_control.hilserl._stage6a_validate_patch_test
"""

import sys

from lerobot_control.hilserl.patches import patch_train_pipeline_config_validate_for_optional_dataset


def main() -> int:
    from lerobot.configs.train import TrainRLServerPipelineConfig
    from lerobot.policies.sac.configuration_sac import SACConfig

    print("[stage6a] UNPATCHED: confirming validate() really does crash "
          "on dataset=None (sanity-checks the diagnosis)...", flush=True)
    cfg = TrainRLServerPipelineConfig(policy=SACConfig(push_to_hub=False))
    if cfg.dataset is not None:
        print(f"[FAIL] expected dataset to default to None, got {cfg.dataset}", flush=True)
        return 1
    try:
        cfg.validate()
        print("[FAIL] expected an AttributeError before patching, got none", flush=True)
        return 1
    except AttributeError as e:
        print(f"[OK] unpatched validate() correctly raised: {e}", flush=True)

    print("[stage6a] applying patch...", flush=True)
    patch_train_pipeline_config_validate_for_optional_dataset()

    print("[stage6a] PATCHED: validate() must succeed with dataset=None, "
          "and leave it as None afterward...", flush=True)
    cfg2 = TrainRLServerPipelineConfig(policy=SACConfig(push_to_hub=False))
    cfg2.validate()
    if cfg2.dataset is not None:
        print(f"[FAIL] expected dataset to be restored to None after validate(), got {cfg2.dataset}", flush=True)
        return 1
    print("[OK] validate() succeeded, dataset correctly restored to None afterward "
          "(actor.py/learner.py's own `if cfg.dataset is not None` checks will see None)", flush=True)

    print("[stage6a] a REAL dataset config must still validate normally (unaffected by this patch)...", flush=True)
    from lerobot.configs.default import DatasetConfig

    cfg3 = TrainRLServerPipelineConfig(
        policy=SACConfig(push_to_hub=False), dataset=DatasetConfig(repo_id="fake/repo")
    )
    cfg3.validate()
    if cfg3.dataset is None or cfg3.dataset.repo_id != "fake/repo":
        print(f"[FAIL] real dataset config was altered: {cfg3.dataset}", flush=True)
        return 1
    print("[OK] real dataset config passes through validate() unchanged", flush=True)

    print("[stage6a] re-applying patch must be a no-op (idempotent)...", flush=True)
    before = TrainRLServerPipelineConfig.validate
    patch_train_pipeline_config_validate_for_optional_dataset()
    if TrainRLServerPipelineConfig.validate is not before:
        print("[FAIL] re-applying the patch replaced the method again", flush=True)
        return 1
    print("[OK] idempotent", flush=True)

    print("[stage6a] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
