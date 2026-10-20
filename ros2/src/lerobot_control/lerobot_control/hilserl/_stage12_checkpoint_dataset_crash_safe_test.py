"""patch_learner_checkpoint_saves_dataset_atomically(): the actual
regression test for the fix, not just the happy path -- if the slow
rewrite is interrupted partway through, the PREVIOUS good dataset_dir
must still be there and still loadable afterward. The old (unpatched)
code would have already rmtree'd the good copy before this point, so
this test would fail against the original lerobot behavior and pass only
with the patch applied. No ROS2/hardware needed.

Run: python3 -m lerobot_control.hilserl._stage12_checkpoint_dataset_crash_safe_test
"""

import os
import shutil
import sys
import tempfile
from unittest.mock import MagicMock

import torch

from lerobot_control.hilserl import patches


def _make_buffer():
    from lerobot.rl.buffer import ReplayBuffer

    buf = ReplayBuffer(
        capacity=50, device="cpu", state_keys=["observation.state"],
        storage_device="cpu", optimize_memory=True,
    )
    for step in range(3):
        buf.add(
            state={"observation.state": torch.full((1, 2), float(step))},
            action=torch.tensor([[float(step)]]),
            reward=1.0 if step == 2 else 0.0,
            next_state={"observation.state": torch.full((1, 2), float(step + 1))},
            done=step == 2,
            truncated=False,
        )
    return buf


def main() -> int:
    patches.patch_learner_checkpoint_saves_dataset_atomically()
    import lerobot.rl.learner as learner_mod
    from lerobot.rl.buffer import ReplayBuffer

    with tempfile.TemporaryDirectory() as d:
        output_dir = os.path.join(d, "run")
        os.makedirs(output_dir)
        dataset_dir = os.path.join(output_dir, "dataset")

        cfg = MagicMock()
        cfg.output_dir = output_dir
        cfg.env.task = "crash-safe-test"
        cfg.dataset.repo_id = "local/crash-safe-test"
        policy = MagicMock()
        learner_mod.get_step_checkpoint_dir = lambda *a, **kw: os.path.join(output_dir, "checkpoints", "000100")
        learner_mod.save_checkpoint = lambda **kw: os.makedirs(kw["checkpoint_dir"], exist_ok=True)
        learner_mod.update_last_checkpoint = lambda *a, **kw: None

        # --- first, a real successful save, establishing a good dataset_dir ---
        buf1 = _make_buffer()
        learner_mod.save_training_checkpoint(
            cfg=cfg, optimization_step=100, online_steps=1000, interaction_message=None,
            policy=policy, optimizers={}, replay_buffer=buf1, offline_replay_buffer=None,
            dataset_repo_id=None, fps=10,
        )
        if not os.path.isdir(dataset_dir):
            print("[FAIL] setup: first save did not create dataset_dir", flush=True)
            return 1
        good_episode_count_before = _episode_count(dataset_dir, "local/crash-safe-test")
        print(f"[setup] first save OK, {good_episode_count_before} episode(s)", flush=True)

        # --- second save, interrupted partway through the slow rewrite ---
        buf2 = _make_buffer()
        orig_to_lerobot_dataset = ReplayBuffer.to_lerobot_dataset

        def _boom(self, *a, **kw):
            raise RuntimeError("simulated crash mid-write (Ctrl+C / hardware fault / OOM)")

        ReplayBuffer.to_lerobot_dataset = _boom
        try:
            raised = False
            try:
                learner_mod.save_training_checkpoint(
                    cfg=cfg, optimization_step=200, online_steps=1000, interaction_message=None,
                    policy=policy, optimizers={}, replay_buffer=buf2, offline_replay_buffer=None,
                    dataset_repo_id=None, fps=10,
                )
            except RuntimeError:
                raised = True
            if not raised:
                print("[FAIL] expected the simulated crash to propagate", flush=True)
                return 1
        finally:
            ReplayBuffer.to_lerobot_dataset = orig_to_lerobot_dataset

        # --- the OLD good snapshot must still be there, intact and loadable ---
        if not os.path.isdir(dataset_dir):
            print("[FAIL] dataset_dir is GONE after the interrupted save -- "
                  "this is exactly the bug the patch exists to fix", flush=True)
            return 1
        if os.path.isdir(dataset_dir + ".tmp"):
            print("[FAIL] leftover .tmp directory from the interrupted write was not cleaned up "
                  "on the next attempt (not fatal on its own, but should not linger)", flush=True)
            # not fatal -- continue checking the real assertion below
        good_episode_count_after = _episode_count(dataset_dir, "local/crash-safe-test")
        if good_episode_count_after != good_episode_count_before:
            print(f"[FAIL] episode count changed after a crashed save "
                  f"({good_episode_count_before} -> {good_episode_count_after}) -- "
                  "dataset_dir is not the untouched old snapshot", flush=True)
            return 1
        print(f"[OK] after a crash mid-write, dataset_dir is still the OLD good snapshot "
              f"({good_episode_count_after} episode(s), unchanged)", flush=True)

    print("\n[stage12] PASSED", flush=True)
    return 0


def _episode_count(dataset_dir: str, repo_id: str) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id=repo_id, root=dataset_dir).meta.total_episodes


if __name__ == "__main__":
    sys.exit(main())
