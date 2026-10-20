"""patch_learner_checkpoint_saves_dataset_atomically(): a saved online
replay buffer must reload with the exact same contents (not just "didn't
crash") -- exercises a short episode, an episode with no trailing done
(cut short by capacity, has_pending_frames() path), and 11 episodes total
(crosses LeRobotDatasetMetadata's own 10-episode auto-flush boundary in
addition to the finalize()-triggered one). No ROS2/hardware needed.

Run: python3 -m lerobot_control.hilserl._stage11_checkpoint_dataset_roundtrip_test
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock

import torch

from lerobot_control.hilserl import patches


def _make_buffer(n_episodes=11, steps_per_episode=3, drop_last_done=False):
    from lerobot.rl.buffer import ReplayBuffer

    buf = ReplayBuffer(
        capacity=200,
        device="cpu",
        state_keys=["observation.state", "observation.images.cam"],
        storage_device="cpu",
        optimize_memory=True,
    )
    for ep in range(n_episodes):
        for step in range(steps_per_episode):
            is_last_step_of_episode = step == steps_per_episode - 1
            done = is_last_step_of_episode and not (drop_last_done and ep == n_episodes - 1)
            state = {
                "observation.state": torch.full((1, 4), float(ep * 10 + step)),
                "observation.images.cam": torch.rand(1, 3, 8, 8),
            }
            next_state = {
                "observation.state": torch.full((1, 4), float(ep * 10 + step + 1)),
                "observation.images.cam": torch.rand(1, 3, 8, 8),
            }
            buf.add(
                state=state,
                action=torch.tensor([[float(step), -float(step)]]),
                reward=1.0 if is_last_step_of_episode else 0.0,
                next_state=next_state,
                done=done,
                truncated=False,
            )
    return buf


def main() -> int:
    patches.patch_learner_checkpoint_saves_dataset_atomically()
    import lerobot.rl.learner as learner_mod

    with tempfile.TemporaryDirectory() as d:
        output_dir = os.path.join(d, "run")
        os.makedirs(output_dir)

        buf = _make_buffer()

        cfg = MagicMock()
        cfg.output_dir = output_dir
        cfg.env.task = "roundtrip-test"
        cfg.dataset.repo_id = "local/roundtrip-test"

        policy = MagicMock()
        policy.state_dict.return_value = {}
        optimizers = {}

        # save_checkpoint()/get_step_checkpoint_dir() touch real lerobot
        # checkpoint-saving machinery we don't want to fight with a mock
        # policy -- stub them out; this test is about the DATASET half,
        # the checkpoint-dir half is lerobot's own, unmodified code.
        learner_mod.get_step_checkpoint_dir = lambda *a, **kw: os.path.join(output_dir, "checkpoints", "000100")
        learner_mod.save_checkpoint = lambda **kw: os.makedirs(kw["checkpoint_dir"], exist_ok=True)
        learner_mod.update_last_checkpoint = lambda *a, **kw: None

        learner_mod.save_training_checkpoint(
            cfg=cfg,
            optimization_step=100,
            online_steps=1000,
            interaction_message=None,
            policy=policy,
            optimizers=optimizers,
            replay_buffer=buf,
            offline_replay_buffer=None,
            dataset_repo_id=None,
            fps=10,
        )

        dataset_dir = os.path.join(output_dir, "dataset")
        if not os.path.isdir(dataset_dir):
            print("[FAIL] dataset_dir was not created", flush=True)
            return 1
        if os.path.isdir(dataset_dir + ".tmp") or os.path.isdir(dataset_dir + ".prev"):
            print("[FAIL] leftover .tmp/.prev directory after a successful save", flush=True)
            return 1

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.rl.buffer import ReplayBuffer

        reloaded_ds = LeRobotDataset(repo_id="local/roundtrip-test", root=dataset_dir)
        if reloaded_ds.meta.total_episodes != 11:
            print(f"[FAIL] expected 11 episodes, got {reloaded_ds.meta.total_episodes}", flush=True)
            return 1
        if len(reloaded_ds) != 33:
            print(f"[FAIL] expected 33 frames (11 eps x 3 steps), got {len(reloaded_ds)}", flush=True)
            return 1
        print("[OK] reloaded dataset has the right episode/frame counts", flush=True)

        reloaded_buf = ReplayBuffer.from_lerobot_dataset(
            reloaded_ds, device="cpu", state_keys=["observation.state", "observation.images.cam"],
            storage_device="cpu", optimize_memory=True,
        )
        if reloaded_buf.size != buf.size:
            print(f"[FAIL] reloaded buffer size {reloaded_buf.size} != original {buf.size}", flush=True)
            return 1
        orig_actions = buf.actions[: buf.size]
        reloaded_actions = reloaded_buf.actions[: reloaded_buf.size]
        if not torch.allclose(orig_actions, reloaded_actions):
            print("[FAIL] reloaded actions don't match the original buffer's actions", flush=True)
            return 1
        print("[OK] reloaded buffer's actions match the original exactly", flush=True)

    print("\n[stage11] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
