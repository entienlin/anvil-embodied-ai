"""patch_learner_pauses_training_when_actor_idle(): confirms the real
2026-10-08 bug fix -- a freshly constructed _StalenessAwareReplayBuffer
must report len()==0 IMMEDIATELY (before any .add() call), not just after
STALE_AFTER_S seconds of idling. The old behavior (seeding the staleness
clock to "now" at construction) gave every learner restart/resume a free
window to keep training on stale, already-loaded data with zero actor
connected -- confirmed directly on real hardware: 3 restarts this morning
each produced ~100 extra optimization steps against 6+ hour old data,
degrading the policy with no fresh correction. Pure logic, no hardware.

Loads patches.py directly by file path (not via the lerobot_control.hilserl
package __init__, which eagerly imports rclpy-dependent modules this test
doesn't need and the host venv doesn't have installed).

Run: python3 _stage9_idle_pause_starts_stale_test.py
"""

import importlib.util
import sys
import time
from pathlib import Path

import lerobot.rl.learner as learner_mod


def _load_patches_module():
    path = Path(__file__).resolve().parent / "patches.py"
    spec = importlib.util.spec_from_file_location("_anvil_patches_standalone9", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patches = _load_patches_module()


class _FakeReal:
    def __init__(self, n):
        self._n = n

    def add(self, *a, **kw):
        self._n += 1

    def __len__(self):
        return self._n


def main() -> None:
    orig_initialize_replay_buffer = learner_mod.initialize_replay_buffer
    learner_mod.initialize_replay_buffer = lambda cfg, device, storage_device: _FakeReal(5300)

    patches.patch_learner_pauses_training_when_actor_idle()

    buf = learner_mod.initialize_replay_buffer(None, None, None)

    # The real bug: immediately after construction, with a real (large,
    # already-loaded-from-disk) buffer underneath and zero .add() calls
    # yet, len() must be 0 -- not the real buffer's size.
    assert len(buf) == 0, f"expected 0 immediately after construction (stale), got {len(buf)}"
    print("[OK] fresh construction reports 0 immediately, even with a large pre-loaded real buffer")

    # A real .add() (an actual actor transition arriving) must un-stale it.
    buf.add()
    assert len(buf) == 5301, f"expected 5301 right after a real .add(), got {len(buf)}"
    print("[OK] a real .add() call makes it non-stale again")

    # And it must go stale again after enough idle time -- simulate by
    # directly backdating the internal timestamp (same mechanism the real
    # 90s wait would exercise, without actually sleeping 90s in a test).
    buf._anvil_last_add = time.time() - 91.0
    assert len(buf) == 0, "expected 0 again after simulated 91s of no .add()"
    print("[OK] goes stale again after no .add() for > STALE_AFTER_S")

    learner_mod.initialize_replay_buffer = orig_initialize_replay_buffer
    print("\nAll checks passed.")


if __name__ == "__main__":
    sys.exit(main())
