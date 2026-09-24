"""patch_learner_episode_stats_logging(): every interaction message the
learner receives (one per episode: Episodic reward, Intervention rate, ...)
must be appended to <output_dir>/episode_stats.jsonl, the original function's
return value must be unchanged, and a logging failure must never break
training. Pure logic, no ROS2/hardware.

Run: python3 -m lerobot_control.hilserl._stage7c_episode_stats_logging_test
"""

import json
import os
import sys
import tempfile

import lerobot.rl.learner as learner_mod
from lerobot.transport.utils import python_object_to_bytes

from lerobot_control.hilserl import patches
from lerobot_control.hilserl.patches import patch_learner_episode_stats_logging


def main() -> int:
    patch_learner_episode_stats_logging()
    patch_learner_episode_stats_logging()  # idempotent

    msg = {
        "Episodic reward": 1.0,
        "Interaction step": 150,
        "Episode intervention": 1,
        "Intervention rate": 0.4,
        "Policy frequency [Hz]": 9.5,
    }

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "run_output", "episode_stats.jsonl")  # dir doesn't exist yet
        patches._episode_stats_state["path"] = path

        result = learner_mod.process_interaction_message(python_object_to_bytes(msg), 1000, None)
        if result["Interaction step"] != 1150:
            print(f"[FAIL] return value changed: {result}", flush=True)
            return 1
        print("[OK] original behaviour preserved (interaction_step_shift applied)", flush=True)

        learner_mod.process_interaction_message(python_object_to_bytes({**msg, "Interaction step": 300}), 1000, None)
        with open(path) as f:
            rows = [json.loads(line) for line in f]
        if len(rows) != 2 or rows[0]["Intervention rate"] != 0.4 or rows[1]["Interaction step"] != 1300:
            print(f"[FAIL] unexpected file contents: {rows}", flush=True)
            return 1
        print("[OK] one JSON line per episode, dir auto-created, values intact", flush=True)

        # A failing write must not raise into the training loop.
        patches._episode_stats_state["path"] = "/proc/definitely/not/writable/x.jsonl"
        try:
            out = learner_mod.process_interaction_message(python_object_to_bytes(msg), 0, None)
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] logging failure leaked into the caller: {e}", flush=True)
            return 1
        if out["Interaction step"] != 150:
            print("[FAIL] return value wrong when logging failed", flush=True)
            return 1
        print("[OK] a logging failure is swallowed, training path unaffected", flush=True)

    print("[stage7c] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
