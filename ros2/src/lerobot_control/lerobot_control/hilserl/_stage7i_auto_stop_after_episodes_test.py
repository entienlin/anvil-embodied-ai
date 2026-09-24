"""HILSERL_MAX_EPISODES: the actor SIGINTs itself right after the Nth episode is
saved/queued; the learner SIGINTs itself after the Nth episode summary, but only
in BC-only mode (a training learner must not be cut short -- it has checkpoints
to write). Unset/0 = never stops. Pure logic, no ROS2/hardware.

Run: python3 -m lerobot_control.hilserl._stage7i_auto_stop_after_episodes_test
"""

import os
import signal
import sys
import tempfile
import time

import lerobot.rl.actor as actor_mod
import lerobot.rl.learner as learner_mod
from lerobot.transport.utils import python_object_to_bytes

from lerobot_control.hilserl import patches

got = []
signal.signal(signal.SIGINT, lambda *_: got.append(1))

# stubs BEFORE applying patches so the wrappers sit on top of them
sent = []
actor_mod.push_transitions_to_transport_queue = lambda transitions, q: sent.append(len(transitions))
learner_mod.process_interaction_message = lambda message, shift, wandb=None: {"Interaction step": 1}
patches.patch_learner_progress_visibility()
patches.patch_learner_episode_stats_logging()


def check(cond, msg):
    if not cond:
        print(f"[FAIL] {msg}", flush=True)
        sys.exit(1)
    print(f"[OK] {msg}", flush=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        os.environ["HILSERL_TRAJ_DIR"] = d

        # --- actor ---
        os.environ.pop("HILSERL_MAX_EPISODES", None)
        for _ in range(3):
            actor_mod.push_transitions_to_transport_queue([{"bad": 1}], None)
        check(not got, "unset: actor never stops itself")

        os.environ["HILSERL_MAX_EPISODES"] = "5"
        patches._auto_stop_state["episodes"] = 0
        for i in range(4):
            actor_mod.push_transitions_to_transport_queue([{"bad": 1}], None)
        check(not got, "actor keeps running before the Nth episode")
        actor_mod.push_transitions_to_transport_queue([{"bad": 1}], None)
        check(len(got) == 1 and len(sent) == 3 + 5, "actor SIGINTs after the 5th episode; every episode still queued to the learner")
        actor_mod.push_transitions_to_transport_queue([{"bad": 1}], None)
        check(len(got) == 1, "no repeated SIGINT after the Nth episode")

        # --- learner ---
        got.clear()
        os.environ["HILSERL_MAX_EPISODES"] = "2"
        patches._episode_stats_state["path"] = None
        patches._episode_stats_state["bc_only"] = False
        for _ in range(3):
            learner_mod.process_interaction_message(python_object_to_bytes({}), 0, None)
        time.sleep(3.5)
        check(not got, "training learner is never stopped by the episode count")

        patches._episode_stats_state["bc_only"] = True
        patches._auto_stop_state["learner_episodes"] = 0
        learner_mod.process_interaction_message(python_object_to_bytes({}), 0, None)
        time.sleep(3.5)
        check(not got, "BC-only learner keeps running before the Nth episode")
        learner_mod.process_interaction_message(python_object_to_bytes({}), 0, None)
        check(not got, "BC-only learner waits a moment (last transitions in flight) before stopping")
        time.sleep(3.5)
        check(len(got) == 1, "BC-only learner SIGINTs itself after the 2nd episode summary")
    print("[stage7i] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
