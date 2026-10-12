"""Isolated test for patch_logging_includes_tracebacks() -- no robot/ROS2,
no gRPC needed. Reproduces the exact mechanism that swallowed the real
exception behind both 2026-10-08 "Exception iterating requests!" crashes:
logging.exception() (which always attaches exc_info) going through
lerobot's own init_logging()-installed formatter, which previously dropped
the traceback on the floor.

Loads patches.py directly by file path (not via the lerobot_control.hilserl
package __init__, which eagerly imports rclpy-dependent modules this test
doesn't need and the host venv doesn't have installed).

Usage: python3 _stage8_logging_traceback_patch_test.py
"""

import importlib.util
import logging
import sys
import tempfile
from pathlib import Path


def _load_patches_module():
    path = Path(__file__).resolve().parent / "patches.py"
    spec = importlib.util.spec_from_file_location("_anvil_patches_standalone", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    import lerobot.rl.actor as actor_mod
    import lerobot.rl.learner as learner_mod
    import lerobot.utils.utils as utils_mod

    with tempfile.TemporaryDirectory() as tmp:
        log_file = Path(tmp) / "unpatched.log"

        # --- Sanity check: UNPATCHED behaviour really does lose the traceback ---
        utils_mod.init_logging(log_file=log_file)
        try:
            raise ValueError("synthetic underlying error, e.g. a pickling failure")
        except ValueError:
            logging.getLogger("grpc._channel").exception("Exception iterating requests!")
        text = log_file.read_text()
        assert "Exception iterating requests!" in text, "sanity: message itself should log"
        assert "ValueError" not in text and "Traceback" not in text, (
            f"sanity check failed -- unpatched formatter already includes traceback?!\n{text}"
        )
        print("[OK] confirmed unpatched: traceback is lost, matching the real incident")

        # --- Apply the patch (mirrors apply_all_patches()'s own idempotency guard) ---
        patches_mod = _load_patches_module()
        patches_mod.patch_logging_includes_tracebacks()
        assert utils_mod.init_logging is not None
        assert actor_mod.init_logging is utils_mod.init_logging, (
            "actor.py's already-bound init_logging name was not repointed"
        )
        assert learner_mod.init_logging is utils_mod.init_logging, (
            "learner.py's already-bound init_logging name was not repointed"
        )
        print("[OK] actor.py / learner.py's bound init_logging names both repointed")

        # --- Re-init logging (as every forked actor/learner subprocess does) and retest ---
        log_file2 = Path(tmp) / "patched.log"
        actor_mod.init_logging(log_file=log_file2)
        try:
            raise ValueError("synthetic underlying error, e.g. a pickling failure")
        except ValueError:
            logging.getLogger("grpc._channel").exception("Exception iterating requests!")
        text2 = log_file2.read_text()
        assert "Exception iterating requests!" in text2
        assert "ValueError" in text2, f"traceback still missing after patch:\n{text2}"
        assert "Traceback (most recent call last)" in text2
        assert "synthetic underlying error" in text2
        print("[OK] patched: full traceback now present in the log")

        # --- Regular (non-exception) records must be completely unaffected ---
        logging.info("plain message, no exception")
        text3 = log_file2.read_text()
        assert text3.count("\n") < text3.count("Traceback") * 20 + 50  # loose: no runaway growth
        assert "plain message, no exception" in text3
        print("[OK] plain log records unaffected")

        # --- Idempotency: calling the patch again, and re-init'ing again, must not double-wrap ---
        patches_mod.patch_logging_includes_tracebacks()
        log_file3 = Path(tmp) / "patched_again.log"
        learner_mod.init_logging(log_file=log_file3)
        try:
            raise RuntimeError("second synthetic error")
        except RuntimeError:
            logging.getLogger("grpc._channel").exception("Exception iterating requests!")
        text4 = log_file3.read_text()
        occurrences = text4.count("Traceback (most recent call last)")
        assert occurrences == 1, f"expected exactly one traceback block, got {occurrences}:\n{text4}"
        print("[OK] idempotent -- no double-wrapped formatter, exactly one traceback block")

    print("\nAll checks passed.")


if __name__ == "__main__":
    sys.exit(main())
