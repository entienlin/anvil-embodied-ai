"""Prints a one-line startup fingerprint (git commit/dirty, image build
time, host, config arg, GPU/ROS env) for every real HIL-SERL process.

Motivation (2026-10-01): a real-hardware debugging session spent real time
on "did something change since the last successful run" -- answerable in
seconds if the answer is just sitting at the top of the log, instead of
requiring live archaeology (checking git log, docker image timestamps,
nvidia-smi, etc. after the fact). See patches.py's own changelog index for
the same motivation applied to code changes; this is the runtime/
environment half of the same problem.

Best-effort only: never raises, never blocks startup. A missing
BUILD_INFO.json (image built before this existed, or via a bare `docker
build` instead of scripts/build_hilserl_image.sh) degrades gracefully to
"unknown" fields rather than failing the real run over a diagnostics gap.
"""

import json
import os
import socket
import sys
from datetime import UTC, datetime

_BUILD_INFO_PATH = "/workspace/BUILD_INFO.json"


def _read_build_info() -> dict:
    try:
        with open(_BUILD_INFO_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _config_path_from_argv(argv: list[str]) -> str:
    for i, arg in enumerate(argv):
        if arg == "--config_path" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--config_path="):
            return arg.split("=", 1)[1]
    return "unknown"


def log_session_provenance(process_name: str) -> None:
    """Call once, as early as possible in each entrypoint's main() --
    before apply_all_patches(), since this should never be skipped even if
    a later patch fails to apply."""
    try:
        build_info = _read_build_info()
        config_path = _config_path_from_argv(sys.argv)

        line = (
            f"[SESSION] process={process_name} "
            f"git={build_info.get('git_commit', 'unknown')}"
            f"({build_info.get('git_dirty', 'unknown')}) "
            f"image_built={build_info.get('built_at', 'unknown')} "
            f"host={socket.gethostname()} pid={os.getpid()} "
            f"config={os.path.basename(config_path)} "
            f"cuda_devices={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')} "
            f"ros_domain={os.environ.get('ROS_DOMAIN_ID', 'unset')} "
            f"started_at={datetime.now(UTC).isoformat(timespec='seconds')}"
        )
        print(line, flush=True)

        try:
            manifest_path = f"/tmp/hilserl_session_{process_name}_{os.getpid()}.json"
            with open(manifest_path, "w") as f:
                json.dump(
                    {
                        "process": process_name,
                        "build_info": build_info,
                        "host": socket.gethostname(),
                        "pid": os.getpid(),
                        "config_path": config_path,
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
                        "started_at": datetime.now(UTC).isoformat(),
                        "argv": sys.argv,
                    },
                    f,
                    indent=2,
                )
        except OSError:
            pass  # /tmp manifest is a convenience, not load-bearing
    except Exception:  # noqa: BLE001 -- must never block a real run over diagnostics
        print(f"[SESSION] process={process_name} -- provenance logging failed, continuing", flush=True)
