#!/usr/bin/env bash
# Checks for leftover hilserl-test containers / GPU processes before
# starting a real HIL-SERL session.
#
# Built after finding a 2-day-old orphaned `hilserl_learner` process
# (started without --rm, never torn down) silently sitting on the same
# GPU a real session was about to use. Read-only -- reports what it finds,
# does not kill anything, since deciding what's safe to remove is a human
# call (see HILSERL_STATUS.md).
#
# Usage: ./scripts/check_hilserl_gpu_health.sh
set -uo pipefail

echo "=== hilserl-test containers currently running ==="
RUNNING=$(docker ps --filter "ancestor=hilserl-test" --format '{{.ID}}\t{{.Status}}\t{{.Names}}\t{{.Command}}')
if [ -z "$RUNNING" ]; then
    echo "(none)"
else
    echo "$RUNNING"
    echo ""
    echo "^ If any of these have been up for more than your current session's"
    echo "  expected duration, or you don't recognize the command, it's"
    echo "  likely a leftover diagnostic run. Inspect with:"
    echo "    docker top <container-id>"
    echo "  and stop it with:"
    echo "    docker kill <container-id> && docker rm <container-id>"
fi

echo ""
echo "=== hilserl-test containers stopped but not removed (no --rm) ==="
STOPPED=$(docker ps -a --filter "ancestor=hilserl-test" --filter "status=exited" --format '{{.ID}}\t{{.Status}}\t{{.Names}}')
if [ -z "$STOPPED" ]; then
    echo "(none)"
else
    echo "$STOPPED"
    echo ""
    echo "^ These finished but weren't run with --rm. Safe to remove with:"
    echo "    docker rm <container-id>"
fi

echo ""
echo "=== GPU memory/process snapshot ==="
if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
    echo ""
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
    echo ""
    echo "^ Cross-check any PID here against 'docker top' output above --"
    echo "  a PID holding real VRAM with 0% utilization and no corresponding"
    echo "  running container is almost certainly orphaned."
else
    echo "nvidia-smi not found -- skipping GPU check"
fi
