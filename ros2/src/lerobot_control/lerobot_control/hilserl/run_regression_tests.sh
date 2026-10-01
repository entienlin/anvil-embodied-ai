#!/usr/bin/env bash
# Runs every offline HIL-SERL regression test (_stageN_*_test.py,
# _act_*_test.py) in this directory and prints a pass/fail summary.
#
# These accumulated one at a time across the real-hardware debugging
# marathon (see HILSERL_STATUS.md) with no shared runner -- re-verifying
# "did my change break anything already fixed" meant remembering which of
# ~28 scattered scripts to run by hand. This replaces that.
#
# Each test is a standalone script (not pytest -- they predate any
# decision to add a test framework, and converting them isn't in scope
# here), so this just shells out to each with `python3` and checks exit
# code. They need lerobot + rclpy + torch, i.e. they must run inside the
# built hilserl-test image, not on the bare host:
#
#   docker run --rm hilserl-test bash -c \
#     '. /opt/ros/jazzy/setup.sh && . /workspace/install/setup.sh && \
#      bash /workspace/src/lerobot_control/lerobot_control/hilserl/run_regression_tests.sh'
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mapfile -t test_files < <(find "$SCRIPT_DIR" -maxdepth 1 -name '_stage*_test.py' -o -name '_act*_test.py' | sort)

if [ ${#test_files[@]} -eq 0 ]; then
    echo "No regression test files found in $SCRIPT_DIR"
    exit 1
fi

echo "Running ${#test_files[@]} offline regression tests..."
echo ""

declare -a failed=()
for f in "${test_files[@]}"; do
    name=$(basename "$f")
    printf '%-55s ' "$name"
    if python3 "$f" > /tmp/"${name}".log 2>&1; then
        echo "PASS"
    else
        echo "FAIL"
        failed+=("$name")
    fi
done

echo ""
if [ ${#failed[@]} -eq 0 ]; then
    echo "All ${#test_files[@]} regression tests passed."
    exit 0
else
    echo "${#failed[@]} test(s) FAILED:"
    for f in "${failed[@]}"; do
        echo "  - $f  (log: /tmp/${f}.log)"
    done
    exit 1
fi
