#!/usr/bin/env bash
# Build the hilserl-test image with git provenance baked in.
#
# Plain `docker build` leaves a running container unable to answer "what
# code is this, actually" -- a real source of confusion this session (is
# this the same code as the last successful run, or did the image just go
# stale relative to the working tree). This wrapper computes the git
# commit + dirty status at build time and passes them as build-args; the
# Dockerfile writes them into /workspace/BUILD_INFO.json, and
# session_provenance.py prints them at actor/learner/gym_manipulator
# startup.
#
# Usage:
#   ./scripts/build_hilserl_image.sh [extra docker build args...]
set -euo pipefail
cd "$(dirname "$0")/.."

GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
if git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
    GIT_DIRTY=clean
else
    GIT_DIRTY=dirty
fi

echo "[build_hilserl_image] building from commit ${GIT_COMMIT} (${GIT_DIRTY})"

# LEROBOT_EXTRAS=hilserl pulls grpcio, gym-hil, and placo -- without it the
# image silently builds "successfully" but RobotKinematics/actor.py/
# learner.py all fail at runtime (ModuleNotFoundError, caught here directly
# via the regression suite 2026-10-01: an earlier version of this script
# omitted this and every kinematics/gRPC-touching test failed, not because
# of a code bug). hilserl-test is never used for the production
# imitation-learning image (that one's LEROBOT_EXTRAS comes from .env via
# docker-compose, e.g. smolvla/groot/pi) -- it always needs exactly this.
docker build \
    --build-arg GIT_COMMIT="${GIT_COMMIT}" \
    --build-arg GIT_DIRTY="${GIT_DIRTY}" \
    --build-arg LEROBOT_EXTRAS=hilserl \
    -t hilserl-test \
    -f docker/inference/Dockerfile \
    "$@" \
    .

echo "[build_hilserl_image] built hilserl-test from ${GIT_COMMIT} (${GIT_DIRTY})"
