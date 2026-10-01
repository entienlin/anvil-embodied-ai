# HIL-SERL push-bottle: what's actually current

This directory accumulated across a long real-hardware debugging marathon
(dozens of bugs found only by actual hardware runs -- see `patches.py`'s own
changelog index for the code-level history). This file exists so "is this
file still used, or is it debris from a past investigation" has a written
answer instead of needing to be re-derived from conversation history every
time. Keep it in sync when adding/retiring a file -- same spirit as
`patches.py`'s index.

## Active production path (single-arm, joint-space)

The current, working HIL-SERL loop for this task uses **joint-space**
actions (the follower's own next-tick `observation.state`), not the
original EE-delta/IK pipeline -- confirmed in 2026-09-30 that the EE-delta
action label was causally disconnected from the follower's real motion.

| File | Role |
|---|---|
| `robot_anvil_openarm.py` | `AnvilOpenArmFollower` -- the real robot, ROS2-native |
| `teleop_anvil_openarm_leader.py` | `AnvilOpenArmLeader` -- the real leader arm as a HIL-SERL `Teleoperator` |
| `switch_service_client.py` | State-aware `controller_manager/switch_controller` wrapper |
| `arms_resetter_client.py` | Real homing via the devbox's `arms_resetter_node` |
| `keyboard_intervention.py` | termios/tty-based (not pynput/X11) space-to-intervene toggle |
| `patches.py` | All runtime patches -- **see its own top-of-file index first** |
| `session_provenance.py` | Startup `[SESSION]` fingerprint (git/build/host) -- see below |
| `run_actor.py` | `hilserl_actor` console script -- SAC actor, real-hardware control loop |
| `run_learner.py` | `hilserl_learner` console script -- SAC learner, gRPC, no real hardware |
| `run_gym_manipulator.py` | `hilserl_gym_manipulator` console script -- demo recording (no RL) |
| `bc_pretrain.py` | Offline BC pretraining from a recorded dataset |

Current config: `configs/hilserl/train_rl_push_bottle_jointspace.yaml` (see
`configs/hilserl/README.md` for the full config inventory -- there are
several superseded/historical configs sitting alongside it).

## Diagnostic tools (not part of the regular loop, kept for reference)

These each answered a specific past question and are generic enough to be
useful again, but **none of them are run routinely**:

| File | Built for | Status |
|---|---|---|
| `run_joint_replay.py` | 2026-09-30: isolate pipeline vs. data bugs via raw joint-position replay | Answered: confirmed EE-delta was the problem, not the pipeline |
| `run_joint_policy_eval.py` | 2026-09-30: real-hardware eval of the joint-space ACT policy | Superseded by `hilserl_actor` once joint-space SAC worked end-to-end |
| `run_action_replay.py` | 2026-09-29: isolate ACT's bad motion from SAC/HIL-SERL-specific causes | Answered: same symptom without any RL code involved |
| `run_policy_eval.py` | 2026-09-29: real-hardware eval for a plain (non-RL) policy, e.g. ACT | Still valid if evaluating a non-SAC policy again |
| `eval_checkpoint.py` | Compare SAC checkpoints against one fixed reference dataset | **Still actively useful** -- use this, not an ad-hoc script, for any future checkpoint comparison (fixes the "different checkpoints measured against different references" mistake made earlier) |
| `verify_arms_resetter.py` | 2026-09-24: one-shot check that `/arms_resetter/reset` works at all | One-off bring-up verification; re-run only if the resetter integration itself is suspect |
| `live_ee_position.py` | 2026-09-24: measure real `end_effector_bounds` against the rig | Re-run only if the workspace/task geometry changes |

## Regression tests (`_stage*_test.py`, `_act*_test.py`)

~28 standalone offline scripts (not pytest -- predate any decision to add a
test framework), each verifying one specific fix without needing real
hardware. Run all of them with:

```bash
docker run --rm hilserl-test bash -c \
  '. /opt/ros/jazzy/setup.sh && . /workspace/install/setup.sh && \
   bash /workspace/src/lerobot_control/lerobot_control/hilserl/run_regression_tests.sh'
```

Run this after any change to `patches.py`, `robot_anvil_openarm.py`, or
`teleop_anvil_openarm_leader.py` -- it's the fast, no-hardware-needed check
that a fix for one bug didn't silently reopen another.

## Preventing GPU/container debris (the "dangling learner on GPU1" class of bug)

2026-10-01: found a `hilserl_learner` process from an old diagnostic
session (`train_rl_push_bottle.yaml` / `dataset_diag2`) still running on
GPU1 **two days later** -- the container was started without `--rm` and
never torn down. Before starting a real session:

```bash
./scripts/check_hilserl_gpu_health.sh
```

This lists any `hilserl-test`-based container that's still running and any
GPU process that isn't accounted for, so a leftover from a past diagnostic
run gets caught before it silently shares GPU time (or VRAM) with the next
real session. Always prefer `docker run --rm` for one-off diagnostic runs
specifically so this can't recur -- the production `hilserl_actor`/
`hilserl_learner` commands already use `--rm`.

## Building

Use `./scripts/build_hilserl_image.sh` instead of a bare `docker build` --
it bakes the git commit/dirty flag into the image so `session_provenance.py`
can report it at startup.
