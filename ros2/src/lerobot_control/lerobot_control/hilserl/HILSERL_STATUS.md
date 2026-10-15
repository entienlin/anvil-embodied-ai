# HIL-SERL push-bottle: what's actually current

This directory accumulated across a long real-hardware debugging marathon
(dozens of bugs found only by actual hardware runs -- see `patches.py`'s own
changelog index for the code-level history). This file exists so "is this
file still used, or is it debris from a past investigation" has a written
answer instead of needing to be re-derived from conversation history every
time. Keep it in sync when adding/retiring a file -- same spirit as
`patches.py`'s index.

## Active production path (bimanual, 2-arm / 4-camera, joint-space)

**As of 2026-10-15, the bimanual path below is the active production
path, not the single-arm one.** The single-arm config
(`train_rl_push_bottle_jointspace.yaml`) is historical -- it proved
joint-space over EE-delta and seeded every real-hardware fix the bimanual
composite inherited, but it is not what's actually being run now.

The current, working HIL-SERL loop for this task uses **joint-space**
actions (the follower's own next-tick `observation.state`), not the
original EE-delta/IK pipeline -- confirmed in 2026-09-30 that the EE-delta
action label was causally disconnected from the follower's real motion,
and reconfirmed directly in the bimanual config itself (`train_rl_push_bottle_bimanual_jointspace.yaml`):
`processor.inverse_kinematics: null` and both leader arms set
`action_mode: joint_space` -- `AnvilOpenArmLeader._get_action_joint_space()`
returns the leader's own real current joint positions directly, never an
FK-derived EE delta. There is no EE-delta flow anywhere in this branch's
data/training/inference path.

| File | Role |
|---|---|
| `robot_anvil_openarm.py` | `AnvilOpenArmFollower` -- the real robot, ROS2-native |
| `teleop_anvil_openarm_leader.py` | `AnvilOpenArmLeader` -- the real leader arm as a HIL-SERL `Teleoperator` |
| `robot_anvil_openarm_multi.py` | `AnvilOpenArmMultiFollower` -- N real `AnvilOpenArmFollower`s, 1 combined `Robot` (bimanual) |
| `teleop_anvil_openarm_multi_leader.py` | `AnvilOpenArmMultiLeader` -- N real `AnvilOpenArmLeader`s + ONE shared `KeyboardIntervention` (bimanual) |
| `switch_service_client.py` | State-aware `controller_manager/switch_controller` wrapper |
| `arms_resetter_client.py` | Real homing via the devbox's `arms_resetter_node` |
| `keyboard_intervention.py` | termios/tty-based (not pynput/X11) space-to-intervene toggle |
| `patches.py` | All runtime patches -- **see its own top-of-file index first** (updated 2026-10-15, all 25 `patch_*` functions registered in `apply_all_patches()` are documented there) |
| `session_provenance.py` | Startup `[SESSION]` fingerprint (git/build/host) -- see below |
| `run_actor.py` | `hilserl_actor` console script -- SAC actor, real-hardware control loop |
| `run_learner.py` | `hilserl_learner` console script -- SAC learner, gRPC, no real hardware |
| `run_gym_manipulator.py` | `hilserl_gym_manipulator` console script -- demo recording (no RL) |
| `bc_pretrain.py` | Offline BC pretraining from a recorded dataset |

Current config: `configs/hilserl/train_rl_push_bottle_bimanual_jointspace.yaml`,
launched via `scripts/run_hilserl_bimanual.sh` (learner/actor two-terminal
pattern, `--resume` to continue a checkpoint, `--actor-warmup-steps N` for
the opt-in critic-only warmup window -- see "Current investigation" below
for why that knob has NOT resolved the open problem).

**IMPORTANT -- main checkout vs. this worktree:** the repo root
(`~/openarm/anvil-embodied-ai`, not this worktree) holds a stale,
single-arm-era snapshot: its `configs/hilserl/` has no bimanual config at
all, its `patches.py` is missing 6 functions present here, and its BC
checkpoints are the old single-arm `bc_pretrained_v*` ones, not any of the
`bc_bimanual_*`/`bc_staging*` checkpoints actually in use. This worktree
(`.claude/worktrees/agent-a2985bd74533642e2`) is the sole source of truth
for all of the above files -- any tool or agent investigating this branch
must be pointed here explicitly, not at the repo root.

**Real-hardware validated, extensively**, not just offline: dozens of real
sessions since early October, Stage 0 through Stage 6 of the original
build-out plan all passed on real hardware (kinematics, fake-hardware
dry runs, real leader-arm intervention, real actor/learner gRPC training),
multiple full-task completions (`reward=1.0`) routinely across sessions,
learner checkpoint progressed past step 3200 as of 2026-10-15. The
`_bimanual1`-`_bimanual3` composite-class offline tests below were the
*pre*-hardware gate, not the final validation.

| File | Role |
|---|---|
| `_bimanual1_composite_follower_test.py` | Offline: composite follower, incl. single-arm behavioral equivalence |
| `_bimanual2_composite_leader_test.py` | Offline: composite leader, shared-keyboard + joint-space-only validation |
| `_bimanual3_leader_registry_test.py` | Offline: `_registered_leaders` keyed-by-prefix, no overwrite with 2 leaders |

Previously-open risks, now resolved or superseded:
- Liveness watchdog false-trips: fixed by widening `liveness_window_size`
  10->40 (both arms' config entries in the bimanual YAML) -- stable across
  many real sessions since.
- Multi-camera trajectory-save bug (only `wrist_r` was ever saved to
  per-episode `.npz` diagnostics): fixed on both the writer
  (`save_actor_trajectory()`) and reader (`bc_pretrain.py`) sides.
- Devbox camera `replay_buffer` OOM under 4-camera 1080p60 load: has not
  recurred across weeks of real bimanual sessions since this note was
  first written; treat as resolved unless it reappears, not as an active
  blocker.

## Current investigation (2026-10-15): episode-2-only success, static-vs-complete bimodal outcome

Open, unresolved. Across every real session since BC v3/the current
checkpoint lineage, `episode 2` of each session (and only episode 2) has
completed the full task with **zero human intervention**, while every
other episode in the same session needs correction -- reproduced in 4
consecutive sessions. The symptom itself, per direct operator report: the
policy either keeps moving toward the goal to completion, or reaches a
point after the downward/reach motion and goes **static** (commands stop
changing) until the human takes over -- a sharp bimodal outcome, not a
graded one.

Ruled out directly, with real evidence, not assumption:
- **Actor training/drift**: varied `--actor-warmup-steps` across 0
  (unconfirmed off), 200, and 800 (covering the entire session's step
  budget, i.e. actor+shared-encoder essentially frozen) -- the episode-2
  pattern was identical every time. If frozen weights still produce this
  split, it isn't caused by what the actor is learning session-to-session.
- **Object-reset confound**: operator confirmed the bottle is manually
  reset to the same position before/after every episode, not just at
  session start.
- **Operator-behavior confound**: operator confirmed intervention is
  purely reactive (only steps in once the arm visibly goes static), not a
  deliberate "let episode 2 run unaided" choice.

Leading hypothesis, not yet confirmed: ties back to a much earlier offline
finding (2026-09-24, same codebase) that the critic was measured to be
nearly action-blind -- perturbing the commanded action changed the
critic's Q-estimate by only ~0.2-0.6% of its across-state variance. A
critic that flat near the "keep pushing vs. stop" decision boundary would
make a frozen, deterministic policy's real-world outcome highly sensitive
to small, unavoidable physical variance (sub-degree reset precision,
camera-frame timing at episode start) -- enough to tip the same policy
into either basin on nominally-identical restarts. This has NOT been
re-verified on the current checkpoint (the 2026-09-24 measurement is from
an earlier point in this lineage) -- that re-verification, plus
accumulating substantially more real online training (not more warmup,
which has been ruled out as the lever) to sharpen the critic's landscape
near that boundary, is the next concrete step, not yet executed.

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
