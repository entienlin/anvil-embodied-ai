#!/usr/bin/env bash
# Launch the bimanual HIL-SERL learner or actor with consistent, reusable
# settings -- replaces hand-writing a new docker run command every session.
#
# Why this exists: two real mistakes already cost data earlier in this
# project -- forgetting to mount the learner's output_dir (lost a trained
# checkpoint entirely when the --rm container exited) and reusing a stale
# output_dir across runs (TrainPipelineConfig.validate() refuses to reuse
# one that already exists -> FileExistsError). This script gives every
# run its own never-reused subdirectory under outputs/, always mounted,
# and mounts the PARENT dir rather than the exact leaf path (same fix
# already used for dataset.root -- a Docker bind-mount target always
# presents as an already-existing directory the instant the container
# starts, which trips validate()'s exist_ok=False check if mounted at the
# exact leaf).
#
# Usage:
#   ./scripts/run_hilserl_bimanual.sh learner [--online-steps N] [--run-name NAME] [--resume] [--save-freq N] [--actor-warmup-steps N] [--critic-action-embed-dim N] [--actor-bc-anchor-weight W]
#   ./scripts/run_hilserl_bimanual.sh actor   [--online-steps N] [--run-name NAME] [--resume] [--critic-action-embed-dim N]
#   ./scripts/run_hilserl_bimanual.sh stop
#   ./scripts/run_hilserl_bimanual.sh prune   [--run-name NAME] [--keep N]
#
# --online-steps N is always RELATIVE -- "train N more steps this
# session," regardless of how far along a resumed run already is. The
# learner resolves the real current step from the checkpoint it's
# resuming from (if any) and adds N to it itself, so the caller never has
# to look up or compute an absolute target by hand (confirmed needed
# directly, 2026-10-08: passing the same small N on every resume silently
# disabled the auto-stop safety net once the real step count ran past it
# -- see the comment at this script's learner branch for the full
# mechanism).
#
# --actor-warmup-steps N (learner only, default off): hold the actor
# optimizer's first N steps as critic-only updates after a fresh start or
# resume, via patches.py's patch_sac_actor_warmup() (HILSERL_ACTOR_WARMUP_
# STEPS env var). Confirmed directly, 2026-10-08: with this off (every
# bimanual session so far), per-episode intervention rate rose 0.48 ->
# 0.77 within a single 4-episode session -- the actor drifting on a
# critic that's had almost no real calibration yet, matching this
# project's own earlier single-arm diagnosis of the same symptom. Off by
# default since this exact lever has never been validated before; try
# --actor-warmup-steps 200 as a starting point and adjust from there.
#
# --actor-bc-anchor-weight W (learner only, default off): adds
# W * MSE(actor_output, executed_action) to the actor loss, via patches.py's
# patch_sac_actor_bc_anchor() (HILSERL_ACTOR_BC_ANCHOR_WEIGHT env var).
# Confirmed directly from run 20261020's 4 real episodes, 2026-10-21: the
# actor's own deterministic output collapsed 10-20x in per-dim std between
# episode 2 and 3 across every action dim -- not tanh saturation (no dim's
# normalized mean exceeds |0.95|), not the robot getting physically stuck
# (step-to-step state motion stayed 6-8 deg/step throughout). It learned
# coarse task-progress tracking (critic's across-state Q-variance grew 16x
# over training) but lost fine, local state-reactivity, since the sparse/
# terminal reward never gives the critic a local action-sensitivity signal
# to pass on, and the frozen-near-zero temperature (patch_sac_freeze_
# temperature(), needed for a different problem) removes the entropy term
# that would otherwise resist this kind of collapse. This anchor term
# directly penalizes diverging from real executed (demo/intervention)
# actions, independent of whether the critic ever becomes better
# calibrated. Off by default since this lever has never been validated on
# real hardware before; try --actor-bc-anchor-weight 1.0 as a starting
# point and adjust from there.
#
# Start the learner first in one terminal; once it reports serving gRPC,
# start the actor in a SECOND terminal (needs a real TTY for keyboard
# intervention -- never run it in the background) with the same
# --run-name so both processes' logs/checkpoints land side by side. If
# --run-name is omitted: learner creates a new timestamped one (fresh run)
# or reuses the most recent one (--resume); actor always auto-reuses the
# most recently created one.
#
# --resume: continue training from the last saved checkpoint in that
# run's learner dir, instead of starting fresh from the BC checkpoint.
# This genuinely restores everything -- policy/critic/optimizer state,
# step counters, AND the online+offline replay buffers (confirmed by
# reading lerobot/rl/learner.py's handle_resume_logic()/
# initialize_replay_buffer() directly) -- not just the policy weights.
# Pass it to BOTH learner and actor when continuing an existing run:
# the learner actually resumes training state; the actor only needs it
# to bypass TrainPipelineConfig.validate()'s "directory already exists"
# guard (the actor has no persistent training state of its own to
# restore -- it just reconnects and receives fresh policy params from
# the learner as usual).
#
# stop/prune: the learner's own training loop never pauses on its own
# when the actor disconnects (confirmed by reading learner.py directly --
# its main loop is a bare `while True` gated only on a shutdown signal,
# never on cfg.policy.online_steps or on whether fresh actor data is
# arriving). It will happily keep training on stale replay-buffer data
# indefinitely if left running alone -- this is how one session reached
# 39,000 steps overnight unintentionally. ALWAYS start/stop the learner
# and actor together; use `stop` as the one command to do that reliably,
# and `prune` afterward (or periodically during a long session) to keep
# checkpoint disk usage bounded -- each checkpoint is a genuinely full,
# independent copy of policy+critic+3 optimizers' state (~87MB measured
# on this task), so saving every save_freq steps without ever deleting
# old ones grows unbounded (one overnight run alone reached 14GB).
set -euo pipefail

MODE="${1:-}"
shift || true
if [[ "$MODE" != "learner" && "$MODE" != "actor" && "$MODE" != "stop" && "$MODE" != "prune" ]]; then
    echo "Usage: $0 {learner|actor|stop|prune} [--online-steps N] [--run-name NAME] [--resume] [--save-freq N] [--keep N] [--actor-warmup-steps N] [--critic-action-embed-dim N] [--actor-bc-anchor-weight W]" >&2
    exit 1
fi

ONLINE_STEPS=2000
SAVE_FREQ=100
RUN_NAME=""
RESUME=false
KEEP=1
ACTOR_WARMUP_STEPS=0
ACTOR_BC_ANCHOR_WEIGHT=0.0
# Default 0 (off, byte-for-byte original critic architecture) --
# --critic-action-embed-dim N (both learner AND actor, unlike
# --actor-warmup-steps which is learner-only) applies
# patch_sac_critic_action_embedding(): gives the action its own learned
# embedding before concatenation with the encoder output, instead of raw
# concatenation. MUST be passed identically to both processes -- both
# construct a full SACPolicy (the actor for inference too), and a
# mismatched value between them means the actor would try to load a
# checkpoint shaped for one architecture into the other with no
# shape-tolerant loader installed (that loader is only wired up when this
# value is >0), i.e. a hard crash rather than a silent mismatch.
CRITIC_ACTION_EMBED_DIM=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --online-steps) ONLINE_STEPS="$2"; shift 2 ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --run-name) RUN_NAME="$2"; shift 2 ;;
        --resume) RESUME=true; shift ;;
        --keep) KEEP="$2"; shift 2 ;;
        --actor-warmup-steps) ACTOR_WARMUP_STEPS="$2"; shift 2 ;;
        --actor-bc-anchor-weight) ACTOR_BC_ANCHOR_WEIGHT="$2"; shift 2 ;;
        --critic-action-embed-dim) CRITIC_ACTION_EMBED_DIM="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ "$MODE" == "stop" ]]; then
    # Stops every running hilserl-learner-*/hilserl-actor-* container --
    # in practice there's at most one of each active at a time, so a
    # blanket stop is the simplest reliable way to enforce "learner and
    # actor run and stop together" without tracking individual container
    # IDs ourselves.
    #
    # Filters by NAME PREFIX, not `--filter ancestor=hilserl-test:latest`
    # (confirmed broken directly, 2026-10-07): the `hilserl-test:latest`
    # TAG gets reassigned to a new image ID on every rebuild, so an
    # ancestor filter stops matching a container that's still genuinely
    # running from the PREVIOUS image the moment a rebuild happens --
    # `stop` reported "nothing running" while a stuck learner from an
    # older image was still up. Container names are stable regardless of
    # what the image tag currently points to.
    IDS="$( { docker ps --filter "name=^/hilserl-learner-" -q; docker ps --filter "name=^/hilserl-actor-" -q; } )"
    if [[ -z "$IDS" ]]; then
        echo "[run_hilserl_bimanual] nothing running."
        exit 0
    fi
    echo "[run_hilserl_bimanual] stopping: $IDS"
    docker stop $IDS
    exit 0
fi

# Real recorded data (datasets/outputs) lives in the main checkout, not
# whichever worktree this script happens to run from -- git worktrees
# share tracked code, not untracked/gitignored data dirs. Resolve the main
# checkout via the shared .git dir so this works from any worktree without
# a hardcoded path.
MAIN_REPO="$(cd "$(dirname "$(git rev-parse --git-common-dir)")" && pwd)"
# A separate top-level dir, not nested under outputs/hilserl_push_bottle_bimanual_jointspace/ --
# that directory is already root-owned from earlier container writes (a
# known pattern in this project), so the host user can't mkdir inside it
# at all. Each run's own learner/actor subdirs will still end up
# root-owned too (created by the container's own root process) -- that's
# expected and fine; only the host-side RUNS_ROOT/RUN_DIR creation below
# needs to avoid the pre-existing root-owned tree.
RUNS_ROOT="$MAIN_REPO/outputs/hilserl_bimanual_runs"
DATASET_HOST="$MAIN_REPO/data/datasets/grasp_place_bimanual_demos"

if [[ ! -d "$DATASET_HOST" ]]; then
    echo "Dataset not found at $DATASET_HOST -- check MAIN_REPO resolved correctly." >&2
    exit 1
fi

mkdir -p "$RUNS_ROOT"

# Fresh learner runs get a new timestamped dir; actor always joins an
# existing one; a resuming learner also joins an existing one (there's
# nothing to resume FROM in a brand-new empty dir).
if [[ "$MODE" == "learner" && "$RESUME" == "false" ]]; then
    RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
else
    if [[ -z "$RUN_NAME" ]]; then
        RUN_NAME="$(ls -t "$RUNS_ROOT" 2>/dev/null | head -1)"
        if [[ -z "$RUN_NAME" ]]; then
            echo "No existing run under $RUNS_ROOT -- start the learner first, or pass --run-name explicitly." >&2
            exit 1
        fi
        echo "[run_hilserl_bimanual] no --run-name given, reusing most recent: $RUN_NAME" >&2
    fi
fi

RUN_DIR="$RUNS_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"

if [[ "$MODE" == "prune" ]]; then
    CKPT_DIR="$RUN_DIR/learner/checkpoints"
    if [[ ! -d "$CKPT_DIR" ]]; then
        echo "No checkpoints dir at $CKPT_DIR -- nothing to prune." >&2
        exit 0
    fi
    # Numeric-sort the step directories (exclude the 'last' symlink itself),
    # keep the highest KEEP of them, delete the rest. 'last' always points
    # at the newest one, which is always inside the kept set, so it's never
    # left dangling.
    TO_DELETE="$(cd "$CKPT_DIR" && ls -d [0-9]* 2>/dev/null | sort -n | head -n -"$KEEP")"
    if [[ -z "$TO_DELETE" ]]; then
        echo "[run_hilserl_bimanual] nothing to prune (already <= $KEEP checkpoints)."
        exit 0
    fi
    echo "[run_hilserl_bimanual] pruning $(echo "$TO_DELETE" | wc -l) checkpoint(s), keeping the newest $KEEP:"
    echo "$TO_DELETE"
    DELETE_ARGS="$(echo "$TO_DELETE" | sed 's#^#/data/learner/checkpoints/#')"
    docker run --rm -v "$RUN_DIR:/data" alpine sh -c "rm -rf $DELETE_ARGS"
    du -sh "$RUN_DIR/learner/checkpoints"
    exit 0
fi

if [[ "$MODE" == "learner" && "$RESUME" == "true" ]]; then
    LAST_CKPT="$RUN_DIR/learner/checkpoints/last"
    if [[ ! -e "$LAST_CKPT" ]]; then
        echo "--resume given but no checkpoint found at $LAST_CKPT -- can't resume a run that never saved one." >&2
        exit 1
    fi
fi

# Provenance: record exactly what was launched, next to the run's own
# data -- the first thing worth checking if a run behaves unexpectedly
# later (mirrors session_provenance.py's in-container fingerprint, but
# host-side and command-level rather than git/image-level).
{
    echo "mode=$MODE"
    echo "run_name=$RUN_NAME"
    echo "online_steps=$ONLINE_STEPS"
    echo "save_freq=$SAVE_FREQ"
    echo "resume=$RESUME"
    echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "worktree_commit=$(git rev-parse HEAD) ($(git diff --quiet && git diff --cached --quiet && echo clean || echo dirty))"
} >> "$RUN_DIR/${MODE}_provenance.txt"

# Matches scripts/build_hilserl_image.sh's actual output tag -- rebuilding
# that script's image updates this run automatically, no alias to keep in
# sync by hand.
IMAGE="hilserl-test:latest"
CONFIG_CONTAINER_PATH="/workspace/configs/hilserl/train_rl_push_bottle_bimanual_jointspace.yaml"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Image '$IMAGE' not found -- build it first: ./scripts/build_hilserl_image.sh" >&2
    exit 1
fi

COMMON_ARGS=(
    --rm --network host --gpus device=0
    -e ROS_DOMAIN_ID=1
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    -e CYCLONEDDS_URI=file:///workspace/configs/cyclonedds/two_pc_gpu.xml
    -v "$DATASET_HOST:/tmp/grasp_place_bimanual_demos:ro"
    # Required by BOTH learner and actor -- each constructs its own SAC
    # policy locally via make_policy(cfg.policy), and the config's
    # policy.pretrained_path points here. Without this mount the path
    # doesn't exist in the container, so from_pretrained() falls through
    # to treating it as a HF Hub repo id and fails validation instead.
    # 2026-10-19: bc_grasp_place_v2 -- supersedes v1, which was trained
    # without BC_ACTION_BOUNDS=auto, so its raw-degree joint targets got
    # hard-clamped to +-0.99 (destroying virtually all positional signal
    # for the 14 arm-joint dims) and it never saved an action_bounds.json
    # at all, so deployment also got zero de-normalization -- matches the
    # real-hardware symptom exactly ("goes straight down, opens/closes
    # gripper a couple times"). v2 fixes both: real per-dim bounds now
    # saved and auto-loaded, PLUS 520 human-correction steps (x3 weight)
    # folded in from the v1 real-hardware session via BC_TRAJ_DIRS. Train
    # corr now 0.994-1.0 across all 16 dims, no held-out set -- corrections
    # (weighted) now outweigh the 5 original demos ~4.5:1 in the training
    # mix, so this checkpoint's character leans toward "recovering from a
    # broken run" more than a balanced reflection of the clean demos.
    -v "$MAIN_REPO/outputs/bc_staging/bc_grasp_place_v2:/workspace/configs/hilserl/bc_grasp_place_poc:ro"
    # Mount the RUN's parent dir, not /workspace/outputs/<role> directly --
    # each process creates its own leaf subdir at startup, so it must not
    # already exist inside the container (see header comment).
    -v "$RUN_DIR:/workspace/outputs"
)

# Give every container a stable, role+run-scoped name, and refuse to
# start a second one under the same name while one's already running --
# confirmed directly, 2026-10-07: multiple learner instances were started
# across the same morning (including one launched for a one-off script
# test and never explicitly stopped, now that it runs detached via -d)
# without a `stop` in between, each silently resuming from whatever
# `last` happened to be and racing the others to write the next
# checkpoint. The resulting step count (4000) is real but its training
# history is now unreconstructable.
#
# An earlier version of this check grepped `docker ps`'s own
# {{.Command}} column for "hilserl_$MODE" -- confirmed by direct test to
# never match, because docker ps truncates long commands (its own
# display already showed "/entrypoint.sh bash…" elsewhere in this
# project) well before that substring. --name avoids string-matching
# entirely: Docker itself refuses a second container under a name that's
# already in use, and the check below gives a clear message before even
# attempting that, by matching on the exact name instead of command text.
CONTAINER_NAME="hilserl-$MODE-$RUN_NAME"
EXISTING="$(docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.ID}}')"
if [[ -n "$EXISTING" ]]; then
    echo "A $MODE is already running for run $RUN_NAME ($EXISTING) -- stop it first: $0 stop" >&2
    exit 1
fi
COMMON_ARGS+=(--name "$CONTAINER_NAME")

if [[ "$MODE" == "learner" ]]; then
    RESUME_ARG=""
    if [[ "$RESUME" == "true" ]]; then
        RESUME_ARG="--resume=true"
    fi

    # --online-steps is always a RELATIVE "how many more steps this
    # session" budget, never an absolute target the caller has to compute
    # by hand. The learner's own optimization_step counter persists across
    # resumes (loaded straight from the checkpoint), but patch_learner_
    # stop_at_online_steps() in patches.py stops on an EXACT match against
    # cfg.policy.online_steps -- passing the same small relative value
    # (e.g. 800) on every resume meant that check could never fire again
    # once the real step count ran past it (confirmed directly, 2026-10-08:
    # a resume from step 5300 with --online-steps 800 would never hit
    # optimization_step == 800, silently disabling the auto-stop safety
    # net for the whole session). Fix: resolve the real current step from
    # the checkpoint being resumed from and add the user's requested
    # per-session budget to it, so the caller only ever has to think in
    # "how many more steps," never in absolute totals -- the --online-steps
    # 6100 workaround from earlier today should never be necessary again.
    #
    # (Note this does NOT apply to the actor -- see its own branch below:
    # actor.py's online loop is `for interaction_step in range(cfg.policy.
    # online_steps)`, a fresh local counter starting at 0 every process
    # launch, so the actor was already correctly relative/per-session.)
    CURRENT_STEP=0
    if [[ "$RESUME" == "true" ]]; then
        LAST_CKPT="$RUN_DIR/learner/checkpoints/last"
        if [[ -e "$LAST_CKPT" ]]; then
            # Checkpoint dirs are zero-padded step numbers (e.g. "005300") --
            # the 10# base prefix is required so bash's arithmetic context
            # doesn't misinterpret a leading-zero number as octal (which
            # would silently compute the wrong value, or error outright on
            # an 8/9 digit).
            CURRENT_STEP=$((10#$(basename "$(readlink -f "$LAST_CKPT")")))
        fi
    fi
    LEARNER_ONLINE_STEPS=$((CURRENT_STEP + ONLINE_STEPS))

    echo "[run_hilserl_bimanual] learner -- run_name=$RUN_NAME current_step=$CURRENT_STEP +$ONLINE_STEPS -> online_steps=$LEARNER_ONLINE_STEPS save_freq=$SAVE_FREQ resume=$RESUME actor_warmup_steps=$ACTOR_WARMUP_STEPS actor_bc_anchor_weight=$ACTOR_BC_ANCHOR_WEIGHT run_dir=$RUN_DIR"

    # Runs detached (-d) rather than via setsid/nohup/disown -- Docker's own
    # daemon owns the container's lifecycle this way, completely independent
    # of this shell, this terminal, or this terminal's session. That also
    # sidesteps the still-unexplained issue from 2026-10-06/07 where a
    # setsid+nohup+disown'd learner process still died shortly after a
    # Ctrl+C in the ACTOR's own (different) terminal tab -- there is no
    # backgrounded shell job left for anything to reach anymore.
    CONTAINER_ID="$(docker run -d "${COMMON_ARGS[@]}" \
        -e HILSERL_ACTOR_WARMUP_STEPS="$ACTOR_WARMUP_STEPS" \
        -e HILSERL_CRITIC_ACTION_EMBED_DIM="$CRITIC_ACTION_EMBED_DIM" \
        -e HILSERL_ACTOR_BC_ANCHOR_WEIGHT="$ACTOR_BC_ANCHOR_WEIGHT" \
        "$IMAGE" \
        bash -c ". /opt/ros/jazzy/setup.sh && . /workspace/install/setup.sh && ros2 run lerobot_control hilserl_learner --config_path=$CONFIG_CONTAINER_PATH --output_dir=/workspace/outputs/learner --policy.online_steps=$LEARNER_ONLINE_STEPS --save_freq=$SAVE_FREQ $RESUME_ARG")"
    echo "[run_hilserl_bimanual] container $CONTAINER_ID started, waiting for it to be genuinely ready..."
    echo "[run_hilserl_bimanual] (full logs: docker logs -f $CONTAINER_ID)"

    # "gRPC server started" fires BEFORE the (possibly slow, and on resume
    # previously broken) replay-buffer reload -- confirmed directly, this is
    # what made a crashed-during-reload learner look "ready" on 2026-10-07.
    # "Starting learner thread" only fires once that reload has genuinely
    # finished, so it's the real readiness signal.
    WAITED=0
    while ! docker logs "$CONTAINER_ID" 2>&1 | grep -q "Starting learner thread"; do
        if ! docker ps -q --no-trunc | grep -qx "$CONTAINER_ID"; then
            echo "[run_hilserl_bimanual] learner container exited before becoming ready -- check: docker logs $CONTAINER_ID" >&2
            exit 1
        fi
        WAITED=$((WAITED + 1))
        if [[ "$WAITED" -ge 180 ]]; then
            echo "[run_hilserl_bimanual] still not ready after 180s -- check: docker logs $CONTAINER_ID" >&2
            exit 1
        fi
        sleep 1
    done
    RESUME_HINT=""
    if [[ "$RESUME" == "true" ]]; then
        RESUME_HINT=" --resume"
    fi
    echo "[run_hilserl_bimanual] learner ready (resuming from step $CURRENT_STEP, this session's target: $LEARNER_ONLINE_STEPS). Start the actor in another terminal:"
    echo "  $0 actor --run-name $RUN_NAME --online-steps $ONLINE_STEPS$RESUME_HINT"
else
    # The actor never has persistent training state of its own to resume --
    # it just reconnects and receives fresh policy params from the learner.
    # --resume=true is deliberately NEVER passed here: validate()'s generic
    # resume branch (lerobot/configs/train.py) reinterprets --config_path as
    # a CHECKPOINT's train_config.json and overwrites policy.pretrained_path
    # to that file's parent dir -- harmless for the learner (handle_resume_logic()
    # immediately replaces the whole config from the real checkpoint anyway)
    # but wrong for the actor, which has no such follow-up and ends up trying
    # to load model.safetensors from the wrong directory (confirmed by hitting
    # this directly: FileNotFoundError .../configs/hilserl/model.safetensors).
    # A fresh, never-reused output subdir per attempt avoids the *actual*
    # problem (output_dir already exists) without needing --resume at all.
    ACTOR_OUTPUT_SUBDIR="actor_$(date +%s)"

    # When --resume is given, start the actor's OWN local policy from the
    # run's latest saved checkpoint instead of the static bc_bimanual_poc
    # default baked into the YAML. Without this, update_policy_parameters()
    # only swaps the actor onto the learner's current (resumed) weights at
    # an episode BOUNDARY (confirmed by reading actor.py directly -- it's
    # called only inside `if done or truncated:`, never on a timer) -- a
    # session that crashes before any episode completes would otherwise run
    # entirely on the stale original BC checkpoint despite --resume, which
    # is exactly what happened and looked like a regression. Deliberately
    # NOT using --resume=true here (see the comment above) -- this is a
    # plain, independent --policy.pretrained_path override.
    PRETRAINED_ARG=""
    if [[ "$RESUME" == "true" ]]; then
        LAST_CKPT_MODEL="$RUN_DIR/learner/checkpoints/last/pretrained_model"
        if [[ ! -e "$LAST_CKPT_MODEL" ]]; then
            echo "--resume given but no checkpoint found at $LAST_CKPT_MODEL -- can't start the actor from it." >&2
            exit 1
        fi
        PRETRAINED_ARG="--policy.pretrained_path=/workspace/outputs/learner/checkpoints/last/pretrained_model"
    fi

    echo "[run_hilserl_bimanual] actor -- run_name=$RUN_NAME online_steps=$ONLINE_STEPS resume=$RESUME run_dir=$RUN_DIR"
    echo "[run_hilserl_bimanual] needs a real TTY for keyboard intervention -- running interactively."
    exec docker run -it "${COMMON_ARGS[@]}" \
        -e HILSERL_CRITIC_ACTION_EMBED_DIM="$CRITIC_ACTION_EMBED_DIM" \
        "$IMAGE" \
        bash -c ". /opt/ros/jazzy/setup.sh && . /workspace/install/setup.sh && ros2 run lerobot_control hilserl_actor --config_path=$CONFIG_CONTAINER_PATH --output_dir=/workspace/outputs/$ACTOR_OUTPUT_SUBDIR --policy.online_steps=$ONLINE_STEPS $PRETRAINED_ARG"
fi
