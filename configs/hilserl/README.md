# configs/hilserl: what's current

See `ros2/.../hilserl/HILSERL_STATUS.md` for the code-side equivalent of
this inventory.

## Current production config

**`train_rl_push_bottle_jointspace.yaml`** -- the real single-arm
joint-space actor/learner config. `teleop.action_mode: joint_space`,
`processor.inverse_kinematics: null`, dataset `demos_v2_jointspace` (40
episodes, relabeled from the original EE-delta recordings so the action
column is the follower's own next-tick state, not the leader-derived
EE-delta that was confirmed causally disconnected from real follower
motion). `policy.pretrained_path: bc_jointspace_poc` (-> `outputs/
bc_jointspace_v1`, mounted read-only at run time, not baked into this repo).

**`eval_sac_jointspace_push_bottle.yaml`** -- matching eval-only config for
the same joint-space checkpoint, used with `eval_checkpoint.py`.

## Bimanual (2-arm, 4-camera) -- `feat/hilserl-bimanual-4cam` branch only

**`env_push_bottle_bimanual_jointspace.yaml`** -- recording-only
(`hilserl_gym_manipulator`, no RL). **Use this first** -- no bimanual demos
exist yet, so `train_rl_push_bottle_bimanual_jointspace.yaml` below has
nothing to train on until this produces a dataset.

**`train_rl_push_bottle_bimanual_jointspace.yaml`** -- actor/learner
config, same joint-space convention as the single-arm config, 14-dim
state/action (left-then-right), 4 cameras. Has no `dataset:`/
`policy.pretrained_path` yet -- add both once the recording config above
has produced real demos, mirroring the single-arm config's own.

Keep these two files' `env:` blocks (camera topics, controller names,
per-arm limits) in sync if either changes -- see each file's own header
comment.

## Historical / superseded (EE-delta era, before the 2026-09-30 joint-space pivot)

These predate the finding that the EE-delta action label was causally
disconnected from the follower's real motion. Kept for reference, not used
by the current actor/learner loop:

- **`env_push_bottle.yaml`** -- the original EE-delta recording/task
  config (`hilserl_gym_manipulator`, no RL). Still has real, measured
  `end_effector_bounds` for this rig (not a placeholder) -- if EE-delta
  recording is ever needed again, this is the config to start from.
- **`train_rl_push_bottle.yaml`** -- the original EE-delta actor/learner
  config. Its `bc_pretrained_v8`/DAgger-round history (see `patches.py`'s
  `patch_sac_actor_warmup` docstring) is the source of the
  `patch_sac_actor_warmup`/`patch_sac_freeze_temperature` fixes that the
  joint-space config also benefits from.
- **`bc_pretrained/` through `bc_pretrained_v8/`** -- EE-delta BC
  checkpoints (v5-v8 are the three DAgger rounds). Superseded by
  `outputs/bc_jointspace_v1` for the current task.
- **`rl_ckpt_0923_step4000/`, `rl_ckpt_0924_step4000/`** -- EE-delta RL
  checkpoints from early training runs, before the critic
  action-scale-mismatch fix (`patch_sac_normalizes_replay_buffer_actions_for_critic_loss`).
  Do not use as a `pretrained_path` for new joint-space training.
- **`eval_act_push_bottle.yaml`** -- EE-delta ACT eval config, from the
  ACT-vs-HIL-SERL data-causality investigation.

## Test / diagnostic only (not safe to point real training at)

- **`_stage3_test_config.yaml`**, **`_stage5_test_config.yaml`** --
  fake-hardware / widened-bounds configs used during initial bring-up
  (Stage 3, Stage 5 in the project history). `_stage5_test_config.yaml`
  in particular has `end_effector_bounds` deliberately widened to
  `[-1,-1,-1]`/`[1,1,1]` -- never use it as a template for a real config.
- **`_stage6c_dataset_diag_config.yaml`**, **`_stage6d_jointspace_dataset_diag_config.yaml`**
  -- fake-hardware dataset-wiring smoke tests (stock keyboard teleop
  substituted in, bypasses the real switch_controller dependency).
- **`train_rl_push_bottle_smoketest.yaml`** -- the original pure-plumbing
  gRPC smoke test, before `dataset:` was wired in. Superseded by
  `train_rl_push_bottle_jointspace.yaml` for anything beyond a bare
  actor<->learner connectivity check.
- **`eval_act_jointspace_push_bottle.yaml`** -- joint-space ACT eval,
  from the same data-causality investigation as `eval_act_push_bottle.yaml`
  above; answered its question (joint-space is correct), not part of the
  regular loop.

## Shared, still-used regardless of config

- **`resnet10/`** -- the frozen vision encoder backbone, used by every
  SAC config above.
- **`urdf/openarm_teleop_kinematics.urdf`** -- mesh-stripped URDF for
  `RobotKinematics`/`placo`; `openarm_teleop.urdf` (not mesh-stripped) is
  kept alongside it only for any future visualization use.
