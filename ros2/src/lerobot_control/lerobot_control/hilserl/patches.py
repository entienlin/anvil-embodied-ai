"""Runtime patches for lerobot 0.5.1 gaps affecting this project's
push-bottle HIL-SERL task, applied via apply_all_patches() from all three
entrypoints (run_actor.py, run_learner.py, run_gym_manipulator.py).

INDEX -- every patch currently registered in apply_all_patches(), in that
function's own order, with the date each was first introduced (from git
log) and the one-line symptom it exists to fix. Keep this in sync when
adding/removing a patch -- it exists specifically so a future "did
something change" question (ours or a real-hardware fault) can be answered
by scanning this list instead of re-reading the whole file, or worse,
re-deriving it live during an active debugging session.

  2026-09-17  patch_map_delta_action_for_gripperless_robots
      MapDeltaActionToRobotActionStep.action() unconditionally pop()s
      "gripper" with no use_gripper guard -- crashes gripper-less robots.
  2026-09-17  patch_ee_bounds_and_safety_to_clamp_not_crash
      EEBoundsAndSafety.action() computes a clamp for an oversized EE jump
      then raises anyway instead of using it -- crashed real give-backs.
  2026-09-17  patch_inverse_kinematics_to_always_reseed_from_real_joints
      InverseKinematicsRLStep reseeds its IK solver from its OWN prior
      output, not the real robot -- drifts during human intervention, then
      snaps to a bogus joint config on give-back ("shutdown position").
  2026-09-17  patch_robot_env_reset_to_keep_publishing
      RobotEnv.reset()'s blocking sleep publishes zero commands for its
      whole duration -- a real, guaranteed gap in the command topic.
  2026-09-18  patch_robot_env_reset_to_home_via_arms_resetter
      Upgrades the above hold-in-place reset to a real homing move via the
      devbox's arms_resetter_node, with a post-homing baseline controller
      handoff (policy drives by default after every reset).
  2026-09-29  patch_leader_skip_reset_when_stopping
      The actor's own auto-stop can race the next episode's reset(),
      losing ROS context mid-homing-call for an episode that'll never run.
  (bimanual branch)  patch_multi_leader_skip_reset_when_stopping
      Bimanual counterpart to the patch above -- AnvilOpenArmMultiLeader's
      perform_episode_reset_and_wait() lives on a different class, so the
      single-arm patch never covers it; same stop/reset race otherwise.
  2026-09-22  patch_sac_deterministic_inference
      SACPolicy.select_action() resamples fresh stochastic noise every
      10Hz tick -- causes persistent per-tick "creeping" jitter on real
      hardware. Patched to return tanh(mean) (the mode) instead.
  2026-10-01  patch_sac_loads_action_bounds_on_from_pretrained
      Auto-loads action_bounds.json (joint-space de-normalization) on
      EVERY SACPolicy.from_pretrained() call, not just one entrypoint --
      missing this caused a real "arm went to dead position" incident.
  2026-10-01  patch_sac_normalizes_replay_buffer_actions_for_critic_loss
      compute_loss_critic() mixed raw-degree replay-buffer actions with
      the actor's own tanh-bounded (-1,1) samples -- a TD-learning
      scale-mismatch that corrupted the critic for joint-space actions.
  2026-10-01  patch_sac_checkpoints_carry_action_bounds
      Periodic RL checkpoint saves never copied action_bounds.json
      forward -- would silently reintroduce the de-norm bug for any
      checkpoint later used as a pretrained_path.
  2026-09-22  patch_sac_freeze_temperature
      SAC's auto-entropy tuner re-inflates action noise back up within
      ~250 steps regardless of temperature_init -- zeroes its loss
      gradient so the configured temperature actually holds.
  2026-09-24  patch_learner_episode_stats_logging
      Surfaces per-episode intervention-rate stats (the docs' own
      definition of HIL-SERL progress) that lerobot already computes but
      never prints.
  2026-10-20  patch_sac_critic_action_embedding
      Confirmed (offline Q-variance diagnostic, this task's own real
      checkpoints): the critic is nearly action-blind -- action is only
      14-16 of ~1280 critic-input dims, pure raw concatenation. Opt-in
      (HILSERL_CRITIC_ACTION_EMBED_DIM) action embedding before
      concatenation. Companion patch_sac_tolerates_shape_mismatch_on_load()
      is invoked conditionally from inside this one (not independently
      registered below) to let a pre-existing checkpoint load with only
      the now-reshaped critic layers starting fresh.
  2026-09-29  patch_sac_actor_warmup
      First real-RL session: gradient updates measurably WORSENED the
      policy immediately (intervention rate ~0.39-0.50 -> ~0.54) -- an
      undertrained critic giving the actor bad early guidance. Opt-in
      (HILSERL_ACTOR_WARMUP_STEPS) critic-only warm-up window. Redesigned
      2026-10-12 from a hard .step()-skip to a linear LR ramp (skipping
      pinned Adam's internal step counter at 0 for the whole window,
      making the first unskipped call always a maximal cold-start step)
      and to also freeze the shared encoder for the window (optimizers["critic"]
      otherwise keeps updating it regardless of the actor optimizer
      freeze). As of 2026-10-15: tested at 0/200/800 steps across real
      sessions with no effect on the episode-2-only-success pattern under
      active investigation (see HILSERL_STATUS.md) -- warmup duration is
      not the lever for that problem.
  2026-10-07  patch_learner_resume_preserves_cli_online_steps_and_save_freq
      handle_resume_logic() unconditionally replaces the whole cfg with
      the checkpoint's OWN saved train_config.json, silently discarding
      whatever --policy.online_steps/--save_freq this invocation was
      launched with -- also made the online-steps auto-stop safety net
      permanently inert across any resume.
  2026-10-07  patch_learner_resume_offline_buffer_from_source
      On --resume=true, reloading the offline buffer from its saved
      snapshot hits a real lerobot save/load mismatch (snapshot is
      missing meta/episodes/) that silently falls back to a Hub download
      using a repo_id never actually pushed to HF -- a hard 401 crash
      blocking every resumed actor connection.
  2026-10-07  patch_learner_resume_online_buffer_tolerates_broken_snapshot
      Same root cause as the patch above, applied to the online replay
      buffer (genuinely-accumulated live transitions with no other source
      to fall back to) -- reload failure now logs and continues with an
      empty online buffer instead of crashing the whole resume.
  2026-10-07  patch_learner_pauses_training_when_actor_idle
      The learner's main loop never pauses based on whether an actor is
      still connected -- one real completed episode was enough to let it
      keep training for 550+ further optimization steps across three
      subsequent sessions that each crashed before completing a single
      episode, grounded in nothing but stale data. Verified working
      overnight (2026-10-08) across a ~14-hour unattended gap.
  2026-10-08  patch_logging_includes_tracebacks
      lerobot's init_logging() never renders exc_info, so grpc's own
      logging.exception() calls (e.g. "Exception iterating requests!")
      silently lost the real underlying error -- blocked root-causing two
      separate real session-ending actor crashes from the log alone.
  2026-09-24  patch_robot_env_step_keeps_intervention_flag
      RobotEnv.step() silently zeroes the per-step IS_INTERVENTION flag,
      so logged intervention rate always read 0.0.
  2026-10-01  patch_step_env_squeezes_batched_action
      SAC's batched (1,N) action tensor was never squeezed/CPU-transferred
      for the joint-space path (EE-delta's IK chain squeezed it by
      coincidence) -- crashed env.step(). Also adds ACTOR_TRACE_OUT
      per-tick trace logging. (2026-10-02, bimanual) also replaces
      control_loop()'s hardcoded 3-element non-intervening "neutral_action"
      with the robot's real current joint positions when it's shorter than
      the robot's actual joint count -- the EE-delta IK chain always fixed
      this up before reaching env.step(), joint-space mode has no such
      chain to rely on.
  2026-10-20  patch_learner_checkpoint_saves_dataset_atomically
      save_training_checkpoint() rmtree's the last GOOD replay-buffer
      snapshot BEFORE the slow rewrite -- any interruption (Ctrl+C,
      hardware fault, OOM) between those two steps left dataset_dir gone
      or half-written, with no way back, meaning the online buffer was
      silently lost on nearly every real --resume. Fix: write to a
      sibling tmp dir, atomic-rename swap only once the write succeeds.
  2026-09-21  patch_learner_stop_at_online_steps
      lerobot's learner loop never exits on reaching online_steps -- kept
      re-training on the same finite buffer forever (seen: 4000+ steps for
      a configured 1000-step budget).
  2026-09-24  patch_intervention_flag_in_complementary_info
      actor.py's complementary_info never set IS_INTERVENTION, so the
      learner's own offline-buffer-growth-on-intervention logic was dead
      code in practice.
  2026-09-21  patch_learner_progress_visibility
      Learner-side gRPC/training logging never appeared in the terminal --
      a silent learner and a stuck one looked identical.
  2026-09-17  patch_train_pipeline_config_validate_for_optional_dataset
      TrainPipelineConfig.validate() crashes on dataset=None even though
      it's declared Optional specifically to support pure-online RL.

Patch #1 (patch_map_delta_action_for_gripperless_robots) detail follows,
since its sibling (the observation-side gap) is worked around via config
rather than a second patch here -- both patches target the same class of
bug: a processor step unconditionally
reads/pops a "gripper"-named key with no `use_gripper` check, even though
the step immediately upstream of it *does* correctly gate that key's
presence on `use_gripper`. Confirmed via direct source read, not guessed:

- Observation side: `ForwardKinematicsJointsToEEObservation` /
  `compute_forward_kinematics_joints_to_ee`
  (robots/so_follower/robot_kinematic_processor.py) does
  `joints["gripper.pos"]` unconditionally. Worked around by disabling
  `add_ee_pose_to_observation` in the env config instead of patching --
  it's a pure observation convenience feature, not needed here.
- Action side (this file): `MapDeltaActionToRobotActionStep.action()`
  (processor/delta_action_processor.py) does `action.pop("gripper")`
  unconditionally, while its sibling `MapTensorToDeltaActionDictStep` only
  adds "gripper" when `use_gripper=True`. No config workaround exists here
  (this step is unconditionally in the action pipeline), so it's patched
  directly.

Safe because `gripper_vel` (the value built from the popped "gripper") only
feeds `GripperVelocityToJoint`, which is itself only added to the action
pipeline when `use_gripper=True` -- so for a gripper-less task the value is
fully inert downstream regardless of what it's set to.
"""

from lerobot.processor.delta_action_processor import MapDeltaActionToRobotActionStep


def patch_map_delta_action_for_gripperless_robots() -> None:
    """Idempotent: pop("gripper", 0.0) instead of pop("gripper")."""
    if getattr(MapDeltaActionToRobotActionStep, "_anvil_gripperless_patch_applied", False):
        return

    _orig_action = MapDeltaActionToRobotActionStep.action

    def _patched_action(self, action):
        action.setdefault("gripper", 0.0)
        return _orig_action(self, action)

    MapDeltaActionToRobotActionStep.action = _patched_action
    MapDeltaActionToRobotActionStep._anvil_gripperless_patch_applied = True


def patch_ee_bounds_and_safety_to_clamp_not_crash() -> None:
    """Real hardware finding (Stage 5): `EEBoundsAndSafety.action()`
    (robots/so_follower/robot_kinematic_processor.py) computes a clamped
    position for an oversized EE jump (`pos = self._last_pos + dpos *
    (self.max_ee_step_m / n)`) and then immediately discards it with an
    unconditional `raise ValueError(...)` right after -- dead code. The
    class's own docstring says it "moderates the command to prevent large,
    sudden movements between consecutive steps," but the actual code
    crashes the whole control loop instead of moderating anything.
    `max_ee_step_m` defaults to 0.05m and is NOT exposed via any config
    field HIL-SERL's own `make_processors()` reads -- confirmed via source,
    there is no way to raise this threshold from our YAML.

    This matters specifically for a REAL leader arm (this project's
    intervention design, not gamepad/keyboard): `EEReferenceAndDelta`
    recomputes its reference via fresh FK on the follower's ACTUAL current
    position every tick, so during real physical intervention -- where a
    human's hand can move the leader arm far faster between 10Hz ticks than
    a small-delta gamepad/keyboard ever would -- this step's OWN internal
    `_last_pos` memory can lag behind the real position, producing an
    oversized computed "jump" and crashing the script on give-back.
    Reproduced for real: "EE jump 0.275m > 0.05m" crashed the control loop
    immediately after a real intervention handoff, leaving the script
    unable to switch back to policy control or save the episode.

    Patching this to actually use the already-computed clamped `pos`
    doesn't reduce real safety: during intervention,
    follower_r_forward_position_controller is INACTIVE (only the devbox's
    native teleop_controller drives the real arm then), so whatever target
    this step computes is inert regardless of whether it's clamped or
    would have crashed. On give-back, the incoming target is already a
    fresh FK of the real actual position (our leader's delta is zero when
    not intervening) -- clamping only affects how fast this step's
    internally-tracked `_last_pos` catches up to that real position (capped
    at max_ee_step_m per tick, exactly the "no large sudden movements"
    protection the docstring promises), not whether the eventual published
    target is safe. AnvilOpenArmFollower.send_action()'s independent
    joint-space ActionLimiter clamp (max_position_delta=0.30 rad) remains
    active on every published command regardless of this patch.
    """
    from lerobot.robots.so_follower.robot_kinematic_processor import EEBoundsAndSafety

    if getattr(EEBoundsAndSafety, "_anvil_clamp_not_crash_patch_applied", False):
        return

    import numpy as np

    def _patched_action(self, action):
        x, y, z = action["ee.x"], action["ee.y"], action["ee.z"]
        wx, wy, wz = action["ee.wx"], action["ee.wy"], action["ee.wz"]
        if None in (x, y, z, wx, wy, wz):
            raise ValueError(
                "Missing required end-effector pose components: x, y, z, wx, wy, wz must all be present in action"
            )

        pos = np.clip(np.array([x, y, z], dtype=float), self.end_effector_bounds["min"], self.end_effector_bounds["max"])

        if self._last_pos is not None:
            dpos = pos - self._last_pos
            n = float(np.linalg.norm(dpos))
            if n > self.max_ee_step_m and n > 0:
                pos = self._last_pos + dpos * (self.max_ee_step_m / n)

        self._last_pos = pos

        action["ee.x"], action["ee.y"], action["ee.z"] = float(pos[0]), float(pos[1]), float(pos[2])
        action["ee.wx"], action["ee.wy"], action["ee.wz"] = float(wx), float(wy), float(wz)
        return action

    EEBoundsAndSafety.action = _patched_action
    EEBoundsAndSafety._anvil_clamp_not_crash_patch_applied = True


def patch_inverse_kinematics_to_always_reseed_from_real_joints() -> None:
    """Real hardware finding (Stage 5): after the EE-bounds clamp fix above
    stopped the control loop from crashing on give-back, the NEXT real test
    exposed a deeper, different bug -- the follower snapped to a bizarre
    "shutdown"-looking joint configuration immediately after give-back, with
    no error at all.

    `InverseKinematicsRLStep` (robots/so_follower/robot_kinematic_processor.py)
    is constructed by `make_processors()` with `initial_guess_current_joints=False`
    (hardcoded, not exposed via any YAML field). With that setting, its
    `__call__` seeds `self.q_curr` from the real robot's actual joints only
    ONCE ever (its very first call) -- every call after that reseeds from
    `self.q_curr = q_target`, i.e. the solver's OWN previous output, never
    the real robot's actual joints again.

    This is fine for the stock use case (gamepad/keyboard driving the SAME
    IK-computed joint trajectory the whole time -- solver state and real
    robot state can't diverge). It breaks down for a REAL leader arm: during
    intervention, the real follower is driven directly by the devbox's
    native teleop mirroring, completely bypassing this step -- the real
    robot's actual joints and the solver's internal `q_curr` are now two
    independent, unsynchronized trajectories. This OpenArm is 7-DOF (one
    more than a 6D end-effector pose needs), so there are infinitely many
    joint configurations achieving "the same" EE pose -- even if the
    EE-space target tracks the real robot's real position closely tick by
    tick (true after the clamp fix above), the specific JOINT-SPACE solution
    the solver settles on can drift into a completely different elbow
    configuration than the robot's real one. At the moment of give-back,
    whatever drifted, never-resynced `q_curr` the solver landed on gets
    published directly to the real joints -- a large, sudden joint-space
    snap with no error, matching "went straight to a shutdown position all
    of a sudden" exactly.

    Fix: force every call to seed from the real robot's actual current
    joints (`q_raw`, already read from the live observation every tick),
    matching `initial_guess_current_joints=True` behavior regardless of
    what `make_processors()` constructed the step with. This is exactly how
    Stage 0's own kinematics validation exercised this solver (seeded from
    a real, valid current pose every tick, small delta per step) --
    matches the one scenario this solver was actually confirmed accurate
    for, and eliminates solver-internal drift entirely rather than trying
    to detect and resync only at the give-back moment.
    """
    from lerobot.robots.so_follower.robot_kinematic_processor import InverseKinematicsRLStep

    if getattr(InverseKinematicsRLStep, "_anvil_always_reseed_patch_applied", False):
        return

    import numpy as np
    from lerobot.processor.pipeline import TransitionKey

    def _patched_call(self, transition):
        new_transition = dict(transition)
        action = new_transition.get(TransitionKey.ACTION)
        if action is None:
            raise ValueError("Action is required for InverseKinematicsEEToJoints")
        action = dict(action)

        x = action.pop("ee.x")
        y = action.pop("ee.y")
        z = action.pop("ee.z")
        wx = action.pop("ee.wx")
        wy = action.pop("ee.wy")
        wz = action.pop("ee.wz")
        gripper_pos = action.pop("ee.gripper_pos")

        if None in (x, y, z, wx, wy, wz, gripper_pos):
            raise ValueError(
                "Missing required end-effector pose components: ee.x, ee.y, ee.z, "
                "ee.wx, ee.wy, ee.wz, ee.gripper_pos must all be present in action"
            )

        observation = new_transition.get(TransitionKey.OBSERVATION).copy()
        if observation is None:
            raise ValueError("Joints observation is require for computing robot kinematics")

        q_raw = np.array(
            [float(v) for k, v in observation.items() if isinstance(k, str) and k.endswith(".pos")],
            dtype=float,
        )

        # The one-line fix: ALWAYS reseed from the real robot's actual
        # current joints, never from the solver's own previous output.
        self.q_curr = q_raw

        from scipy.spatial.transform import Rotation

        t_des = np.eye(4, dtype=float)
        t_des[:3, :3] = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
        t_des[:3, 3] = [x, y, z]

        q_target = self.kinematics.inverse_kinematics(self.q_curr, t_des)
        self.q_curr = q_target

        for i, name in enumerate(self.motor_names):
            if name != "gripper":
                action[f"{name}.pos"] = float(q_target[i])
            else:
                action["gripper.pos"] = float(gripper_pos)

        new_transition[TransitionKey.ACTION] = action
        complementary_data = new_transition.get(TransitionKey.COMPLEMENTARY_DATA, {})
        complementary_data["IK_solution"] = q_target
        new_transition[TransitionKey.COMPLEMENTARY_DATA] = complementary_data
        return new_transition

    InverseKinematicsRLStep.__call__ = _patched_call
    InverseKinematicsRLStep._anvil_always_reseed_patch_applied = True


def patch_robot_env_reset_to_keep_publishing() -> None:
    """Real hardware finding (Stage 5, real multi-episode collection):
    `RobotEnv.reset()` (robots/so_follower's caller, `lerobot/rl/
    gym_manipulator.py`) does a single blocking
    `precise_sleep(reset_time_s)` between episodes with `reset_pose is
    None` (our config -- `fixed_reset_joint_positions: null`, a known,
    accepted limitation, see the plan). Confirmed directly from source:
    during that entire multi-second window, ZERO commands are published to
    `/follower_r_forward_position_controller/commands` -- not "usually
    zero", structurally guaranteed zero every single reset, since nothing
    else runs on this thread during the sleep.

    The "follower dead" pattern recurred four times during real testing,
    always arm-wide on follower_r specifically (never leader_r, which is
    driven directly by the devbox's 500Hz teleop-mirroring loop with no
    topic dependency at all) and always below what ros2_control's own
    state reporting can see (every controller still shows "active").
    Devbox's motor-status logging on the 4th occurrence localized the
    fault to ~5.3s after follower_r_forward_position_controller was
    activated -- squarely inside the window this reset() sleep (plus
    dataset.save_episode()'s own I/O time) would occupy. Leading
    hypothesis, well-supported by all of the above: a command-timeout
    watchdog in the motor firmware disables torque if no fresh position
    command arrives within some threshold, and this reset design
    guarantees exactly that gap on every single episode boundary.

    Superseded for real hardware by
    `patch_robot_env_reset_to_home_via_arms_resetter()` below, which does
    real active homing (never silent) instead of a hold-loop and closes a
    second, more serious gap this fix alone didn't: the old hold-in-place
    approach never touched controller state or intervention bookkeeping,
    so it structurally left the untrained policy in sole, unsupervised
    control of the real arm for the entire `reset_time_s` window after
    every episode boundary (confirmed as the proximate cause of a real
    fault). Kept here, still applied first, as the fallback for any run
    without a registered `AnvilOpenArmLeader` (e.g. Stage 3's stock
    `keyboard_ee` fake-hardware dry run) -- the homing patch below
    delegates to this one's own hold-loop when no leader is registered.
    """
    from lerobot.rl.gym_manipulator import RobotEnv

    if getattr(RobotEnv, "_anvil_reset_keepalive_patch_applied", False):
        return

    import gymnasium as gym
    import numpy as np
    from lerobot.rl.gym_manipulator import log_say, reset_follower_position
    from lerobot.teleoperators.utils import TeleopEvents
    from lerobot.utils.robot_utils import precise_sleep

    def _hold_in_place_reset(self, *, seed=None, options=None):
        import time as _time

        start_time = _time.perf_counter()
        if self.reset_pose is not None:
            log_say("Reset the environment.", play_sounds=True)
            reset_follower_position(self.robot, np.array(self.reset_pose))
            log_say("Reset the environment done.", play_sounds=True)

        hold_obs = self.robot.get_observation()
        hold_action = {f"{j}.pos": hold_obs[f"{j}.pos"] for j in self._joint_names}
        while _time.perf_counter() - start_time < self.reset_time_s:
            self.robot.send_action(hold_action)
            precise_sleep(0.1)

        gym.Env.reset(self, seed=seed, options=options)

        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    RobotEnv.reset = _hold_in_place_reset
    RobotEnv._anvil_hold_in_place_reset = _hold_in_place_reset
    RobotEnv._anvil_reset_keepalive_patch_applied = True


def patch_robot_env_reset_to_home_via_arms_resetter() -> None:
    """Real hardware finding, ninth "follower dead" occurrence: the old
    hold-in-place reset (see `patch_robot_env_reset_to_keep_publishing()`
    above) kept the command topic alive, but never touched controller
    state or intervention bookkeeping -- so it structurally left the
    untrained policy in sole, unsupervised control of the real arm for
    `reset_time_s` after EVERY episode boundary, regardless of what the
    human did with the intervention key (RobotEnv.reset() never calls
    into the teleop device at all). Confirmed as the proximate cause of a
    real fault: the human's own space-bar toggle can't re-engage
    teleop_controller until the NEXT real step calls get_action() again,
    which never happens during reset().

    Fix: replace the hold-loop with a real home-position reset via the
    devbox's /arms_resetter/reset (ArmsResetterClient), orchestrated by
    AnvilOpenArmLeader.perform_episode_reset_and_wait() -- releases
    follower_r_forward_position_controller, waits for a real "homed"
    completion (not just the service ack), then explicitly reestablishes
    "policy drives by default" as the clean per-episode baseline (the
    same thing connect() already does once at session start, just
    repeated at every episode boundary now). This is also what finally
    makes real mixed-control HIL-SERL sessions possible: space becomes a
    per-episode choice anchored to a real, observable "homed" event
    instead of an ambiguous timer.

    Requires `patch_robot_env_reset_to_keep_publishing()` to already be
    applied (defines the hold-in-place fallback this delegates to when no
    AnvilOpenArmLeader is registered -- e.g. Stage 3's stock keyboard_ee
    fake-hardware dry run, which has no arms_resetter equivalent to call).
    """
    from lerobot.rl.gym_manipulator import RobotEnv

    if not getattr(RobotEnv, "_anvil_reset_keepalive_patch_applied", False):
        raise RuntimeError(
            "patch_robot_env_reset_to_home_via_arms_resetter() requires "
            "patch_robot_env_reset_to_keep_publishing() to be applied first "
            "(defines the hold-in-place fallback) -- call apply_all_patches() "
            "rather than this function directly."
        )
    if getattr(RobotEnv, "_anvil_reset_home_patch_applied", False):
        return

    from lerobot.teleoperators.utils import TeleopEvents

    from .teleop_anvil_openarm_leader import get_registered_leader

    hold_in_place_reset = RobotEnv._anvil_hold_in_place_reset

    def _patched_reset(self, *, seed=None, options=None):
        leader = get_registered_leader()
        if leader is None:
            # No AnvilOpenArmLeader in this process (e.g. Stage 3's stock
            # keyboard_ee) -- arms_resetter isn't applicable, fall back.
            return hold_in_place_reset(self, seed=seed, options=options)

        leader.perform_episode_reset_and_wait(
            timeout_s=self.reset_time_s,
            before_homing=self.robot.clear_heartbeat,
            after_homed=self.robot.reseed_heartbeat_after_reset,
        )

        import gymnasium as gym

        gym.Env.reset(self, seed=seed, options=options)
        self.current_step = 0
        self.episode_data = None
        obs = self._get_observation()
        self._raw_joint_positions = {f"{key}.pos": obs[f"{key}.pos"] for key in self._joint_names}
        return obs, {TeleopEvents.IS_INTERVENTION: False}

    RobotEnv.reset = _patched_reset
    RobotEnv._anvil_reset_home_patch_applied = True


def patch_leader_skip_reset_when_stopping() -> None:
    """Once an auto-stop has been requested (_request_stop(), e.g.
    HILSERL_MAX_EPISODES reached), the actor's very next line is still
    going to call online_env.reset() for an episode that will never
    actually run -- lerobot's act_with_policy() has no concept of "about
    to stop" and always resets immediately after pushing the just-ended
    episode's transitions (see _request_stop()'s docstring for the crash
    this caused on real hardware). Rather than let that reset race our own
    shutdown, skip the real homing entirely once stopping: no
    switch_to_teleop()/arms_resetter call, no ROS traffic at all, just
    leave the arm exactly where forward_position_controller already holds
    it. That's the same end state a plain Ctrl+C between episodes already
    leaves today -- not a new risk, just made reliable instead of racy.
    """
    from .teleop_anvil_openarm_leader import AnvilOpenArmLeader

    if getattr(AnvilOpenArmLeader, "_anvil_skip_reset_patched", False):
        return
    AnvilOpenArmLeader._anvil_skip_reset_patched = True

    orig = AnvilOpenArmLeader.perform_episode_reset_and_wait

    def _reset_or_skip(self, timeout_s=15.0, before_homing=None, after_homed=None):
        if _auto_stop_state.get("stop_requested"):
            print(
                "[AnvilOpenArmLeader] stop already requested -- skipping the real homing "
                "reset for this (about to be discarded) episode; holding the arm as-is.",
                flush=True,
            )
            return
        return orig(self, timeout_s=timeout_s, before_homing=before_homing, after_homed=after_homed)

    AnvilOpenArmLeader.perform_episode_reset_and_wait = _reset_or_skip


def patch_multi_leader_skip_reset_when_stopping() -> None:
    """Bimanual counterpart to patch_leader_skip_reset_when_stopping()
    above -- same real-hardware race (act_with_policy() calls
    online_env.reset() for an about-to-be-discarded episode immediately
    after the Nth episode's transitions are pushed, with no stop-aware
    check of its own -- see _request_stop()'s docstring), but
    AnvilOpenArmMultiLeader.perform_episode_reset_and_wait() lives on a
    DIFFERENT class (teleop_anvil_openarm_multi_leader.py's composite, not
    AnvilOpenArmLeader itself) -- the patch above only rebinds
    AnvilOpenArmLeader's own method, so it never applies to the composite.
    Without this, a bimanual run's auto-stop (HILSERL_MAX_EPISODES) would
    still attempt a real homing reset for a discarded episode, reproducing
    the exact "arms_resetter did not report homed" crash for the composite
    leader instead of the single-arm one.

    Reuses the same process-wide _auto_stop_state dict as the single-arm
    patch -- stop_requested isn't arm-specific.
    """
    from .teleop_anvil_openarm_multi_leader import AnvilOpenArmMultiLeader

    if getattr(AnvilOpenArmMultiLeader, "_anvil_skip_reset_patched", False):
        return
    AnvilOpenArmMultiLeader._anvil_skip_reset_patched = True

    orig = AnvilOpenArmMultiLeader.perform_episode_reset_and_wait

    def _reset_or_skip(self, timeout_s=15.0, before_homing=None, after_homed=None):
        if _auto_stop_state.get("stop_requested"):
            print(
                "[AnvilOpenArmMultiLeader] stop already requested -- skipping the real "
                "homing reset for this (about to be discarded) episode; holding every "
                "arm as-is.",
                flush=True,
            )
            return
        return orig(self, timeout_s=timeout_s, before_homing=before_homing, after_homed=after_homed)

    AnvilOpenArmMultiLeader.perform_episode_reset_and_wait = _reset_or_skip


def patch_sac_deterministic_inference() -> None:
    """SAC session (2026-09-22): user reported policy motion is "a bit
    shaky (not smooth, but not the rapid shaky movement that leads to
    limp)". Root cause: `SACPolicy.select_action()` (modeling_sac.py) is
    the ONLY call site used for real-robot stepping (rl/actor.py's
    act_with_policy() -> step_env_and_process_transition() ->
    env.step(policy.select_action(...)); the learner's training path calls
    policy.forward(batch, model="actor") -> compute_loss_actor() directly,
    never select_action()) -- and it always does
    `actions, _, _ = self.actor(batch, ...)`, i.e. a fresh independent
    `dist.rsample()` draw every single 10Hz tick, discarding the 3rd
    return value (`means`, the pre-tanh mean) it already computed for
    free. Even with a small, held std (see patch_sac_freeze_temperature
    below), per-tick resampling plus our own max_command_step_rad rate
    limiter (robot_anvil_openarm.py) produces a small persistent
    "creeping" jitter: the rate limiter only caps the derivative, it
    doesn't stop each tick's target from moving toward a fresh random
    point.

    Fix: reuse the already-computed `means` and return
    torch.tanh(means) -- the distribution's mode, per
    TanhMultivariateNormalDiag.mode() (modeling_sac.py) confirms this
    (no low/high rescale is passed when constructing the dist in
    Policy.forward(), so mode() reduces to a plain tanh of the mean).
    Minimal diff from upstream: reuses the SAME forward pass, only
    changes which of the 3 already-computed outputs gets used.

    Scope: ONLY this one real-robot call site is affected.
    compute_loss_actor()/compute_loss_critic()/compute_loss_temperature()
    (the learner's training math) still call self.actor(...) directly and
    still use rsample()/log_prob() exactly as before -- SAC's training
    objective is unchanged, only what gets published to the real motors.

    Trade-off, stated plainly: this removes SAC's native action-space
    exploration noise from real rollouts. Accepted for this project --
    HIL-SERL's real exploration/correction signal here is the human
    intervention pathway (untouched by this patch), not raw per-tick
    Gaussian noise on real hardware, which is also the exact mechanism
    behind every "shaky"/"limp" fault investigated earlier in this file's
    sibling patches.
    """
    from lerobot.policies.sac.modeling_sac import SACPolicy

    if getattr(SACPolicy, "_anvil_deterministic_patched", False):
        return
    SACPolicy._anvil_deterministic_patched = True

    import torch

    @torch.no_grad()
    def _select_action_deterministic(self, batch):
        observations_features = None
        if self.shared_encoder and self.actor.encoder.has_images:
            observations_features = self.actor.encoder.get_cached_image_features(batch)
        _, _, means = self.actor(batch, observations_features)
        actions = torch.tanh(means)
        # De-normalize back to the checkpoint's own action units when a
        # joint-space (or any non-[-1,1]) action_bounds.json was loaded for
        # THIS policy instance -- see load_sac_action_bounds() in
        # run_policy_eval.py. Per-instance, not global: the original
        # EE-delta checkpoints (already close to (-1,1) natively) have no
        # such attribute and are returned unchanged, exactly as before.
        mid = getattr(self, "_anvil_action_mid", None)
        if mid is not None:
            actions = actions * self._anvil_action_half.to(actions.device) + mid.to(actions.device)
        if self.config.num_discrete_actions is not None:
            discrete_action_value = self.discrete_critic(batch, observations_features)
            discrete_action = torch.argmax(discrete_action_value, dim=-1, keepdim=True)
            actions = torch.cat([actions, discrete_action], dim=-1)
        return actions

    SACPolicy.select_action = _select_action_deterministic


def patch_sac_loads_action_bounds_on_from_pretrained() -> None:
    """Real hardware incident (2026-09-30): the joint-space BC checkpoint's
    action_bounds.json (see bc_pretrain.py's BC_ACTION_BOUNDS=auto and
    run_policy_eval.py's load_sac_action_bounds()) was wired into
    run_joint_policy_eval.py's eval script manually and verified there --
    but hilserl_actor never goes through that script. It calls straight
    into lerobot's own act_with_policy(), which constructs the policy via
    stock make_policy()/SACPolicy.from_pretrained() with no knowledge of
    our custom loader. Result: patch_sac_deterministic_inference()'s
    de-normalization branch (`if mid is not None`) never fired for the
    actor's own policy instance, sending every joint to raw tanh(means) in
    (-1,1) -- i.e. every joint to ~0+-1 degree, the same "went to a dead
    position" failure ACT had before its own normalization fix. Confirmed
    live: the arm went to that position and had to be Ctrl+C'd.

    Real fix, not per-entrypoint wiring: patch SACPolicy.from_pretrained
    itself so ANY code path that loads a SAC checkpoint with an
    action_bounds.json next to it gets the bounds attached automatically,
    with no caller needing to remember to call load_sac_action_bounds() at
    all. Fully backward compatible: checkpoints without that file (every
    original ee_delta checkpoint) hit load_sac_action_bounds()'s existing
    "file doesn't exist -> return False" guard and are unaffected.
    """
    from lerobot.policies.sac.modeling_sac import SACPolicy

    if getattr(SACPolicy, "_anvil_action_bounds_autoload_patched", False):
        return
    SACPolicy._anvil_action_bounds_autoload_patched = True

    orig_from_pretrained = SACPolicy.from_pretrained.__func__

    @classmethod
    def _from_pretrained(cls, pretrained_name_or_path, **kwargs):
        policy = orig_from_pretrained(cls, pretrained_name_or_path, **kwargs)
        from lerobot_control.hilserl.run_policy_eval import load_sac_action_bounds

        device = next(policy.parameters()).device
        load_sac_action_bounds(policy, pretrained_name_or_path, device)
        return policy

    SACPolicy.from_pretrained = _from_pretrained


def patch_sac_normalizes_replay_buffer_actions_for_critic_loss() -> None:
    """Real hardware incident (2026-09-30): a live joint-space actor/learner
    session ran for several episodes -- BC's starting behavior looked
    acceptable, then got steadily worse episode over episode until the arm
    went into "weird motion" and died (liveness watchdog). This is a
    DIFFERENT bug from the two already-fixed action_bounds.json gaps
    (select_action() not de-normalizing, and from_pretrained() not
    auto-loading bounds) -- those were both about DEPLOYMENT (what gets
    published to the real robot). This one is about TRAINING: the online
    RL loop was never touched by the action_bounds.json work at all.

    Confirmed by reading modeling_sac.py directly: `SACPolicy.forward()`
    does `actions: Tensor = batch[ACTION]` -- the RAW action straight from
    the replay buffer (offline dataset column or an online transition),
    unnormalized. For joint-space that's real degrees (e.g. -45 to +90).
    compute_loss_critic() feeds this into critic_forward() as `actions`
    for `q_preds` (the recorded transition's Q-estimate) -- but the SAME
    critic_forward() call for `q_targets` (the TD-bootstrap target) uses
    `next_action_preds`, which comes from `self.actor(...)`, i.e. a fresh
    sample from the tanh-squashed distribution -- ALWAYS bounded to
    (-1,1) by construction, regardless of action space. So the critic is
    trained on the SAME logical quantity (the action) represented at two
    wildly different scales depending on which code path produced it --
    real degrees (magnitude up to ~90) from the buffer, versus (-1,1) from
    the actor's own sampling. This is a genuine, severe TD-learning
    corruption: every critic update has to reconcile an internally
    inconsistent action representation, producing increasingly unstable Q-
    estimates, which then directly drives compute_loss_actor()'s gradient
    (it maximizes whatever the critic predicts) -- exactly matching
    "started acceptable, got worse every episode."

    compute_loss_actor()/compute_loss_temperature() have NO such bug --
    confirmed by reading both directly: neither one takes a raw `actions`
    parameter at all, they only ever use observations plus the actor's own
    freshly-sampled (self-consistent, always tanh-bounded) action.

    Fix: patch SACPolicy.forward() (the single entry point where
    `batch[ACTION]` is extracted, feeding both compute_loss_critic() and
    compute_loss_discrete_critic()) to normalize the batch's action into
    the same (-1,1) space the actor's own samples live in, using the exact
    same per-instance _anvil_action_mid/_anvil_action_half bounds already
    attached by patch_sac_loads_action_bounds_on_from_pretrained() --
    reusing the identical bounds already proven correct for inference,
    just applied in the other direction (normalize in, not de-normalize
    out). No-op (unchanged) for the original ee_delta checkpoints, which
    have no action_bounds.json and therefore no _anvil_action_mid.
    """
    from lerobot.policies.sac.modeling_sac import ACTION, SACPolicy

    if getattr(SACPolicy, "_anvil_critic_action_normalize_patched", False):
        return
    SACPolicy._anvil_critic_action_normalize_patched = True

    orig_forward = SACPolicy.forward

    def _forward(self, batch, model="critic"):
        mid = getattr(self, "_anvil_action_mid", None)
        if mid is not None and ACTION in batch:
            batch = dict(batch)
            half = self._anvil_action_half.to(batch[ACTION].device)
            batch[ACTION] = (batch[ACTION] - mid.to(batch[ACTION].device)) / half
        return orig_forward(self, batch, model=model)

    SACPolicy.forward = _forward


def patch_sac_checkpoints_carry_action_bounds() -> None:
    """Real gap found during a joint-space-consistency audit (2026-10-01,
    after a live RL session's checkpoint saved at step 250): the learner's
    periodic checkpoint saving (lerobot.utils.train_utils.save_checkpoint(),
    called from learner.py's save_training_checkpoint()) does
    `policy.save_pretrained(pretrained_dir)` and nothing else -- it has no
    knowledge of action_bounds.json, our own sidecar file (not a stock
    lerobot concept). _anvil_action_mid/_anvil_action_half are plain Python
    attributes on the policy instance, not part of its state_dict/config,
    so they're silently absent from every RL-saved checkpoint.

    Consequence if left unfixed: pointing a FUTURE run's pretrained_path at
    one of these RL checkpoints (e.g. to resume training, or to eval it)
    would hit patch_sac_loads_action_bounds_on_from_pretrained()'s "file
    doesn't exist" branch, silently return with no bounds loaded, and
    reintroduce the exact "every joint goes to ~0 degrees" bug this whole
    action_bounds.json mechanism exists to prevent -- this time with ZERO
    warning, since nothing crashes or logs an error, it just quietly
    deploys an un-de-normalized policy.

    Fix: patch save_checkpoint (both the module it's defined in and
    learner.py's own already-bound import, same two-site lesson as
    patch_step_env_squeezes_batched_action) to also write
    action_bounds.json into the new pretrained_dir whenever the policy
    being saved has _anvil_action_mid set. No-op for ee_delta checkpoints
    (no such attribute).
    """
    from lerobot.utils import train_utils

    if getattr(train_utils, "_anvil_checkpoint_bounds_patched", False):
        return
    train_utils._anvil_checkpoint_bounds_patched = True

    orig_save_checkpoint = train_utils.save_checkpoint

    def _save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, scheduler=None, **kwargs):
        orig_save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, scheduler=scheduler, **kwargs)
        mid = getattr(policy, "_anvil_action_mid", None)
        half = getattr(policy, "_anvil_action_half", None)
        if mid is None or half is None:
            return
        import json as _json
        from pathlib import Path as _Path

        from lerobot.utils.constants import PRETRAINED_MODEL_DIR

        pretrained_dir = _Path(checkpoint_dir) / PRETRAINED_MODEL_DIR
        with open(pretrained_dir / "action_bounds.json", "w") as f:
            _json.dump({"mid": mid.tolist(), "half": half.tolist()}, f)
        print(f"[SAC] carried action_bounds.json forward into {pretrained_dir}", flush=True)

    train_utils.save_checkpoint = _save_checkpoint
    import lerobot.rl.learner as learner_mod

    learner_mod.save_checkpoint = _save_checkpoint


def patch_sac_freeze_temperature() -> None:
    """SAC session (2026-09-22): user reported BC v1 and BC v4 (ResNet-10
    encoder, 11 newest human-driven episodes -- measurably a better fit to
    held-out demo frames, 0.21 vs ~0.30 mean error) "doesn't really have
    significant difference" once trained through the real actor/learner
    loop. Root cause, confirmed by reading modeling_sac.py/
    configuration_sac.py directly: SAC auto-tunes temperature (log_alpha)
    via a SEPARATE gradient descent (compute_loss_temperature():
    `temperature_loss = (-log_alpha.exp() * (log_probs + target_entropy)).mean()`),
    completely independent of temperature_init. target_entropy defaults to
    -action_dim/2 (-1.5 for our 3-dim EE-delta action) when unset (our
    config never sets it). For a narrow BC-like tanh-Gaussian, log_probs
    (a continuous DENSITY, unbounded above as std shrinks) starts far
    ABOVE that target the moment BC's tiny std (~0.005-0.02) loads, so the
    gradient pushes log_alpha up hard regardless of its starting value --
    confirmed empirically THIS SESSION: temperature_init=0.01 had ZERO
    lasting effect, sampling std still hit the 0.5 std_max cap by
    optimization step 250, identical to the temperature_init=1.0 run.
    temperature_init only sets a starting point for a value that gets
    auto-relearned away within a few hundred steps -- combined with
    policy_parameters_push_frequency=4 (the learner pushes updated weights
    to the actor roughly every 4 real seconds), a human essentially never
    sees the raw BC checkpoint drive for more than a few seconds before
    this auto-tuner has already started re-inflating its noise. Same
    mechanism erases whatever quality difference exists between BC
    checkpoints, every run, regardless of which BC you start from.

    Fix: pin the temperature so temperature_init actually holds for the
    whole run. NOT implemented as log_alpha.requires_grad_(False) -- that
    breaks: compute_loss_temperature() already wraps its self.actor(...)
    call in torch.no_grad() (so log_probs has requires_grad=False
    independent of anything), so if log_alpha.requires_grad is ALSO
    False, the resulting temperature_loss tensor has requires_grad=False
    too, and loss_temperature.backward() (called unconditionally every
    actor-update step in learner.py's loop) raises "element 0 of tensors
    does not require grad and does not have a grad_fn". Real crash risk,
    not hypothetical -- avoided here.

    Correct implementation: multiply the ORIGINAL (unmodified) loss by
    0.0 before returning it. log_alpha.requires_grad stays True
    (untouched, default), and temperature_loss stays fully inside the
    autograd graph (.backward() succeeds normally) -- but the gradient it
    produces for log_alpha is now a real, valid all-zero tensor, never
    None. Since this patch applies from policy construction, before any
    optimizer step has ever run, Adam's internal moment estimates (m_t,
    v_t) for log_alpha start at their standard zero initialization and
    stay exactly zero forever under an always-zero gradient -- log_alpha
    truly never moves from temperature_init.
    optimizers["temperature"].step() still runs every call (harmless
    no-op), and training_infos["loss_temperature"] correctly reports 0.0
    as a diagnostic signal that this patch is active.
    """
    from lerobot.policies.sac.modeling_sac import SACPolicy

    if getattr(SACPolicy, "_anvil_temperature_frozen_patched", False):
        return
    SACPolicy._anvil_temperature_frozen_patched = True

    _orig_compute_loss_temperature = SACPolicy.compute_loss_temperature

    def _compute_loss_temperature_frozen(self, observations, observation_features=None):
        loss = _orig_compute_loss_temperature(self, observations, observation_features)
        return loss * 0.0

    SACPolicy.compute_loss_temperature = _compute_loss_temperature_frozen


_episode_stats_state: dict = {"path": None}


_auto_stop_state = {"episodes": 0, "stop_requested": False}


def _max_episodes() -> int:
    """HILSERL_MAX_EPISODES=N (default 0 = off): stop the actor after N finished
    episodes, and the learner after N received episode summaries when it is
    not training (BC-only runs). Read on every call so tests can set it."""
    import os as _os

    try:
        return int(_os.environ.get("HILSERL_MAX_EPISODES", "0"))
    except ValueError:
        return 0


def _request_stop(who: str, n: int, delay_s: float = 0.0) -> None:
    """SIGINT ourselves = the same clean shutdown path as pressing Ctrl+C.

    Sets stop_requested immediately (before any delay/signal) -- see
    patch_leader_skip_reset_when_stopping() below, which reads it to avoid
    a real hardware crash: act_with_policy() calls online_env.reset() for
    the next (about-to-be-discarded) episode on the very next line after
    the Nth episode's transitions are pushed, with no check of its own for
    any stop condition. That reset does real ROS service calls (switch
    controller, home via arms_resetter) that take seconds -- exactly the
    window our own SIGINT's shutdown teardown lands in, guaranteed every
    time since there's no gap between "episode N done" and "reset() for
    N+1 starts". First hit 2026-09-29 (round-3 DAgger): the actor crashed
    with `arms_resetter did not report 'homed'` right after saving its
    10th and last episode, because the reset for a would-be 11th episode
    started and lost its ROS context mid-call.
    """
    _auto_stop_state["stop_requested"] = True
    import os as _os
    import signal as _signal
    import threading as _threading

    print(f"[{who}] reached HILSERL_MAX_EPISODES={n} -- stopping (same as Ctrl+C).", flush=True)
    if delay_s > 0:
        t = _threading.Timer(delay_s, lambda: _os.kill(_os.getpid(), _signal.SIGINT))
        t.daemon = True
        t.start()
    else:
        _os.kill(_os.getpid(), _signal.SIGINT)


def patch_learner_episode_stats_logging() -> None:
    """Official HIL-SERL docs (huggingface.co/docs/lerobot/en/hilserl) define
    a successful run as one where "the human has to intervene at the start
    but then reduces the amount of interventions as the policy improves",
    monitored via the intervention rate. lerobot's actor.py already computes
    exactly that per episode (`Intervention rate` = intervention steps /
    total steps, plus `Episodic reward`, `Episode intervention`,
    `Interaction step`) and sends it to the learner in every interaction
    message -- but the learner only forwards it to wandb, which this project
    keeps disabled, so it was silently dropped every session.

    Wrap process_interaction_message() (called once per received episode
    summary) to also append each message as one JSON line to
    <output_dir>/episode_stats.jsonl (output_dir is on the mounted volume,
    so it survives the container) and print a one-line summary live.
    train() is wrapped only to learn cfg.output_dir. Pure logging: the
    original function still runs first and its return value is unchanged.
    """
    try:
        import lerobot.rl.learner as learner_mod
    except Exception:  # noqa: BLE001
        return
    if getattr(learner_mod, "_anvil_episode_stats_patched", False):
        return
    learner_mod._anvil_episode_stats_patched = True

    import json as _json
    import os as _os

    orig_train = learner_mod.train

    def _train(cfg, *a, **kw):
        _episode_stats_state["path"] = _os.path.join(cfg.output_dir, "episode_stats.jsonl")
        # BC-only run (learning never starts): nothing to save, safe to stop on episode count
        _episode_stats_state["bc_only"] = cfg.policy.online_step_before_learning >= cfg.policy.online_steps
        return orig_train(cfg, *a, **kw)

    learner_mod.train = _train

    orig_msg = learner_mod.process_interaction_message

    def _process_interaction_message(message, interaction_step_shift, wandb_logger=None):
        result = orig_msg(message, interaction_step_shift, wandb_logger)
        try:
            path = _episode_stats_state["path"]
            if path is not None and isinstance(result, dict):
                _os.makedirs(_os.path.dirname(path), exist_ok=True)
                record = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in result.items()}
                with open(path, "a") as f:
                    f.write(_json.dumps(record) + "\n")
                print(
                    f"[EPISODE-STATS] step={record.get('Interaction step')} "
                    f"reward={record.get('Episodic reward')} "
                    f"intervention_rate={record.get('Intervention rate')}",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001 -- logging must never break training
            print(f"[EPISODE-STATS] could not record episode stats: {e}", flush=True)
        n = _max_episodes()
        if n > 0 and _episode_stats_state.get("bc_only"):
            _auto_stop_state["learner_episodes"] = _auto_stop_state.get("learner_episodes", 0) + 1
            if _auto_stop_state["learner_episodes"] == n:
                _request_stop("LEARNER", n, delay_s=3.0)
        return result

    learner_mod.process_interaction_message = _process_interaction_message


def patch_sac_critic_action_embedding() -> None:
    """2026-10-20 (gripper branch): confirmed directly (offline Q-variance
    diagnostic, re-run on this task's own real RL checkpoints, not just by
    analogy to the push-bottle lineage) that the critic is nearly
    action-blind -- perturbing the commanded action barely moves min-Q
    compared to perturbing the state. Root cause confirmed architecturally,
    not just empirically: CriticEnsemble.forward() (lerobot/policies/sac/
    modeling_sac.py) does `torch.cat([obs_enc, actions], dim=-1)` --
    actions are only 14-16 of ~1280 total critic-input dims (4 cameras x
    256-dim encoder slots each). A controlled offline test (perturb 14
    RANDOM encoder dims vs. the real 14 action dims, same count, same
    perturbation scale) produced the SAME tiny Q-variance either way,
    confirming this is pure dimension-count dilution, not the critic
    specifically discounting action -- weight/gradient inspection on the
    real checkpoint showed near-1:1 per-dimension magnitude between action
    and encoder columns.

    Fix: give the action its own learned embedding (Linear + SiLU,
    HILSERL_CRITIC_ACTION_EMBED_DIM-wide, default 64) before
    concatenation, instead of raw concatenation -- raises action's share
    of the critic's input from ~1% to ~5% of total dims. Already
    validated offline (scratch experiment, this session, not committed):
    on the push-bottle lineage this raised the Q-variance ratio
    post-training from 0.146% to 0.289% -- a real but modest improvement,
    NOT a complete fix (the dominant remaining limiter there was data
    starvation on a small, now-separately-fixed-elsewhere buffer, not
    this dimension issue alone). This patch is the first time this
    architecture change is implemented as a real, runtime-applied patch
    rather than a one-off scratch script.

    Opt-in via HILSERL_CRITIC_ACTION_EMBED_DIM (unset/0 = off, byte-for-
    byte original architecture and behavior -- same convention as
    HILSERL_ACTOR_WARMUP_STEPS). When off, none of this file's other
    behavior changes; this is the ONLY place that env var is read.

    Shares ONE action_embed module between critic_ensemble and
    critic_target (not a separately Polyak-tracked copy) -- mirrors the
    existing design already used for the observation encoder itself
    (_init_critics already passes the SAME encoder=self.encoder_critic to
    both CriticEnsemble instances, confirmed by reading that method
    directly). Attaching the same nn.Module instance as a plain attribute
    on both auto-registers it as a submodule of each (nn.Module.__setattr__),
    so optimizer_critic = Adam(critic_ensemble.parameters()) already
    includes it with zero further wiring, and update_target_networks()'s
    `zip(critic_target.parameters(), critic_ensemble.parameters(),
    strict=True)` sees the exact same tensors on both sides (Polyak-
    blending a tensor with itself is an exact no-op at any weight), so
    there is no target-lag concern and no strict=True count mismatch.

    Loading an existing checkpoint trained WITHOUT this patch (e.g.
    bc_grasp_place_v2) hits a real shape mismatch on the critic's first
    layer (input_dim changed) -- PyTorch's load_state_dict raises on a
    shape mismatch for a key present on both sides regardless of
    strict=False (that flag only tolerates missing/extra keys, confirmed
    by reading safetensors.torch.load_model() and torch's own
    load_state_dict directly). patch_sac_tolerates_shape_mismatch_on_load()
    below makes this a warn-and-skip instead of a crash -- everything
    that still matches shape (encoder, actor entirely, critic's later
    layers) loads normally; only the critic's first layer + the new
    action_embed start fresh/random.
    """
    import os as _os

    import torch as _torch
    import torch.nn as _nn

    from lerobot.policies.sac.modeling_sac import CriticEnsemble, CriticHead, SACPolicy

    embed_dim = int(_os.environ.get("HILSERL_CRITIC_ACTION_EMBED_DIM", "0"))
    if embed_dim <= 0:
        return

    if not getattr(CriticEnsemble, "_anvil_action_embed_forward_patched", False):
        CriticEnsemble._anvil_action_embed_forward_patched = True
        orig_forward = CriticEnsemble.forward

        def _forward(self, observations, actions, observation_features=None):
            embed = getattr(self, "action_embed", None)
            if embed is not None:
                actions = embed(actions)
            return orig_forward(self, observations, actions, observation_features=observation_features)

        CriticEnsemble.forward = _forward

    if getattr(SACPolicy, "_anvil_critic_action_embed_init_patched", False):
        return
    SACPolicy._anvil_critic_action_embed_init_patched = True

    from dataclasses import asdict as _asdict

    def _init_critics(self, continuous_action_dim):
        action_embed = _nn.Sequential(_nn.Linear(continuous_action_dim, embed_dim), _nn.SiLU())

        heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + embed_dim,
                **_asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_ensemble = CriticEnsemble(encoder=self.encoder_critic, ensemble=heads)
        self.critic_ensemble.action_embed = action_embed  # auto-registers as a submodule

        target_heads = [
            CriticHead(
                input_dim=self.encoder_critic.output_dim + embed_dim,
                **_asdict(self.config.critic_network_kwargs),
            )
            for _ in range(self.config.num_critics)
        ]
        self.critic_target = CriticEnsemble(encoder=self.encoder_critic, ensemble=target_heads)
        # Attach BEFORE load_state_dict, not after -- critic_ensemble's
        # state_dict() already has action_embed's keys (attached above),
        # so critic_target needs the same submodule present first or the
        # sync raises "Unexpected key(s): action_embed.0.weight/.bias".
        self.critic_target.action_embed = action_embed  # SAME object, not a copy -- see docstring
        self.critic_target.load_state_dict(self.critic_ensemble.state_dict())

        if self.config.use_torch_compile:
            self.critic_ensemble = _torch.compile(self.critic_ensemble)
            self.critic_target = _torch.compile(self.critic_target)

        if self.config.num_discrete_actions is not None:
            self._init_discrete_critics()

    SACPolicy._init_critics = _init_critics

    print(f"[LEARNER-DBG] critic action-embedding enabled (HILSERL_CRITIC_ACTION_EMBED_DIM={embed_dim}) "
          "-- action now gets its own learned embedding before concatenation with the encoder output.",
          flush=True)

    patch_sac_tolerates_shape_mismatch_on_load()


def patch_sac_tolerates_shape_mismatch_on_load() -> None:
    """Companion to patch_sac_critic_action_embedding() -- see that
    function's docstring for why a shape mismatch is EXPECTED when loading
    a checkpoint trained before the action-embedding change, and why
    strict=False alone doesn't help (PyTorch's load_state_dict raises on a
    shape mismatch for a key present on both sides regardless of that
    flag -- confirmed by reading safetensors.torch.load_model() and
    load_state_dict directly, not assumed).

    Fix: replace SACPolicy._load_as_safetensor (classmethod) with a
    version that reads the raw tensors (same safetensors.torch.load_file()
    call the stock path uses internally), filters out any key whose saved
    shape doesn't match the freshly-constructed model's own parameter
    shape, loads the rest via the model's normal load_state_dict(...,
    strict=False), and logs exactly which keys were skipped. Only called
    by patch_sac_critic_action_embedding() when that feature is actually
    on -- not registered standalone in apply_all_patches(), since it has
    no reason to exist when every shape already matches.
    """
    from lerobot.policies.sac.modeling_sac import SACPolicy

    if getattr(SACPolicy, "_anvil_shape_mismatch_tolerant_load_patched", False):
        return
    SACPolicy._anvil_shape_mismatch_tolerant_load_patched = True

    import packaging.version
    import safetensors
    from safetensors.torch import load_file

    @classmethod
    def _load_as_safetensor(cls, model, model_file, map_location, strict):
        kwargs = {}
        if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
            kwargs["device"] = map_location
        saved_state_dict = load_file(model_file, **kwargs)
        model_state_dict = model.state_dict()

        skipped = []
        for key in list(saved_state_dict.keys()):
            if key in model_state_dict and saved_state_dict[key].shape != model_state_dict[key].shape:
                skipped.append(
                    f"{key}: saved {tuple(saved_state_dict[key].shape)} != "
                    f"model {tuple(model_state_dict[key].shape)}"
                )
                del saved_state_dict[key]

        if skipped:
            print(
                f"[LEARNER-DBG] loading checkpoint with {len(skipped)} shape-mismatched key(s) "
                "skipped (expected when loading a pre-action-embedding checkpoint under "
                "HILSERL_CRITIC_ACTION_EMBED_DIM -- these start fresh/random instead):",
                flush=True,
            )
            for s in skipped:
                print(f"  {s}", flush=True)

        model.load_state_dict(saved_state_dict, strict=False)
        return model

    SACPolicy._load_as_safetensor = _load_as_safetensor


def patch_sac_actor_warmup() -> None:
    """First real-RL session (2026-09-29, BC v8 -> RL, 1500 steps,
    online_step_before_learning=10): the policy came back measurably WORSE
    -- intervention rate rose to ~0.54 (vs the ~0.39-0.50 BC/DAgger-only
    plateau) -- the very first time gradient updates were ever allowed to
    run for real. Directly ruled out the temperature freeze as the cause:
    the saved step-1500 checkpoint's log_alpha is bit-identical to
    temperature_init (-4.605170249938965 == log(0.01)), so
    patch_sac_freeze_temperature() held throughout.

    Matches this project's own much older offline diagnostic exactly
    (see the plan's "offline learning-health diagnostics" section): the
    critic is the only source of the actor's improvement signal
    (compute_loss_actor = temp*logpi - min_Q(s,pi(s))), Adam gives a
    full-size step regardless of how reliable that gradient is, and with
    online_step_before_learning=10 the actor starts moving after about
    one second of real data -- a critic that's had ~zero real calibration
    yet. The actor DOES move, just in a noisy, uninformed direction, drifting
    away from a BC start that was already decent. This was flagged as a
    follow-up back when A (deterministic inference) + B (frozen
    temperature) were first added ("do only if A+B aren't enough") --
    today's session is the first real evidence they aren't.

    Fix (ORIGINAL design, superseded below 2026-10-12 -- kept for history):
    let the critic train against real transitions for
    HILSERL_ACTOR_WARMUP_STEPS actor-update opportunities before the actor
    optimizer is allowed to take its first step, by skipping
    optimizers["actor"].step() entirely for that many calls.
    Critic/discrete_critic/temperature optimizers are completely untouched.

    Opt-in via HILSERL_ACTOR_WARMUP_STEPS (default 0 = off, i.e. today's
    exact prior behavior) rather than always-on -- this is a new, only
    offline-reasoned idea, never validated even in a fake-hardware run
    yet, and every other lever pulled during real-RL sessions in this
    project has been introduced as an explicit opt-in first.

    REAL GAP FOUND AND CLOSED, 2026-10-08 (bimanual push-bottle, two
    reproductions from a clean, just-reverted checkpoint): episode 1 was
    confident/BC-level (commanded delta from the real current state
    ~10-24 deg/step, measured directly from saved actor_trajectories
    .npz files); episode 2, immediately after only ~100 real optimization
    steps (well under HILSERL_ACTOR_WARMUP_STEPS=200), collapsed to
    near-zero commanded deltas (~0.4-1.0 deg/step) -- essentially a
    frozen/no-op policy -- despite the actor optimizer's own .step()
    being correctly skipped the whole time (confirmed: 100 <= 200).

    Root cause, confirmed directly from modeling_sac.py/learner.py
    source: SACConfig.shared_encoder defaults to True (not overridden by
    this task's config), and make_optimizers_and_scheduler()'s OWN
    docstring says exactly this -- "if the encoder is shared, its
    parameters are excluded from the actor's optimization process" --
    excluded from optimizer_actor, but NOT excluded from optimizer_critic
    (`torch.optim.Adam(params=policy.critic_ensemble.parameters(), ...)`,
    and critic_ensemble owns/wraps the SAME encoder object as the actor
    when shared -- policy.encoder_critic is policy.actor.encoder).
    optimizers["critic"].step() is NEVER warmup-gated (runs from
    optimization_step 1) -- so the shared representation the actor's
    forward pass depends on was drifting, unconstrained, under a
    critic that had barely any real calibration yet, the entire time
    the actor's OWN head was correctly frozen. Freezing only
    optimizers["actor"] protects half the pipeline that actually
    determines the actor's output; the other half (the shared encoder)
    was never covered at all.

    Fix: when policy.shared_encoder is True, ALSO toggle requires_grad on
    policy.encoder_critic's parameters in lockstep with the same warmup
    counter -- frozen (no critic-driven drift) for the first
    HILSERL_ACTOR_WARMUP_STEPS calls, unfrozen at the exact same moment
    the ramp below completes. Safe (no "does not require grad" crash
    risk, unlike the temperature-loss fix): these are LEAF parameter
    tensors, not a whole loss graph -- loss_critic still has other
    (critic-head) parameters requiring grad, so .backward() still
    succeeds normally; the frozen encoder params simply never accumulate
    a .grad, and Adam silently skips params with grad=None.

    SECOND REAL GAP FOUND AND CLOSED, 2026-10-12 (same task, same
    checkpoint lineage, AFTER the shared-encoder fix above was already
    active and confirmed): resumed from 4200 with HILSERL_ACTOR_WARMUP_
    STEPS=200; episodes 1-2 stayed confident, episode 3 collapsed to
    near-zero commanded deltas again -- and checkpoint-timestamp
    cross-referencing placed that collapse within seconds of
    checkpoint 4400, i.e. almost exactly the moment the ORIGINAL
    skip-based warmup would have ended (resumed at 4200, warmup=200).
    Confirmed directly from torch's own Adam source
    (torch/optim/adam.py): `step_size = lr / (1 - beta1**step)` --
    LARGEST at step=1, shrinking toward the nominal `lr` as `step`
    grows. Skipping optimizers["actor"].step() entirely for the whole
    warm-up window means Adam's own internal step counter for every
    actor parameter stays pinned at 0 the ENTIRE time, no matter how
    large HILSERL_ACTOR_WARMUP_STEPS is set -- so the first unskipped
    call was always Adam's maximally-aggressive cold-start step, not a
    calibrated one. Raising the warmup_steps value alone cannot fix
    this -- it only delays WHEN the cold-start spike happens, matching
    exactly what was observed (immediate collapse before the shared-
    encoder fix; collapse delayed to right after the warm-up boundary
    once that leak was closed).

    Fix: replace the skip with a linear learning-rate RAMP (near-zero
    -> base_lr over HILSERL_ACTOR_WARMUP_STEPS calls), calling
    orig_step() on EVERY call rather than skipping it. Adam's internal
    step counter and moment estimates now build up normally against
    real (just heavily down-scaled) gradients throughout the ramp, so
    bias-correction has already mostly washed out by the time the ramp
    reaches 1.0 -- there is no instant full-LR step at any point, cold
    or otherwise. This is standard LR warmup, not a new/novel
    mechanism. Not yet validated on real hardware -- the 2026-10-08
    shared-encoder fix took one real-hardware attempt to find its own
    edge case; treat this the same way (opt-in, same env var, watch
    the next real session closely).
    """
    try:
        import lerobot.rl.learner as learner_mod
    except Exception:  # noqa: BLE001
        return
    if getattr(learner_mod, "_anvil_actor_warmup_patched", False):
        return
    learner_mod._anvil_actor_warmup_patched = True

    import os as _os

    warmup_steps = int(_os.environ.get("HILSERL_ACTOR_WARMUP_STEPS", "0"))
    if warmup_steps <= 0:
        return

    orig_make = learner_mod.make_optimizers_and_scheduler

    def _make(cfg, policy):
        optimizers, lr_scheduler = orig_make(cfg, policy)
        actor_opt = optimizers.get("actor")
        if actor_opt is not None:
            orig_step = actor_opt.step
            state = {"n": 0}
            base_lrs = [g["lr"] for g in actor_opt.param_groups]

            shared_encoder_params = None
            if getattr(policy, "shared_encoder", False):
                shared_encoder_params = list(policy.encoder_critic.parameters())
                for p in shared_encoder_params:
                    p.requires_grad_(False)
                print(
                    f"[LEARNER-DBG] actor warm-up: ALSO freezing {len(shared_encoder_params)} "
                    "shared-encoder parameters for the warm-up window -- optimizers['critic'] "
                    "would otherwise keep updating them every step regardless of the actor "
                    "optimizer freeze (see patch_sac_actor_warmup()'s docstring).",
                    flush=True,
                )

            def _step(*a, **kw):
                state["n"] += 1
                if state["n"] <= warmup_steps:
                    # Linear LR ramp (near-zero -> base_lr), NOT a skipped
                    # .step() call -- confirmed, 2026-10-12, via direct read
                    # of torch's own Adam source: step_size = lr /
                    # (1 - beta1**t), which is LARGEST at t=1 and shrinks as
                    # t grows. The original "skip orig_step() entirely for N
                    # calls" design left Adam's internal step counter pinned
                    # at 0 for the whole warm-up window, REGARDLESS of how
                    # large N was -- so the first unskipped call was always
                    # Adam's maximally-aggressive cold-start step, not a
                    # calibrated one. Confirmed on real hardware: with
                    # warm-up=200 and the shared-encoder leak already fixed,
                    # the policy still collapsed almost exactly at the
                    # warm-up boundary (episode 3, right after checkpoint
                    # 4400 -- 200 real optimization steps after resuming
                    # from 4200) -- raising warmup_steps alone would only
                    # have delayed this, not prevented it, since Adam's
                    # counter never moves during a skip either way. Calling
                    # orig_step() every time (never skipping) lets Adam's
                    # own step counter and moment estimates build up
                    # normally on real, just heavily down-scaled, gradients,
                    # so bias-correction has already mostly washed out by
                    # the time the ramp reaches 1.0 -- no sudden transition.
                    ramp = state["n"] / warmup_steps
                    for group, base_lr in zip(actor_opt.param_groups, base_lrs):
                        group["lr"] = base_lr * ramp
                    if state["n"] == warmup_steps:
                        for group, base_lr in zip(actor_opt.param_groups, base_lrs):
                            group["lr"] = base_lr
                        if shared_encoder_params is not None:
                            for p in shared_encoder_params:
                                p.requires_grad_(True)
                        print(
                            f"[LEARNER-DBG] actor warm-up ramp complete after {warmup_steps} "
                            f"steps -- actor LR restored to {base_lrs}, shared-encoder "
                            "parameters (if any) unfrozen too.",
                            flush=True,
                        )
                return orig_step(*a, **kw)

            actor_opt.step = _step
        return optimizers, lr_scheduler

    learner_mod.make_optimizers_and_scheduler = _make


def patch_robot_env_step_keeps_intervention_flag() -> None:
    """Found 2026-09-24 while adding per-episode intervention-rate logging:
    the rate read 0.0 in every episode even though the operator was
    intervening heavily. Root cause (lerobot 0.5.1, not this project's code):
    `RobotEnv.step()` unconditionally returns `{IS_INTERVENTION: False}` as
    its `info`, and `step_env_and_process_transition()` then does
    `new_info = <info the intervention processor set>; new_info.update(info)`
    -- the env's hard-coded False overwrites the True that
    InterventionActionProcessorStep had just recorded. So actor.py's
    `intervention_info.get(IS_INTERVENTION)` is always False: episode
    intervention counts/rate are always 0, and learner.py's offline-buffer
    growth for interventions (`transition["complementary_info"]
    .get(IS_INTERVENTION)`) can never fire. (The ACTIONS sent to the learner
    are unaffected -- they come from complementary_data["teleop_action"],
    set before the overwrite.)

    Fix: drop the key from the env's returned info so `update()` cannot
    clobber the processor's value. Every reader of the key uses
    `.get(IS_INTERVENTION, False)`, so an absent key still reads False when
    the operator is not intervening.
    """
    from lerobot.rl.gym_manipulator import RobotEnv
    from lerobot.teleoperators.utils import TeleopEvents

    if getattr(RobotEnv, "_anvil_step_flag_patched", False):
        return
    RobotEnv._anvil_step_flag_patched = True

    orig_step = RobotEnv.step

    def _step(self, action):
        obs, reward, terminated, truncated, info = orig_step(self, action)
        info = dict(info)
        info.pop(TeleopEvents.IS_INTERVENTION, None)
        return obs, reward, terminated, truncated, info

    RobotEnv.step = _step


def patch_step_env_squeezes_batched_action() -> None:
    """Real fake-hardware finding, joint-space actor/learner smoke test
    (2026-09-30): `step_env_and_process_transition()`
    (lerobot.rl.gym_manipulator, used unchanged by actor.py) hands
    `processed_action` straight to `env.step()`, which does
    `{f"{key}.pos": action[i] for i, key in enumerate(self.robot.bus.motors.keys())}`
    -- this assumes a flat, unbatched action. For the original ee_delta
    config, that's true by the time the action reaches here: the EE-delta/
    IK chain's last step (RobotActionToPolicyActionProcessorStep) happens
    to squeeze its own output. For a joint-space config
    (inverse_kinematics: null, see train_rl_push_bottle_jointspace.yaml),
    action_processor skips that whole chain by design -- the policy's raw
    select_action() output (SAC always returns a batched (1, N) tensor)
    reaches env.step() unmodified, crashing with
    `IndexError: index 1 is out of bounds for dimension 0 with size 1`.
    This is the exact squeeze already applied manually in
    run_joint_policy_eval.py's loop (`action.squeeze(0) if action.dim() > 1
    else action`) -- applying it here via patch covers the actor/learner
    path too, which reuses lerobot's own step_env_and_process_transition()
    rather than a loop we control directly.

    Safe for the ee_delta path too: squeezing an action that's already
    flat (dim() == 1) is a no-op, so this patch changes nothing for any
    config that isn't joint-space.

    Also moves the action to CPU before handing it to env.step(): same
    root cause as the squeeze above -- AnvilOpenArmFollower.send_action()
    needs plain Python floats (it does np.array([action[f"{j}.pos"] ...]),
    matching run_joint_policy_eval.py's own manual loop, which already does
    action.detach().cpu() for this exact reason), but the ee_delta IK
    chain's own final steps happen to already return a CPU tensor, so this
    was never needed on that path before. Confirmed on real fake-hardware:
    `TypeError: can't convert cuda:0 device type tensor to numpy`.

    ACTOR_TRACE_OUT=/path/to/trace.jsonl (2026-09-30, after the actor path's
    own liveness-watchdog trip at step 54 -- same still-unresolved "commands
    sent, arm didn't respond" pattern seen repeatedly on the ACT eval path,
    now confirmed on this path too): appends one JSON line per tick
    (wall-clock time, target/before/after joint degrees) so the NEXT
    occurrence gives precise per-tick data directly, mirroring
    run_joint_policy_eval.py's JOINT_EVAL_TRACE_OUT -- that trace file is
    what originally proved "the model kept computing correctly while our
    own observation froze" for the ACT case. Appends incrementally (flushed
    every write), not just at process exit, so a crash mid-run still
    preserves every tick up to the one that failed. Wrapped in its own
    try/except so a tracing bug can never break the real control loop.
    """
    from lerobot.rl import gym_manipulator
    from lerobot.types import TransitionKey
    from lerobot.rl.gym_manipulator import create_transition

    if getattr(gym_manipulator, "_anvil_squeeze_action_patched", False):
        return
    gym_manipulator._anvil_squeeze_action_patched = True

    import os as _os

    _trace_path = _os.environ.get("ACTOR_TRACE_OUT")
    _trace_state = {"file": None}

    def _trace_tick(before: dict, target, after: dict) -> None:
        if not _trace_path:
            return
        try:
            import json as _json
            import time as _time

            if _trace_state["file"] is None:
                _trace_state["file"] = open(_trace_path, "a")
            joint_names = [f"follower_r_joint{i}" for i in range(1, 8)]
            row = {
                "time": _time.time(),
                "target": [float(target[i]) for i in range(min(len(target), 7))],
                "before": [float(before.get(f"{j}.pos", float("nan"))) for j in joint_names],
                "after": [float(after.get(f"{j}.pos", float("nan"))) for j in joint_names],
            }
            _trace_state["file"].write(_json.dumps(row) + "\n")
            _trace_state["file"].flush()
        except Exception:  # noqa: BLE001 -- tracing must never break the control loop
            pass

    def _step_env_and_process_transition(env, transition, action, env_processor, action_processor):
        transition[TransitionKey.ACTION] = action
        transition[TransitionKey.OBSERVATION] = (
            env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
        )
        before_obs = transition[TransitionKey.OBSERVATION]
        processed_action_transition = action_processor(transition)
        processed_action = processed_action_transition[TransitionKey.ACTION]
        if hasattr(processed_action, "device") and processed_action.device.type != "cpu":
            processed_action = processed_action.detach().cpu()
        if hasattr(processed_action, "dim") and processed_action.dim() > 1:
            processed_action = processed_action.squeeze(0)

        # Real bimanual-recording finding (2026-10-02): control_loop()'s
        # non-intervening tick always sends a HARDCODED 3-element (4 with
        # gripper) "neutral_action" -- a stock lerobot EE-delta-era
        # assumption (lerobot/rl/gym_manipulator.py's own control_loop()).
        # For ee_delta configs this is harmless: InverseKinematicsRLStep
        # converts it into a real-joint-count action regardless of
        # intervention state. For joint-space configs
        # (processor.inverse_kinematics: null), that whole IK chain is
        # skipped by design, so the 3-element placeholder reaches env.step()
        # unchanged whenever IS_INTERVENTION is False -- crashing with
        # `IndexError: index 3 is out of bounds for dimension 0 with size 3`
        # the moment a robot has more than 3 joints (hit on the very first
        # bimanual attempt at 14 joints; every single-arm session so far
        # happened to have the human intervening from the first tick, which
        # is why this was never triggered at 7 joints either).
        #
        # Fix: if the action is shorter than the robot's real joint count,
        # replace it with the robot's OWN CURRENT joint positions (a
        # genuine "hold still"), not zeros -- joint-space actions are
        # absolute next-tick positions, not deltas, so padding/replacing
        # with 0.0 would command a real, large, unsafe motion toward
        # whatever "0 degrees for every joint" happens to be, not a no-op.
        real_joint_names = list(env.robot.bus.motors.keys())
        if len(processed_action) < len(real_joint_names):
            current_obs = env.robot.get_observation()
            processed_action = [current_obs[f"{j}.pos"] for j in real_joint_names]

        obs, reward, terminated, truncated, info = env.step(processed_action)
        _trace_tick(before_obs, processed_action, obs)

        reward = reward + processed_action_transition[TransitionKey.REWARD]
        terminated = terminated or processed_action_transition[TransitionKey.DONE]
        truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
        complementary_data = processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
        new_info = processed_action_transition[TransitionKey.INFO].copy()
        new_info.update(info)

        # Real bimanual-recording finding (2026-10-02), second half of the
        # neutral_action problem above: InterventionActionProcessorStep
        # (lerobot/processor/hil_processor.py) only overwrites
        # complementary_data["teleop_action"] with the REAL action while
        # `is_intervention` is True -- on a non-intervening tick it leaves
        # that key as whatever action came in (the 3-element neutral_action
        # placeholder), regardless of the fix above, which only affects
        # what's actually SENT to the robot. control_loop()'s own recording
        # logic reads `complementary_data.get("teleop_action", ...)`
        # specifically for the saved dataset frame -- confirmed on real
        # hardware: `ValueError: The feature 'action' of shape '(3,)' does
        # not have the expected shape '(14,)'` the first time a
        # non-intervening tick's frame was actually saved. Same fix,
        # applied to the recorded value too: reuse the already-corrected
        # `processed_action` (guaranteed real-joint-count by this point)
        # whenever the recorded teleop_action is undersized.
        recorded_teleop_action = complementary_data.get("teleop_action")
        if recorded_teleop_action is not None and len(recorded_teleop_action) < len(real_joint_names):
            import torch as _torch_recorded

            complementary_data["teleop_action"] = _torch_recorded.as_tensor(processed_action, dtype=_torch_recorded.float32)

        # Real bimanual-recording finding (2026-10-02): env_processor's own
        # batch-action step (to_batch_action_processor) requires a
        # torch.Tensor ("Action should be a PolicyAction type (tensor), but
        # got <class 'list'>") -- but for joint-space configs,
        # InterventionActionProcessorStep's ndarray branch (used whenever
        # actually intervening, confirmed via a real pressed-space episode
        # tonight) produces action.tolist(), a plain Python list, and
        # nothing downstream re-wraps it as a tensor the way the ee_delta/IK
        # chain happens to. Never caught before because this is the FIRST
        # time hilserl_gym_manipulator has ever actually RECORDED in
        # joint-space mode for either arm count -- single-arm's own
        # joint-space training data (demos_v2_jointspace) was relabeled from
        # an old ee_delta recording, never recorded fresh this way. Convert
        # right before building the transition that env_processor will see
        # -- env.step() above already received the untouched list/array
        # form and that path is unaffected.
        import torch as _torch

        action_for_transition = (
            processed_action
            if isinstance(processed_action, _torch.Tensor)
            else _torch.as_tensor(processed_action, dtype=_torch.float32)
        )

        new_transition = create_transition(
            observation=obs, action=action_for_transition, reward=reward, done=terminated,
            truncated=truncated, info=new_info, complementary_data=complementary_data,
        )
        return env_processor(new_transition)

    gym_manipulator.step_env_and_process_transition = _step_env_and_process_transition
    import lerobot.rl.actor as actor_mod

    actor_mod.step_env_and_process_transition = _step_env_and_process_transition


def apply_all_patches() -> None:
    """Single entry point for every runtime patch in this file. All of
    lerobot.rl.gym_manipulator's make_robot_env()/make_processors() pipeline
    is reused UNCHANGED by lerobot.rl.actor and lerobot.rl.learner (confirmed
    via direct import: actor.py imports create_transition/make_processors/
    make_robot_env/step_env_and_process_transition from gym_manipulator.py
    itself, not a separate implementation) -- every one of these patches
    applies equally to the actor/learner path, not just the recording path
    hilserl_gym_manipulator uses. This matters more there, not less: the
    actor publishes a freshly-initialized (untrained) policy's raw output
    straight to the real robot from its very first control step, with no
    exploration gating -- exactly the kind of out-of-distribution input
    that the EEBoundsAndSafety crash-vs-clamp bug is most likely to hit.
    Call this from every entrypoint that reaches that pipeline, not just
    hilserl_gym_manipulator's own.
    """
    patch_logging_includes_tracebacks()
    patch_map_delta_action_for_gripperless_robots()
    patch_ee_bounds_and_safety_to_clamp_not_crash()
    patch_inverse_kinematics_to_always_reseed_from_real_joints()
    patch_robot_env_reset_to_keep_publishing()
    patch_robot_env_reset_to_home_via_arms_resetter()
    patch_leader_skip_reset_when_stopping()
    patch_multi_leader_skip_reset_when_stopping()
    patch_train_pipeline_config_validate_for_optional_dataset()
    patch_learner_progress_visibility()
    patch_learner_checkpoint_saves_dataset_atomically()
    patch_learner_stop_at_online_steps()
    patch_learner_resume_offline_buffer_from_source()
    patch_learner_resume_online_buffer_tolerates_broken_snapshot()
    patch_learner_resume_preserves_cli_online_steps_and_save_freq()
    patch_learner_pauses_training_when_actor_idle()
    patch_sac_deterministic_inference()
    patch_sac_loads_action_bounds_on_from_pretrained()
    patch_sac_normalizes_replay_buffer_actions_for_critic_loss()
    patch_sac_checkpoints_carry_action_bounds()
    patch_sac_freeze_temperature()
    patch_sac_critic_action_embedding()
    patch_sac_actor_warmup()
    patch_learner_episode_stats_logging()
    patch_robot_env_step_keeps_intervention_flag()
    patch_intervention_flag_in_complementary_info()
    patch_step_env_squeezes_batched_action()


def patch_learner_checkpoint_saves_dataset_atomically() -> None:
    """2026-10-19/20 (gripper branch): `save_training_checkpoint()`
    (lerobot/rl/learner.py:674-758) saves the online (and offline) replay
    buffer to `<output_dir>/dataset` via:

        if os.path.exists(dataset_dir) and os.path.isdir(dataset_dir):
            shutil.rmtree(dataset_dir)                      # destroys the last GOOD snapshot FIRST
        replay_buffer.to_lerobot_dataset(repo_id=..., fps=fps, root=dataset_dir)  # slow; only valid once finalize() runs

    `to_lerobot_dataset()` (lerobot/rl/buffer.py:509-612) already correctly
    uses the buffer's own ground-truth done/truncated flags to call
    save_episode() at the right boundaries, with finalize() at the end
    flushing meta/episodes/ -- confirmed directly against real `dataset/`
    snapshots already in this project's outputs/: when a save completes,
    the round-trip is correct and loads cleanly. The bug is CRASH-SAFETY,
    not episode-boundary logic: the rmtree deletes the only good copy
    BEFORE the slow rewrite, so any interruption between that line and the
    eventual finalize() (Ctrl+C, a hardware fault, OOM, SIGKILL -- all
    routine in this project's real sessions) leaves dataset_dir gone or
    half-written, with no way back. This is exactly why
    patch_learner_resume_online_buffer_tolerates_broken_snapshot() (below)
    was written as a catch-and-reset-to-empty safety net -- it treats the
    symptom (a FileNotFoundError on resume), not the cause. That patch's
    own docstring explains why reconstructing episode boundaries from raw
    frames at LOAD time was correctly rejected as unverifiable -- but
    that's not needed here: the ground-truth-done-flags path already
    exists in to_lerobot_dataset() and already works once given the
    chance to finish.

    Fix: write the new snapshot to a sibling temp dir first (never
    touching the live dataset_dir while writing), then swap it in with
    two fast os.rename() calls -- each one atomic on the same filesystem
    (this project's outputs/ is one mount, no cross-device rename risk).
    If the process dies before both renames, the old dataset_dir is never
    touched at all -- only transitions since the LAST SUCCESSFUL
    checkpoint are ever at risk, never the whole history. This is a full
    reimplementation of save_training_checkpoint (not a before/after
    wrap) because the risky rmtree-then-write logic is inline in that
    function's body, not a separately wrappable sub-step -- steps 1-4
    (checkpoint dir, policy/optimizer save, training-state, "last"
    symlink) are unchanged, calling the exact same already-imported
    helpers the original uses so this tracks lerobot's own helpers rather
    than duplicating their logic.

    MUST be registered before patch_learner_stop_at_online_steps() (next
    in this file): that patch captures `orig_save =
    learner_mod.save_training_checkpoint` at the moment IT runs and wraps
    it -- this patch must already be in place first so the stop-patch
    wraps OUR atomic version, not the raw original.
    """
    try:
        import lerobot.rl.learner as learner_mod
    except Exception:  # noqa: BLE001
        return
    if getattr(learner_mod, "_anvil_atomic_checkpoint_save_patched", False):
        return
    learner_mod._anvil_atomic_checkpoint_save_patched = True

    import logging
    import os as _os
    import shutil as _shutil

    import torch as _torch

    def _swap_in(tmp_dir: str, live_dir: str) -> None:
        prev_dir = live_dir + ".prev"
        if _os.path.exists(prev_dir):
            _shutil.rmtree(prev_dir)  # leftover from a prior interrupted swap
        if _os.path.exists(live_dir):
            _os.rename(live_dir, prev_dir)
        _os.rename(tmp_dir, live_dir)
        if _os.path.exists(prev_dir):
            _shutil.rmtree(prev_dir)

    def _save_training_checkpoint(
        cfg,
        optimization_step,
        online_steps,
        interaction_message,
        policy,
        optimizers,
        replay_buffer,
        offline_replay_buffer=None,
        dataset_repo_id=None,
        fps=30,
    ) -> None:
        logging.info(f"Checkpoint policy after step {optimization_step}")
        interaction_step = interaction_message["Interaction step"] if interaction_message is not None else 0

        checkpoint_dir = learner_mod.get_step_checkpoint_dir(cfg.output_dir, online_steps, optimization_step)

        learner_mod.save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            step=optimization_step,
            cfg=cfg,
            policy=policy,
            optimizer=optimizers,
            scheduler=None,
        )

        training_state_dir = _os.path.join(checkpoint_dir, learner_mod.TRAINING_STATE_DIR)
        _os.makedirs(training_state_dir, exist_ok=True)
        training_state = {"step": optimization_step, "interaction_step": interaction_step}
        _torch.save(training_state, _os.path.join(training_state_dir, "training_state.pt"))

        learner_mod.update_last_checkpoint(checkpoint_dir)

        # Atomic swap: write to a sibling tmp dir, only touch the live
        # dataset_dir with two fast renames once the slow write succeeds.
        dataset_dir = _os.path.join(cfg.output_dir, "dataset")
        tmp_dir = dataset_dir + ".tmp"
        if _os.path.exists(tmp_dir):
            _shutil.rmtree(tmp_dir)  # leftover from a prior interrupted write
        repo_id_buffer_save = cfg.env.task if dataset_repo_id is None else dataset_repo_id
        replay_buffer.to_lerobot_dataset(repo_id=repo_id_buffer_save, fps=fps, root=tmp_dir)
        _swap_in(tmp_dir, dataset_dir)

        if offline_replay_buffer is not None:
            dataset_offline_dir = _os.path.join(cfg.output_dir, "dataset_offline")
            tmp_offline_dir = dataset_offline_dir + ".tmp"
            if _os.path.exists(tmp_offline_dir):
                _shutil.rmtree(tmp_offline_dir)
            offline_replay_buffer.to_lerobot_dataset(cfg.dataset.repo_id, fps=fps, root=tmp_offline_dir)
            _swap_in(tmp_offline_dir, dataset_offline_dir)

        logging.info("Resume training")

    learner_mod.save_training_checkpoint = _save_training_checkpoint


def patch_learner_stop_at_online_steps() -> None:
    """lerobot's learner loop is `while True` with no exit on online_steps: it
    keeps re-training on the same finite buffer forever after the budget
    (seen 2026-09-21: 4000+ steps for a 1000-step budget). Once the checkpoint
    at optimization_step == online_steps is written, raise SIGINT so lerobot's
    normal shutdown path runs (same as pressing Ctrl+C, after saving)."""
    try:
        import lerobot.rl.learner as learner_mod
    except Exception:  # noqa: BLE001
        return
    if getattr(learner_mod, "_anvil_stop_patched", False):
        return
    learner_mod._anvil_stop_patched = True

    import os as _os
    import signal as _signal

    orig_save = learner_mod.save_training_checkpoint

    def _save(*a, **kw):
        result = orig_save(*a, **kw)
        step, budget = kw.get("optimization_step"), kw.get("online_steps")
        if step is not None and budget is not None and step == budget:
            print(f"[LEARNER-DBG] reached optimization step {step} == online_steps; "
                  "checkpoint saved, shutting down.", flush=True)
            _os.kill(_os.getpid(), _signal.SIGINT)
        return result

    learner_mod.save_training_checkpoint = _save


def patch_learner_resume_offline_buffer_from_source() -> None:
    """On --resume=true, initialize_offline_replay_buffer() tries to reload
    the offline buffer from the snapshot save_training_checkpoint() wrote to
    <output_dir>/dataset_offline, via a raw LeRobotDataset(repo_id=...,
    root=dataset_offline_path) construction. That snapshot's meta/ directory
    is missing meta/episodes/ (confirmed directly: present in the real
    source dataset at data/datasets/push_bottle_bimanual_demos_combined/meta/
    episodes/, absent from the saved dataset_offline snapshot) -- a genuine
    save/load mismatch in lerobot's own checkpoint code, not anything in our
    patches. LeRobotDatasetMetadata._load_metadata() -> load_episodes() then
    raises FileNotFoundError, which LeRobotDatasetMetadata.__init__ catches
    and silently falls back to a Hub download using cfg.dataset.repo_id --
    a cosmetic label that has never actually been pushed to the real HF Hub
    in this project, so that fallback hard-crashes with a 401
    RepositoryNotFoundError (hit directly, 2026-10-07, blocking every
    --resume actor connection: the learner's gRPC server never finishes
    starting).

    The offline buffer is the static original demo set -- it never changes
    across a session, so there is nothing to actually "resume" about it.
    Fix: on resume, build it the exact same way a fresh run already does
    successfully (make_dataset(cfg), confirmed by every prior successful
    session this project has run) instead of trying to reload the broken
    snapshot. The ONLINE buffer's own resume path (initialize_replay_buffer(),
    reloading <output_dir>/dataset) hits the exact same bug -- confirmed
    directly, 2026-10-07, same FileNotFoundError -> 401 crash, just on the
    online dataset snapshot instead of the offline one. See
    patch_learner_resume_online_buffer_tolerates_broken_snapshot() below for
    that one -- it needs a different fix, since there is no equivalent
    "real source" to fall back to for genuinely-accumulated online data."""
    try:
        import lerobot.rl.learner as learner_mod
    except Exception:  # noqa: BLE001
        return
    if getattr(learner_mod, "_anvil_offline_resume_patched", False):
        return
    learner_mod._anvil_offline_resume_patched = True

    def _initialize_offline_replay_buffer(cfg, device, storage_device):
        import logging

        from lerobot.rl.buffer import ReplayBuffer

        logging.info("make_dataset offline buffer (resume-safe: source dataset, not the saved snapshot)")
        offline_dataset = learner_mod.make_dataset(cfg)
        return ReplayBuffer.from_lerobot_dataset(
            lerobot_dataset=offline_dataset,
            capacity=cfg.policy.offline_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True,
        )

    learner_mod.initialize_offline_replay_buffer = _initialize_offline_replay_buffer


def patch_learner_resume_online_buffer_tolerates_broken_snapshot() -> None:
    """Same underlying bug as patch_learner_resume_offline_buffer_from_source()
    (see its docstring) -- LeRobotDataset(repo_id=cfg.dataset.repo_id,
    root=<output_dir>/dataset) hits the identical FileNotFoundError ->
    broken-Hub-fallback -> 401 crash, confirmed directly 2026-10-07, blocking
    every --resume learner startup that had ever accumulated online data.

    Unlike the offline buffer, there is no equivalent real source to fall
    back to here -- the online buffer holds genuinely-accumulated live
    transitions that exist nowhere else. Reconstructing the missing
    meta/episodes/ parquet ourselves was considered and rejected: it would
    require inferring episode boundaries from the raw frame data with no
    way to verify correctness, risking silently-wrong training data, which
    is a worse failure mode than losing some online history outright.

    Fix: try the real reload; if it fails for ANY reason (this bug, or
    anything else), fall back to a fresh empty ReplayBuffer -- the exact
    same construction the non-resume branch already uses. This only
    affects the online buffer's CONTENTS. The thing that actually matters
    across a resume -- policy/critic/optimizer state and the step counter
    -- comes from load_training_state()'s own, completely separate path
    (raw safetensors/torch files, no LeRobotDataset involved at all), so
    it is unaffected either way: this patch can only cost some accumulated
    online sample efficiency, never the trained policy itself."""
    try:
        import lerobot.rl.learner as learner_mod
    except Exception:  # noqa: BLE001
        return
    if getattr(learner_mod, "_anvil_online_resume_patched", False):
        return
    learner_mod._anvil_online_resume_patched = True

    orig_initialize = learner_mod.initialize_replay_buffer

    def _initialize_replay_buffer(cfg, device, storage_device):
        import logging

        from lerobot.rl.buffer import ReplayBuffer

        if not cfg.resume:
            return orig_initialize(cfg, device, storage_device)
        try:
            return orig_initialize(cfg, device, storage_device)
        except Exception as e:  # noqa: BLE001 -- see docstring: fall back, never crash
            logging.warning(
                f"[LEARNER] Could not reload the saved online replay buffer on resume "
                f"({e!r}) -- starting the online buffer empty instead. Policy/critic/"
                f"optimizer state is unaffected (loaded separately); only accumulated "
                f"online sample history is lost for this resume."
            )
            return ReplayBuffer(
                capacity=cfg.policy.online_buffer_capacity,
                device=device,
                state_keys=cfg.policy.input_features.keys(),
                storage_device=storage_device,
                optimize_memory=True,
            )

    learner_mod.initialize_replay_buffer = _initialize_replay_buffer


def save_actor_trajectory(transitions) -> str | None:
    """Write one episode's executed trajectory to disk (2026-09-24).

    BC-only test runs never train, so the learner never exports its replay
    buffer and everything the actor did was lost. This saves, per episode, as
    an uncompressed .npz: joint state (degrees, as observed), EVERY camera
    image the policy saw (float16, 3x128x128 each -- see below), the
    EXECUTED action (the human's when intervening, the policy's otherwise),
    reward, done, and a per-step `is_intervention` flag (carried in
    complementary_info["discrete_penalty"], a slot this gripper-less task
    never uses -- see patch_intervention_flag_in_complementary_info). That
    is exactly what a DAgger-style retrain needs: the human's corrections at
    the states the policy actually reached. Directory: $HILSERL_TRAJ_DIR
    (default /workspace/outputs/actor_trajectories -- mount ~/.../outputs
    there on the ACTOR container to keep it). Logging only: any failure is
    swallowed.

    All-cameras fix, 2026-10-13: this originally hardcoded a single
    "observation.images.wrist_r" key (true single-arm-era design, never
    updated for bimanual's 4 cameras) -- confirmed the hard way, running a
    DAgger-style BC retrain against the 110 episodes saved since crashed
    with KeyError: 'observation.images.wrist_l', because _Arrays/_KEYS in
    bc_pretrain.py (already generalized to N cameras for the base dataset
    path) needs every camera the policy's encoder expects, not just one.
    Now saves every "observation.images.*" key present in the transition's
    state dict generically, each under its own npz array
    (image_<camera_name>, stripping the prefix) -- matches _Frames'
    already-generic approach on the base-dataset side, just applied here
    too. _human_segments()/_Arrays in bc_pretrain.py were updated to read
    these multi-camera arrays back, with a documented fallback for the 110
    already-saved single-camera ("image") files from before this fix.
    """
    import os as _os
    import time as _time

    try:
        import numpy as _np
        import torch as _torch

        def _vec(x):
            return _torch.as_tensor(x).detach().cpu().float().reshape(-1).numpy()

        def _img(x):
            x = _torch.as_tensor(x).detach().cpu().float()
            return x.reshape(-1, *x.shape[-3:])[0].numpy().astype(_np.float16)

        out_dir = _os.environ.get("HILSERL_TRAJ_DIR", "/workspace/outputs/actor_trajectories")
        _os.makedirs(out_dir, exist_ok=True)
        state = _np.stack([_vec(t["state"]["observation.state"]) for t in transitions])
        action = _np.stack([_vec(t["action"]) for t in transitions])
        reward = _np.array([float(_vec(t["reward"])[0]) for t in transitions], dtype=_np.float32)
        done = _np.array([float(_vec(t["done"])[0]) for t in transitions], dtype=_np.float32)
        flag = _np.array(
            [float(_vec((t.get("complementary_info") or {}).get("discrete_penalty", 0.0))[0]) for t in transitions],
            dtype=_np.float32,
        )
        arrays = dict(state_deg=state, action=action, reward=reward, done=done, is_intervention=flag)
        img_prefix = "observation.images."
        cam_keys = [k for k in transitions[0]["state"] if k.startswith(img_prefix)]
        for img_key in cam_keys:
            cam_name = img_key[len(img_prefix) :]
            if all(img_key in t["state"] for t in transitions):
                arrays[f"image_{cam_name}"] = _np.stack([_img(t["state"][img_key]) for t in transitions])
        path = _os.path.join(out_dir, f"ep_{_time.strftime('%Y%m%d_%H%M%S')}_{len(transitions)}steps.npz")
        _np.savez(path, **arrays)
        return path
    except Exception as e:  # noqa: BLE001 -- must never break the control loop
        print(f"[ACTOR-DBG] could not save trajectory: {e}", flush=True)
        return None


def patch_intervention_flag_in_complementary_info() -> None:
    """Per-step intervention flag for the replay buffer / saved trajectories.

    lerobot's actor builds each Transition's complementary_info from
    `complementary_data["discrete_penalty"]` only; nothing carries whether the
    executed action was the human's, so past sessions' data could not be split
    into human corrections vs policy actions (and learner.py's offline-buffer
    growth for interventions, which reads IS_INTERVENTION there, never fired).
    `discrete_penalty` is consumed solely by the discrete-gripper critic
    (modeling_sac.compute_loss_discrete_critic, only when num_discrete_actions
    is set) -- unused by this gripper-less task -- so it is free to carry
    float(is_intervention). InterventionActionProcessorStep already reads the
    true flag from its INPUT info; we just copy it into complementary_data.
    """
    from lerobot.processor.hil_processor import InterventionActionProcessorStep
    from lerobot.teleoperators.utils import TeleopEvents
    from lerobot.types import TransitionKey

    if getattr(InterventionActionProcessorStep, "_anvil_flag_patched", False):
        return
    InterventionActionProcessorStep._anvil_flag_patched = True

    orig_call = InterventionActionProcessorStep.__call__

    def _call(self, transition):
        info_in = transition.get(TransitionKey.INFO) or {}
        is_intervention = bool(info_in.get(TeleopEvents.IS_INTERVENTION, False))
        out = orig_call(self, transition)
        data = out.get(TransitionKey.COMPLEMENTARY_DATA) or {}
        data["discrete_penalty"] = float(is_intervention)
        out[TransitionKey.COMPLEMENTARY_DATA] = data
        return out

    InterventionActionProcessorStep.__call__ = _call


def patch_learner_progress_visibility() -> None:
    """Learner-side gRPC/training threads' logging.info() output never showed
    up in the terminal (2026-09-21: two 1000-step runs looked identical to a
    stuck learner). Print (flush) transition arrivals, parameter pushes and
    training progress so a silent learner can be told apart from a working one.
    No-op if lerobot.rl.learner isn't importable (e.g. actor process)."""
    try:
        import lerobot.rl.learner as learner_mod
    except Exception:  # noqa: BLE001
        return
    if getattr(learner_mod, "_anvil_progress_patched", False):
        return
    learner_mod._anvil_progress_patched = True

    import time as _time

    orig_process = learner_mod.process_transitions
    state = {"n": 0, "t": 0.0}

    def _process(transition_queue, replay_buffer, *a, **kw):
        before = len(replay_buffer)
        orig_process(transition_queue, replay_buffer, *a, **kw)
        after = len(replay_buffer)
        if after != before:
            print(f"[LEARNER-DBG] replay buffer {before} -> {after}", flush=True)
        now = _time.time()
        if now - state["t"] > 15.0:
            state["t"] = now
            print(f"[LEARNER-DBG] alive: online buffer={after}", flush=True)

    learner_mod.process_transitions = _process

    orig_push = learner_mod.push_actor_policy_to_queue

    def _push(parameters_queue, policy):
        state["n"] += 1
        temp = getattr(policy, "temperature", None)
        print(f"[LEARNER-DBG] parameters pushed to queue (#{state['n']}), temperature={temp}", flush=True)
        return orig_push(parameters_queue, policy)

    learner_mod.push_actor_policy_to_queue = _push

    # Which gRPC streams did the actor actually open? (their own logging.info
    # lines are invisible in the terminal -- see docstring above)
    try:
        from lerobot.rl.learner_service import LearnerService

        for name in ("StreamParameters", "SendTransitions", "SendInteractions", "Ready"):
            orig = getattr(LearnerService, name, None)
            if orig is None:
                continue

            def _wrap(orig=orig, name=name):
                def inner(self, *a, **kw):
                    print(f"[LEARNER-DBG] gRPC {name} opened by actor", flush=True)
                    return orig(self, *a, **kw)

                return inner

            setattr(LearnerService, name, _wrap())
    except Exception as e:  # noqa: BLE001
        print(f"[LEARNER-DBG] could not wrap LearnerService: {e}", flush=True)

    # Actor side is patched from the actor entrypoint via the same function
    # (learner import succeeds there too), so instrument transition sends here.
    try:
        import lerobot.rl.actor as actor_mod

        orig_push_t = actor_mod.push_transitions_to_transport_queue

        def _push_t(transitions, transitions_queue):
            print(f"[ACTOR-DBG] queueing {len(transitions)} transitions for the learner", flush=True)
            save_actor_trajectory(transitions)
            result = orig_push_t(transitions, transitions_queue)
            n = _max_episodes()
            if n > 0:
                _auto_stop_state["episodes"] += 1
                print(f"[ACTOR-DBG] episode {_auto_stop_state['episodes']}/{n} saved", flush=True)
                if _auto_stop_state["episodes"] == n:
                    _request_stop("ACTOR", n)  # the episode-end homing still completes first
            return result

        actor_mod.push_transitions_to_transport_queue = _push_t
    except Exception:  # noqa: BLE001
        pass


def patch_train_pipeline_config_validate_for_optional_dataset() -> None:
    """Stage 6 finding: `TrainRLServerPipelineConfig.dataset` is explicitly
    typed `DatasetConfig | None = None` (lerobot/configs/train.py:216) --
    the RL pipeline config deliberately makes the offline dataset optional,
    since HIL-SERL can train purely online. But `validate()`
    (`TrainPipelineConfig.validate()`, inherited unchanged -- the RL
    subclass never overrides it) does `if isinstance(self.dataset.repo_id,
    list):` unconditionally, with no `self.dataset is not None` guard.
    Confirmed directly: `cfg.validate()` raises `AttributeError: 'NoneType'
    object has no attribute 'repo_id'` the moment `dataset:` is omitted from
    the config -- and both `actor.py` and `learner.py` call `cfg.validate()`
    unconditionally at startup, so this blocks the simplest ("pure online,
    no dataset") smoke-test path entirely, not just an edge case.

    Fix: wrap `validate()` so that when `self.dataset is None`, it
    temporarily stands in a minimal duck-typed placeholder (only `.repo_id`
    and `.root` are ever read by the rest of the method) for the duration
    of the original call, then restores `self.dataset` to `None`
    afterward -- every other line of the original method's logic runs
    completely unmodified, and the rest of actor.py/learner.py (which
    checks `cfg.dataset is not None` to decide whether to load an offline
    dataset at all) still correctly sees `None` and skips it, exactly the
    intended pure-online behavior.
    """
    from lerobot.configs.train import TrainPipelineConfig

    if getattr(TrainPipelineConfig, "_anvil_optional_dataset_validate_patch_applied", False):
        return

    import types

    _orig_validate = TrainPipelineConfig.validate

    def _patched_validate(self):
        if self.dataset is None:
            self.dataset = types.SimpleNamespace(repo_id=None, root=None)
            try:
                _orig_validate(self)
            finally:
                self.dataset = None
        else:
            _orig_validate(self)

    TrainPipelineConfig.validate = _patched_validate
    TrainPipelineConfig._anvil_optional_dataset_validate_patch_applied = True


def patch_learner_resume_preserves_cli_online_steps_and_save_freq() -> None:
    """handle_resume_logic() (lerobot/rl/learner.py) unconditionally REPLACES
    the whole cfg object with TrainRLServerPipelineConfig.from_pretrained()
    of the checkpoint's own saved train_config.json -- i.e. whatever
    --policy.online_steps/--save_freq this specific invocation was launched
    with on the CLI is silently discarded, replaced by whatever those
    fields happened to be the very FIRST time this run was ever started
    fresh (confirmed directly, 2026-10-07: a run's checkpoints kept landing
    on multiples of 250, the old save_freq default, despite --save-freq 100
    being passed on every resumed launch since).

    This matters beyond cosmetics: it's also what let the "learner trains
    forever" safety patch (patch_learner_stop_at_online_steps(), which
    fires on optimization_step == online_steps) go silently inert across a
    resume -- the stale online_steps value is already behind the resumed
    step counter the instant training resumes, so that equality can never
    hold again. This is a direct contributor to the runaway-to-4000+-steps
    incident from earlier today (see patch_learner_resume_online_buffer_
    tolerates_broken_snapshot()'s docstring for the rest of that story).

    Fix: capture THIS invocation's actually-requested online_steps/
    save_freq before calling the original (unconditionally, so the
    non-resume FileExistsError-prevention branch is untouched), then
    re-apply them onto whatever handle_resume_logic() returns -- only when
    this call was actually a resume (a fresh run's own online_steps/
    save_freq were never going to be touched by this function anyway).
    Everything ELSE handle_resume_logic() restores from the checkpoint
    (pretrained_path, env/robot/policy architecture, dataset paths, ...)
    is left alone -- only these two fields are meant to be a per-invocation
    choice rather than a frozen historical artifact."""
    try:
        import lerobot.rl.learner as learner_mod
    except Exception:  # noqa: BLE001
        return
    if getattr(learner_mod, "_anvil_resume_cli_steps_patched", False):
        return
    learner_mod._anvil_resume_cli_steps_patched = True

    orig_handle_resume_logic = learner_mod.handle_resume_logic

    def _handle_resume_logic(cfg):
        was_resuming = cfg.resume
        requested_online_steps = cfg.policy.online_steps
        requested_save_freq = cfg.save_freq
        new_cfg = orig_handle_resume_logic(cfg)
        if was_resuming:
            new_cfg.policy.online_steps = requested_online_steps
            new_cfg.save_freq = requested_save_freq
        return new_cfg

    learner_mod.handle_resume_logic = _handle_resume_logic


def patch_learner_pauses_training_when_actor_idle() -> None:
    """lerobot's learner main loop (`while True` in add_actor_information_and_train())
    never pauses training based on whether the ACTOR is still around -- it
    only gates on `len(replay_buffer) < online_step_before_learning`
    (lerobot/rl/learner.py), a one-way check that's satisfied forever once
    enough transitions have EVER arrived. Confirmed directly, 2026-10-07:
    one real completed episode (47 transitions, from an early session) was
    enough to let the learner train for 550+ further optimization steps
    across three SUBSEQUENT actor sessions that each crashed before
    completing a single episode -- i.e. that contributed nothing new. The
    policy kept changing the whole time, grounded in nothing but one old
    episode plus the fixed offline set, repeatedly, while real sessions
    kept failing faster and more consistently each time (same deterministic
    policy + same post-rehome starting state -> an increasingly-overfit
    policy reproducing roughly the same trajectory each attempt is a
    plausible contributor, though not a proven root cause of the hardware
    fault itself).

    Fix: wrap whatever initialize_replay_buffer() currently returns (so
    this composes correctly regardless of patch_learner_resume_online_
    buffer_tolerates_broken_snapshot() -- must be registered AFTER that one
    in apply_all_patches()) in a thin proxy that tracks the wall-clock time
    of the last .add() call and makes __len__ report 0 (not "pause" --
    genuinely report empty) whenever nothing has been added in the last
    STALE_AFTER_S seconds. The existing `len(replay_buffer) <
    online_step_before_learning: continue` check then does the actual
    pausing for free, with zero changes to the loop itself: real data
    already in the buffer is never discarded, sampling/training just can't
    proceed on it until something fresh arrives again.

    STALE_AFTER_S=90 is deliberately generous relative to this task's own
    per-episode timing (processor.reset.control_time_s=20s plus homing/
    intervention overhead, confirmed in the bimanual task config) -- long
    enough that normal multi-episode operation, where fresh transitions
    only arrive at EPISODE boundaries (confirmed separately: actor.py only
    calls push_transitions_to_transport_queue() inside `if done or
    truncated:`, never per-tick), is never falsely paused, while still
    catching real multi-minute gaps between actor attempts (waiting on a
    devbox re-home, reading a message, etc.)."""
    try:
        import lerobot.rl.learner as learner_mod
    except Exception:  # noqa: BLE001
        return
    if getattr(learner_mod, "_anvil_idle_pause_patched", False):
        return
    learner_mod._anvil_idle_pause_patched = True

    import time as _time

    STALE_AFTER_S = 90.0

    class _StalenessAwareReplayBuffer:
        def __init__(self, real):
            self._anvil_real = real
            # Deliberately already-stale, not _time.time() -- confirmed as a
            # real bug 2026-10-08: initializing to "now" gave every fresh
            # learner start/resume up to STALE_AFTER_S seconds of training
            # on the just-loaded (potentially hours-old) buffer before
            # staleness ever kicked in, even with zero actor connected --
            # exactly the pathology this patch exists to prevent, just
            # triggered by a restart instead of a long idle gap. Three
            # learner restarts in one morning (while iterating on this
            # launcher script) each bought ~1-2 real minutes of unprotected
            # retraining on 6+ hour old data with no fresh correction,
            # visibly degrading the policy. "Nothing has been added in the
            # last STALE_AFTER_S seconds" is literally true the instant this
            # object is constructed -- no .add() has happened yet in this
            # process's lifetime -- so __len__ must report 0 from the very
            # first call, not just after the first real idle period.
            self._anvil_last_add = _time.time() - STALE_AFTER_S - 1.0

        def add(self, *args, **kwargs):
            self._anvil_last_add = _time.time()
            return self._anvil_real.add(*args, **kwargs)

        def __len__(self):
            if _time.time() - self._anvil_last_add > STALE_AFTER_S:
                return 0
            return len(self._anvil_real)

        def __getattr__(self, name):
            return getattr(self._anvil_real, name)

    orig_initialize_replay_buffer = learner_mod.initialize_replay_buffer

    def _initialize_replay_buffer(cfg, device, storage_device):
        real_buffer = orig_initialize_replay_buffer(cfg, device, storage_device)
        return _StalenessAwareReplayBuffer(real_buffer)

    learner_mod.initialize_replay_buffer = _initialize_replay_buffer


def patch_logging_includes_tracebacks() -> None:
    """lerobot's init_logging() (lerobot/utils/utils.py) installs a
    custom_format() that only ever calls record.getMessage() -- it never
    renders record.exc_info, so any logging.exception()/log(..., exc_info=True)
    call silently loses its traceback, even though the LogRecord carries one.

    This is the reason two separate real, session-ending actor crashes
    (2026-10-08, global steps ~107 and ~289, both logged as a bare
    "ERROR ... _channel.py:284 Exception iterating requests!" with no
    further detail) could not be root-caused from the log alone: grpc's own
    _consume_request_iterator() (grpc/_channel.py) wraps the request
    generator in `except Exception: ... _LOGGER.exception(details)` --
    logging.exception() always attaches the real underlying exception (the
    one that actually broke mid-stream, e.g. a tensor pickling/serialization
    error, a CUDA error, a broken pipe) via exc_info -- but init_logging()'s
    formatter throws that traceback away before it ever reaches the log
    file. Confirmed directly: neither occurrence had any traceback text in
    the log, despite grpc's own code unconditionally calling the
    exc_info-attaching logging method right before raising.

    Fix: wrap init_logging itself (in every module that already imported it
    by name -- lerobot.rl.actor and lerobot.rl.learner both do `from
    lerobot.utils.utils import init_logging` at module load time, before
    apply_all_patches() ever runs, so patching lerobot.utils.utils.init_logging
    alone would not affect their already-bound references) so that, after
    the real init_logging() finishes (re)installing handlers, every
    handler's formatter also appends the traceback when a record carries
    exc_info -- exactly what Python's own default logging.Formatter does,
    without changing the existing single-line format for the overwhelming
    majority of records that have none. Each handler's formatter is wrapped
    only once (guarded per-formatter-instance, not just per-module), since
    init_logging() clears and recreates handlers on every call (every forked
    actor subprocess calls it again with its own log file)."""
    try:
        import lerobot.utils.utils as utils_mod
    except Exception:  # noqa: BLE001
        return
    if getattr(utils_mod, "_anvil_logging_tracebacks_patched", False):
        return
    utils_mod._anvil_logging_tracebacks_patched = True

    import logging as _logging

    orig_init_logging = utils_mod.init_logging
    _tb_formatter = _logging.Formatter()

    def _init_logging_with_tracebacks(*args, **kwargs):
        orig_init_logging(*args, **kwargs)
        for handler in _logging.getLogger().handlers:
            fmt = handler.formatter
            if fmt is None or getattr(fmt, "_anvil_tb_wrapped", False):
                continue
            base_format = fmt.format

            def _format(record, _base=base_format):
                text = _base(record)
                if record.exc_info:
                    text = f"{text}\n{_tb_formatter.formatException(record.exc_info)}"
                return text

            fmt.format = _format
            fmt._anvil_tb_wrapped = True

    utils_mod.init_logging = _init_logging_with_tracebacks

    for module_name in ("lerobot.rl.actor", "lerobot.rl.learner"):
        try:
            consumer_mod = __import__(module_name, fromlist=["init_logging"])
        except Exception:  # noqa: BLE001
            continue
        if getattr(consumer_mod, "init_logging", None) is orig_init_logging:
            consumer_mod.init_logging = _init_logging_with_tracebacks
