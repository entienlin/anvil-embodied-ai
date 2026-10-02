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
  2026-09-29  patch_sac_actor_warmup
      First real-RL session: gradient updates measurably WORSENED the
      policy immediately (intervention rate ~0.39-0.50 -> ~0.54) -- an
      undertrained critic giving the actor bad early guidance. Opt-in
      (HILSERL_ACTOR_WARMUP_STEPS) critic-only warm-up window.
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

    Fix: let the critic train against real transitions for
    HILSERL_ACTOR_WARMUP_STEPS actor-update opportunities before the actor
    optimizer is allowed to take its first step. compute_loss_actor/
    .backward() still run every time (cheap, keeps the graph/optimizer
    state exercised) -- only optimizers["actor"].step() is skipped, so no
    autograd/requires_grad surgery and no risk of the same
    "loss becomes non-differentiable" crash class the temperature fix had
    to avoid. Critic/discrete_critic/temperature optimizers are completely
    untouched.

    Opt-in via HILSERL_ACTOR_WARMUP_STEPS (default 0 = off, i.e. today's
    exact prior behavior) rather than always-on -- this is a new, only
    offline-reasoned idea, never validated even in a fake-hardware run
    yet, and every other lever pulled during real-RL sessions in this
    project has been introduced as an explicit opt-in first.
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

            def _step(*a, **kw):
                state["n"] += 1
                if state["n"] <= warmup_steps:
                    if state["n"] == warmup_steps:
                        print(
                            f"[LEARNER-DBG] actor warm-up done after {warmup_steps} "
                            "critic-only updates -- actor optimizer now stepping.",
                            flush=True,
                        )
                    return None  # critic-only warm-up: skip this actor update
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

        new_transition = create_transition(
            observation=obs, action=processed_action, reward=reward, done=terminated,
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
    patch_map_delta_action_for_gripperless_robots()
    patch_ee_bounds_and_safety_to_clamp_not_crash()
    patch_inverse_kinematics_to_always_reseed_from_real_joints()
    patch_robot_env_reset_to_keep_publishing()
    patch_robot_env_reset_to_home_via_arms_resetter()
    patch_leader_skip_reset_when_stopping()
    patch_multi_leader_skip_reset_when_stopping()
    patch_train_pipeline_config_validate_for_optional_dataset()
    patch_learner_progress_visibility()
    patch_learner_stop_at_online_steps()
    patch_sac_deterministic_inference()
    patch_sac_loads_action_bounds_on_from_pretrained()
    patch_sac_normalizes_replay_buffer_actions_for_critic_loss()
    patch_sac_checkpoints_carry_action_bounds()
    patch_sac_freeze_temperature()
    patch_sac_actor_warmup()
    patch_learner_episode_stats_logging()
    patch_robot_env_step_keeps_intervention_flag()
    patch_intervention_flag_in_complementary_info()
    patch_step_env_squeezes_batched_action()


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


def save_actor_trajectory(transitions) -> str | None:
    """Write one episode's executed trajectory to disk (2026-09-24).

    BC-only test runs never train, so the learner never exports its replay
    buffer and everything the actor did was lost. This saves, per episode, as
    an uncompressed .npz: joint state (degrees, as observed), the wrist image
    the policy saw (float16, 3x128x128), the EXECUTED action (the human's when
    intervening, the policy's otherwise), reward, done, and a per-step
    `is_intervention` flag (carried in complementary_info["discrete_penalty"],
    a slot this gripper-less task never uses -- see
    patch_intervention_flag_in_complementary_info). That is exactly what a
    DAgger-style retrain needs: the human's corrections at the states the
    policy actually reached. Directory: $HILSERL_TRAJ_DIR (default
    /workspace/outputs/actor_trajectories -- mount ~/.../outputs there on the
    ACTOR container to keep it). Logging only: any failure is swallowed.
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
        img_key = "observation.images.wrist_r"
        if all(img_key in t["state"] for t in transitions):
            arrays["image"] = _np.stack([_img(t["state"][img_key]) for t in transitions])
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
