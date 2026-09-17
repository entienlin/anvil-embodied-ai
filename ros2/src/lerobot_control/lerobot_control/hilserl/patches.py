"""Runtime patches for lerobot 0.5.1 gaps affecting gripper-less HIL-SERL
robots (this project's push-bottle task has none, per the plan's task
decision).

Both patches target the same class of bug: a processor step unconditionally
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

    Fix: replace the blocking sleep with a loop that keeps republishing a
    "hold at current position" command every 100ms for the same total
    duration, so the command topic is never actually silent. Doesn't
    change reset()'s real behavior or timing (still waits reset_time_s
    before returning; still does the optional reset_follower_position()
    first if reset_pose is configured) -- only removes the silent gap.
    """
    from lerobot.rl.gym_manipulator import RobotEnv

    if getattr(RobotEnv, "_anvil_reset_keepalive_patch_applied", False):
        return

    import time as _time

    import gymnasium as gym
    import numpy as np
    from lerobot.rl.gym_manipulator import log_say, reset_follower_position
    from lerobot.teleoperators.utils import TeleopEvents
    from lerobot.utils.robot_utils import precise_sleep

    def _patched_reset(self, *, seed=None, options=None):
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

    RobotEnv.reset = _patched_reset
    RobotEnv._anvil_reset_keepalive_patch_applied = True
