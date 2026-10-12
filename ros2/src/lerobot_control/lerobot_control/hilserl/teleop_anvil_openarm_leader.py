"""ROS2-native LeRobot Teleoperator for the Anvil OpenArm leader_r arm.

Provides HIL-SERL's human-intervention action: reads the leader arm's live
pose off /joint_states (no CAN access needed -- exposed for free whenever
the devbox is in a leader-capable control mode, per the plan's devbox
findings), and converts FRAME-TO-FRAME LEADER MOVEMENT into the same
per-axis-multiplier delta convention InterventionActionProcessorStep reads
via get_action() (see the plan's "Critical undocumented findings" -- keys
"delta_x"/"delta_y"/"delta_z", each a multiplier on end_effector_step_sizes,
not a raw physical delta).

IMPORTANT: this tracks the leader's own frame-to-frame CHANGE in EE
position, not "leader position minus follower position" -- the two arms
are mounted at different fixed base locations (confirmed via Stage 0's
kinematics check: exactly 1.0m apart in y at all-zeros), so an absolute
leader-vs-follower comparison would incorrectly include that fixed offset
as motion. Tracking relative leader movement since the last tick avoids
this entirely and matches how leader-follower teleop is meant to work.
"""

import atexit
import threading
import time
import traceback
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState

from lerobot.model.kinematics import RobotKinematics
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

from .arms_resetter_client import ArmsResetterClient
from .keyboard_intervention import KeyboardIntervention
from .switch_service_client import SwitchServiceClient

# Module-level registry so the reset patch (which only has access to
# RobotEnv's self.robot, not self.teleop -- RobotEnv.__init__ never
# stores the teleoperator, gym_manipulator.py wires it in separately via
# make_processors()) can reach the one AnvilOpenArmLeader instance in this
# process to trigger the post-homing baseline handoff. Set at the end of
# connect(), cleared at the start of disconnect() -- see both for why.
#
# Bimanual gap (2026-10-01): this used to be a single `_registered_leader`
# object, not a dict. The moment a second AnvilOpenArmLeader starts
# existing in the same process (one per arm, for bimanual control), its
# connect() call silently overwrote the first arm's registration -- the
# second-to-connect arm's leader would be the only one _check_liveness()
# and the reset patch could ever see, and the first arm's own watchdog
# would lose its false-positive-during-intervention suppression entirely.
# Keyed by ros_prefix so every arm's leader coexists safely. See
# get_registered_leader()/is_any_leader_intervening() below for how
# callers (both the single-arm ones already in this codebase, and the new
# multi-arm composite in teleop_anvil_openarm_multi_leader.py) use this.
_registered_leaders: dict[str, "AnvilOpenArmLeader"] = {}

# Reserved registry key a composite (multi-arm) leader registers itself
# under, in ADDITION to its sub-leaders each registering under their own
# ros_prefix. See get_registered_leader()'s no-arg branch: a registered
# composite is always preferred there, since it (not either individual
# sub-leader) is the one object whose perform_episode_reset_and_wait()
# correctly orchestrates a reset across every arm at once.
MULTI_LEADER_REGISTRY_KEY = "__multi__"


def get_registered_leader(ros_prefix: str | None = None) -> "AnvilOpenArmLeader | None":
    """Look up a registered AnvilOpenArmLeader (or composite multi-leader).

    ros_prefix given: exact lookup by arm -- used by callers that know
    which specific arm's leader they need.

    ros_prefix=None (the original, single-arm calling convention used
    unchanged by robot_anvil_openarm.py's reset bookkeeping and
    patches.py's patch_robot_env_reset_to_home_via_arms_resetter()):
    prefers a registered composite multi-leader if one exists (it is the
    single object meant to represent "the leader" when there's more than
    one arm), otherwise returns the one registered leader when exactly
    one is registered (byte-for-byte the old singleton's behavior for
    single-arm setups), otherwise None. This makes both existing
    no-argument call sites work unmodified for single-arm, bimanual, and
    any other arm count the composite leader supports.
    """
    if ros_prefix is not None:
        return _registered_leaders.get(ros_prefix)
    composite = _registered_leaders.get(MULTI_LEADER_REGISTRY_KEY)
    if composite is not None:
        return composite
    if len(_registered_leaders) == 1:
        return next(iter(_registered_leaders.values()))
    return None


def is_any_leader_intervening() -> bool:
    """True if any registered leader (any arm, plus a registered composite
    if present) currently reports active-or-just-ended intervention.

    Used by each follower's liveness watchdog (_check_liveness() in
    robot_anvil_openarm.py) to suppress the false-positive "arm looks
    dead" trip while a human is actually driving it via the leader rig.
    Intervention is a single shared human decision covering every arm at
    once (one person, one leader rig, both hands at once -- see
    teleop_anvil_openarm_multi_leader.py's module docstring), so for
    exactly one registered leader this reduces to the original per-leader
    check; for several, checking any of them is correct because they all
    ultimately read the same shared KeyboardIntervention instance.
    """
    return any(leader._was_intervening or leader._keyboard.is_intervention for leader in _registered_leaders.values())


@TeleoperatorConfig.register_subclass("anvil_openarm_leader")
@dataclass
class AnvilOpenArmLeaderConfig(TeleoperatorConfig):
    ros_prefix: str = "leader_r"
    joint_state_topic: str = "/joint_states"
    num_joints: int = 7
    urdf_path: str = "configs/hilserl/urdf/openarm_teleop_kinematics.urdf"
    target_frame_name: str = "leader_r_hand_tcp"
    # Must match the follower's InverseKinematicsConfig.end_effector_step_sizes
    # in the task's env YAML -- this is what converts our per-axis multiplier
    # back into the same physical step size EEReferenceAndDelta applies.
    step_sizes: dict = field(default_factory=lambda: {"x": 0.02, "y": 0.02, "z": 0.02})
    obs_timeout_s: float = 2.0
    # "ee_delta" (default, unchanged): the original FK-based per-axis
    # multiplier convention -- requires cfg.processor.inverse_kinematics to
    # be set so make_processors() adds the EE-delta/IK chain.
    # "joint_space": get_action() returns the leader's own current joint
    # positions directly (a plain ndarray, degrees, follower joint order) --
    # requires cfg.processor.inverse_kinematics: null so make_processors()
    # skips the EE-delta/IK chain entirely (it's hardcoded to a 3-dim
    # ee.x/y/z action and would crash/misbehave on a 7-dim joint action).
    # See get_action()'s docstring for why this needs no kinematics at all.
    action_mode: str = "ee_delta"
    # Switch-service wiring -- see switch_service_client.py. None of these
    # need to change unless the devbox names things differently than planned.
    policy_controller: str = "follower_r_forward_position_controller"
    # NOT arm-scoped in the actual name -- confirmed via devbox source read
    # (controller_yaml_generator.py/arm_nodes.py): the controller instance
    # is always literally "teleop_controller" regardless of how many/which
    # arms are configured. Arm scoping only shows up inside its own
    # ros__parameters (leader_r_joints/follower_r_joints etc), never in the
    # instance name itself.
    teleop_controller: str = "teleop_controller"
    follower_command_topic: str = "/follower_r_forward_position_controller/commands"
    follower_joint_prefix: str = "follower_r"
    # Mirrors AnvilOpenArmFollowerConfig.use_gripper on the teleop side --
    # requires action_mode="joint_space" (validated in __init__): "ee_delta"
    # mode's fixed {"delta_x","delta_y","delta_z"} dict has no gripper slot
    # in this project's design (see _get_action_joint_space()'s docstring).
    use_gripper: bool = False


class AnvilOpenArmLeader(Teleoperator):
    config_class = AnvilOpenArmLeaderConfig
    name = "anvil_openarm_leader"

    def __init__(self, config: AnvilOpenArmLeaderConfig, shared_keyboard: KeyboardIntervention | None = None):
        """shared_keyboard: for bimanual/multi-arm use only (see
        teleop_anvil_openarm_multi_leader.py). Passing an already-started
        KeyboardIntervention instance makes this instance treat keyboard
        lifecycle (start()/stop()) as someone else's responsibility -- two
        AnvilOpenArmLeader instances both independently calling
        KeyboardIntervention.start() in one process would spawn two
        threads racing for every keystroke off the same stdin fd (see
        keyboard_intervention.py's module docstring). Default None
        preserves the original single-arm behavior exactly: this instance
        creates and fully owns its own private KeyboardIntervention.
        """
        super().__init__(config)
        self.config = config
        self._owns_keyboard_lifecycle = shared_keyboard is None
        self.joint_names = [f"{config.ros_prefix}_joint{i}" for i in range(1, config.num_joints + 1)]
        # +1 gripper/finger joint: SwitchServiceClient's bumpless-transfer
        # pre-publish sends these joints' current positions on the same
        # command topic AnvilOpenArmFollower.send_action() uses, which real
        # hardware confirmed (Stage 5) needs 8 values (7 arm + gripper), not
        # 7 -- see robot_anvil_openarm.py's gripper_joint_name docstring for
        # the full story (a 7-value publish gets the controller
        # safety-deactivated by controller_manager).
        self.follower_joint_names = [
            f"{config.follower_joint_prefix}_joint{i}" for i in range(1, config.num_joints + 1)
        ] + [f"{config.follower_joint_prefix}_finger_joint1"]

        if config.use_gripper and config.action_mode != "joint_space":
            raise ValueError(
                "AnvilOpenArmLeaderConfig: use_gripper=True requires action_mode='joint_space' "
                f"(got {config.action_mode!r}) -- see _get_action_joint_space()'s docstring for why "
                "'ee_delta' has no gripper slot."
            )
        # Leader's OWN finger joint (e.g. "leader_r_finger_joint1") -- a
        # real, physically independent trigger/handle on the leader rig
        # itself, confirmed via configs/hilserl/urdf/openarm_teleop_kinematics.urdf:
        # leader_r_finger_joint1 is a real prismatic joint with the same
        # 0.0-0.05m limit as follower_r_finger_joint1. This is NOT
        # self.follower_joint_names' own finger entry above (that one is
        # the FOLLOWER's gripper joint, read only for SwitchServiceClient's
        # bumpless-transfer pre-publish) -- this is the human's actual
        # gripper open/close signal, read off the leader hardware during
        # intervention.
        self.leader_gripper_joint_name = f"{config.ros_prefix}_finger_joint1"

        self._kinematics = RobotKinematics(
            urdf_path=config.urdf_path,
            target_frame_name=config.target_frame_name,
            joint_names=self.joint_names,
        )

        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._spin_stop = threading.Event()
        self._is_connected = False
        self._latest_joint_state: JointState | None = None

        self._keyboard = shared_keyboard if shared_keyboard is not None else KeyboardIntervention()
        self._switch_client: SwitchServiceClient | None = None
        self._arms_resetter: ArmsResetterClient | None = None
        self._was_intervening = False
        self._leader_ee_ref: np.ndarray | None = None  # reset each time intervention (re)starts
        self._pending_switch_retries = 0  # see get_action()'s switch-failure handling
        # True from the moment a reset hands the arm to teleop_controller until that same
        # reset confirms "homed" and hands it back -- see perform_episode_reset_and_wait()'s
        # docstring and disconnect()'s guard below for why this exists.
        self._reset_in_progress_unsafe = False

    # -- Teleoperator ABC ------------------------------------------------

    @property
    def action_features(self) -> dict:
        # Real hardware finding (Stage 5): gym_manipulator.py's control_loop()
        # (record mode) uses teleop_device.action_features directly as a
        # single LeRobotDataset feature spec ({"dtype","shape","names"}),
        # not the generic per-field {field_name: type} convention some other
        # Teleoperator ABC implementations use (e.g. KeyboardRoverTeleop).
        # Confirmed the required shape by reading stock
        # KeyboardEndEffectorTeleop.action_features -- it returns exactly
        # this shape (same delta_x/delta_y/delta_z, no gripper here either),
        # and it's the only stock teleop this task's config path was ever
        # cross-checked against (Stage 3). action_features is read in
        # exactly one place in this whole pipeline (gym_manipulator.py's
        # control_loop), so this format is safe to use unconditionally.
        if self.config.action_mode == "joint_space":
            shape = self.config.num_joints + (1 if self.config.use_gripper else 0)
            return {"dtype": "float32", "shape": (shape,), "names": None}
        return {
            "dtype": "float32",
            "shape": (3,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2},
        }

    @property
    def feedback_features(self) -> dict:
        return {}  # no haptic feedback device

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        self._node = Node(f"anvil_hilserl_leader_{self.config.ros_prefix}")
        self._node.create_subscription(JointState, self.config.joint_state_topic, self._on_joint_state, 10)

        # See the matching comment in robot_anvil_openarm.py's connect():
        # rclpy.spin_once(node) with no explicit executor falls back to a
        # single process-wide global executor, which collides the instant
        # both this class's and AnvilOpenArmFollower's spin threads run
        # together in one process (confirmed on real hardware at Stage 5:
        # "RuntimeError: Executor is already spinning").
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        self._switch_client = SwitchServiceClient(
            self._node,
            policy_controller=self.config.policy_controller,
            teleop_controller=self.config.teleop_controller,
            command_topic=self.config.follower_command_topic,
            joint_state_topic=self.config.joint_state_topic,
            joint_names=self.follower_joint_names,
        )
        if not self._switch_client.wait_for_service(timeout_s=10.0):
            raise RuntimeError(
                f"AnvilOpenArmLeader: controller_manager/switch_controller service "
                "not available after 10s -- is the devbox reachable and running the "
                "dual-controller mode? (cross-machine ROS2 service calls were "
                "unproven as of plan-writing time; see the plan's risk list.)"
            )

        self._arms_resetter = ArmsResetterClient(self._node)
        if not self._arms_resetter.wait_for_service(timeout_s=10.0):
            raise RuntimeError(
                "AnvilOpenArmLeader: /arms_resetter/reset service not available "
                "after 10s -- is the devbox's arms_resetter_node running?"
            )

        self._spin_stop.clear()
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()
        self._is_connected = True

        # See the matching comment in robot_anvil_openarm.py's connect():
        # gym_manipulator.py's main() never calls teleop_device.disconnect(),
        # so without this the background spin thread is still alive when the
        # interpreter tears down, and rclpy aborts instead of exiting cleanly.
        atexit.register(self.disconnect)

        # get_action()'s edge-detection only calls switch_to_teleop()/
        # switch_to_policy() on a CHANGE in self._was_intervening (starts
        # False, i.e. "policy active" is assumed). If the devbox's actual
        # controller state doesn't already match that assumption -- e.g.
        # leader_follower_arms_teleop's default post-homing state is
        # teleop_controller active, confirmed via Stage 1's real
        # list_controllers response -- then as long as the user never
        # presses space, no edge is ever detected and the follower would
        # silently stay on leader-mirroring for the entire session instead
        # of holding position, which breaks "not intervening = policy
        # holds" for the whole non-intervention case. Force the real
        # baseline to match our internal assumption before the control
        # loop starts, rather than inheriting whatever state was already
        # there. Requires follower_r_forward_position_controller to
        # already be loaded on the devbox (not just teleop_controller) --
        # fails loudly if it isn't, rather than silently continuing with a
        # mismatched state. Must happen AFTER the spin thread starts:
        # switch_to_policy()'s call_async() needs something actively
        # spinning to ever mark its future done, unlike wait_for_service()
        # above (which doesn't need spinning).
        #
        # Wait briefly for a real /joint_states message first: this is the
        # very first switch, so it's the one bumpless transfer matters most
        # for, and switch_to_policy()'s pre-publish check has no retry --
        # without this wait it would likely race the spin thread's first
        # message and silently skip the bumpless pre-publish.
        joint_state_deadline = time.monotonic() + 2.0
        while not self._switch_client.has_follower_joint_state() and time.monotonic() < joint_state_deadline:
            time.sleep(0.02)

        if not self._switch_client.switch_to_policy():
            raise RuntimeError(
                "AnvilOpenArmLeader: failed to establish the policy-controlled "
                "baseline on connect() -- is follower_r_forward_position_controller "
                "actually loaded on the devbox yet (not just teleop_controller)?"
            )

        if self._owns_keyboard_lifecycle:
            self._keyboard.start()

        deadline = time.monotonic() + self.config.obs_timeout_s * 3
        while time.monotonic() < deadline:
            if self._has_leader_joint_data():
                _registered_leaders[self.config.ros_prefix] = self
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"AnvilOpenArmLeader: no {self.config.ros_prefix} joint data on "
            f"{self.config.joint_state_topic} within {self.config.obs_timeout_s * 3}s "
            "-- is the devbox in a mode with leader hardware loaded?"
        )

    def perform_episode_reset_and_wait(
        self,
        timeout_s: float = 15.0,
        before_homing=None,
        after_homed=None,
    ) -> None:
        """Real home-position episode reset, replacing the old
        hold-in-place approach -- called from the patched RobotEnv.reset()
        (patches.py) via get_registered_leader(), since RobotEnv only has
        self.robot, not self.teleop.

        Sequence: (1) release follower_r_forward_position_controller --
        the resetter has zero awareness of it and fails safe (no motion)
        if it's still claiming the interfaces the resetter's JTC needs;
        (2) call the resetter and wait for a real "homed"/"failed"
        completion (the service ack alone is not a completion signal);
        (3) on success, reestablish the same "policy drives by default"
        baseline connect() establishes at session start -- this is the
        actual fix for the structural gap where every episode boundary
        used to leave an unsupervised window for the (still-training)
        policy. On failure, deliberately do NOT reestablish that baseline
        -- leave the arm on teleop_controller (already there from step 1,
        a safe hold) and raise loudly rather than silently continuing in
        an unknown state.

        `before_homing` runs first, before anything moves; `after_homed`
        runs after "homed" is confirmed and BEFORE the policy controller is
        re-activated. The reset patch uses them to clear and then re-seed
        the follower's heartbeat: without that, the heartbeat keeps
        republishing the last pre-reset command and overwrites the bumpless
        home-pose message at re-activation, commanding a large step from
        home back toward the old pose (confirmed on real hardware).
        """
        if before_homing is not None:
            before_homing()

        # Set BEFORE switching to teleop, cleared only once switch_to_policy() below
        # actually confirms success -- see disconnect()'s guard for why this exists.
        # Any exception anywhere in this function (including one raised well after
        # this point) leaves the flag True, which is exactly the "we don't actually
        # know it's safe to hand back to policy" state disconnect() must respect.
        self._reset_in_progress_unsafe = True
        self._switch_client.switch_to_teleop()

        if not self._arms_resetter.reset_and_wait(timeout_s=timeout_s):
            raise RuntimeError(
                "AnvilOpenArmLeader: episode reset failed -- arms_resetter did not "
                "report 'homed' (see its own error log above for the specific "
                "reason). Leaving the arm on teleop_controller rather than "
                "re-engaging the policy in an unknown state."
            )

        if after_homed is not None:
            after_homed()

        if not self._switch_client.switch_to_policy():
            raise RuntimeError(
                "AnvilOpenArmLeader: homing succeeded but re-establishing the "
                "policy-controlled baseline afterward failed."
            )
        self._reset_in_progress_unsafe = False
        self._was_intervening = False
        self._leader_ee_ref = None
        # The keyboard toggle persists across episodes (s/f never touch
        # it) -- clear it too, so every episode starts as "policy drives"
        # and space is a fresh, deliberate per-episode choice. Without
        # this, intervening once would silently carry into every later
        # episode, and the raw flag would disagree with the reestablished
        # baseline above.
        self._keyboard.reset_intervention()

    def _spin_loop(self) -> None:
        """Mirrors AnvilOpenArmFollower._spin_loop's guard (see its
        docstring) -- an uncaught exception in ANY callback on this
        executor would otherwise silently kill this whole background
        thread, starving joint_state updates with no crash/error visible
        at the top level."""
        while not self._spin_stop.is_set() and rclpy.ok():
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception:  # noqa: BLE001
                # See robot_anvil_openarm.py's matching guard: rclpy's
                # logger has no exc_info= kwarg, so format the traceback
                # into the message string instead of passing it as a
                # kwarg (which would itself raise and kill this thread).
                self._node.get_logger().error(
                    "AnvilOpenArmLeader: exception in a ROS callback -- dropping this "
                    f"message and continuing rather than silently killing the spin thread.\n"
                    f"{traceback.format_exc()}"
                )

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _has_leader_joint_data(self) -> bool:
        if self._latest_joint_state is None:
            return False
        names = set(self._latest_joint_state.name)
        required = self.joint_names
        if self.config.use_gripper:
            required = [*required, self.leader_gripper_joint_name]
        return all(j in names for j in required)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _leader_ee_position(self) -> np.ndarray | None:
        if not self._has_leader_joint_data():
            return None
        name_to_pos = dict(zip(self._latest_joint_state.name, self._latest_joint_state.position, strict=False))
        joint_deg = np.rad2deg(np.array([name_to_pos[j] for j in self.joint_names], dtype=np.float64))
        T = self._kinematics.forward_kinematics(joint_deg)
        return T[:3, 3]

    def _handle_switch_edge(self, switch_fn, target_was_intervening: bool) -> None:
        """Real hardware finding (Stage 5): get_action() used to call
        switch_to_teleop()/switch_to_policy() and ignore their boolean
        return value entirely -- if the real switch_controller call failed
        (STRICT rejection, transient timeout, etc.), our internal
        _was_intervening still flipped as if it had succeeded. That leaves
        the follower with neither controller properly holding it (reported
        as the arm going "dead" right after a real give-back) while our own
        code believes the handoff completed -- the exact opposite of the
        "should be able to switch modes interchangeably" requirement this
        whole design exists for.

        Fix: only commit the state transition (_was_intervening,
        _leader_ee_ref) on a CONFIRMED successful switch. On failure, leave
        _was_intervening unchanged so the same edge re-fires and retries on
        the next tick (is_intervening vs _was_intervening still mismatch),
        with a bounded retry count so a genuinely broken switch fails loudly
        instead of retrying forever while the arm sits in an unknown state.
        """
        MAX_CONSECUTIVE_SWITCH_FAILURES = 20  # ~2s at 10Hz

        if switch_fn():
            self._pending_switch_retries = 0
            self._was_intervening = target_was_intervening
            self._leader_ee_ref = self._leader_ee_position() if target_was_intervening else None
            return

        self._pending_switch_retries += 1
        self._node.get_logger().warning(
            f"AnvilOpenArmLeader: switch to "
            f"{'teleop' if target_was_intervening else 'policy'} failed "
            f"(attempt {self._pending_switch_retries}/{MAX_CONSECUTIVE_SWITCH_FAILURES}) "
            "-- will retry next tick. The failure itself means the OTHER "
            "controller still holds the command interface, so the follower "
            "stays actively held (at its last commanded target) by whichever "
            "controller it was already on -- NOT unheld; the handoff just "
            "hasn't taken effect yet (confirmed via real motor-status data, "
            "2026-10-02, after this exact retry sequence: both followers "
            "stayed continuously held throughout, never went ownerless)."
        )
        if self._pending_switch_retries >= MAX_CONSECUTIVE_SWITCH_FAILURES:
            raise RuntimeError(
                f"AnvilOpenArmLeader: switch to "
                f"{'teleop' if target_was_intervening else 'policy'} failed "
                f"{MAX_CONSECUTIVE_SWITCH_FAILURES} consecutive times -- giving up "
                "rather than retrying forever with the follower in an unknown state."
            )

    def get_action(self) -> dict | np.ndarray:
        if not self._is_connected:
            raise RuntimeError("AnvilOpenArmLeader is not connected")

        is_intervening = self._keyboard.is_intervention

        # Edge detection: handle the actual control-source switch here so it
        # happens exactly once per transition, not once per tick.
        if is_intervening and not self._was_intervening:
            self._handle_switch_edge(self._switch_client.switch_to_teleop, target_was_intervening=True)
        elif not is_intervening and self._was_intervening:
            self._handle_switch_edge(self._switch_client.switch_to_policy, target_was_intervening=False)

        if self.config.action_mode == "joint_space":
            return self._get_action_joint_space()
        return self._get_action_ee_delta(is_intervening)

    def _get_action_joint_space(self) -> np.ndarray:
        """joint_space mode: return the leader's own current joint positions
        directly (degrees, follower joint order -- leader_r/follower_r are
        the same 7-DOF arm with a direct 1:1 joint correspondence, no
        kinematics needed). InterventionActionProcessorStep only reads this
        when is_intervention is True (see its own source -- `if
        is_intervention and teleop_action is not None`), so it's safe and
        simplest to always return the leader's real live pose regardless of
        intervention state, rather than special-case a "no-op" value the
        way ee_delta's zero-vector convention needs.

        Also a real, unplanned data-fidelity win over ee_delta: during
        intervention the real follower is mirrored directly by the devbox's
        native teleop_controller, bypassing this whole action_processor
        pipeline -- this label is never what actually drives the arm, only
        what gets recorded/trained on. ee_delta's FK-based approximation of
        that real motion was confirmed to only correlate 0.5-0.7 with what
        the follower actually did (2-3x magnitude mismatches, 1-2 tick lag)
        -- the exact bug this whole joint-space investigation started from.
        Reading the leader's own joint state directly has no such
        approximation error: it *is* what the native mirroring is built to
        reproduce on the follower, not a derived estimate of it.

        When use_gripper=True, one extra element is appended: the leader's
        own leader_gripper_joint_name position, in METERS, NOT degrees --
        that joint is prismatic (see leader_gripper_joint_name's docstring
        in __init__), so it is deliberately excluded from the rad2deg
        conversion applied to the 7 revolute arm joints above.
        """
        n = self.config.num_joints + (1 if self.config.use_gripper else 0)
        if not self._has_leader_joint_data():
            return np.zeros(n, dtype=np.float32)
        name_to_pos = dict(zip(self._latest_joint_state.name, self._latest_joint_state.position, strict=False))
        arm_deg = np.rad2deg(np.array([name_to_pos[j] for j in self.joint_names], dtype=np.float32))
        if not self.config.use_gripper:
            return arm_deg
        gripper_m = np.array([name_to_pos[self.leader_gripper_joint_name]], dtype=np.float32)
        return np.concatenate([arm_deg, gripper_m])

    def _get_action_ee_delta(self, is_intervening: bool) -> dict:
        if not is_intervening or self._leader_ee_ref is None:
            return {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0}

        current_pos = self._leader_ee_position()
        if current_pos is None:
            return {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0}

        raw_delta = current_pos - self._leader_ee_ref
        self._leader_ee_ref = current_pos  # relative tracking: compare against last tick, not a fixed origin

        steps = self.config.step_sizes
        return {
            "delta_x": float(raw_delta[0] / steps["x"]) if steps["x"] else 0.0,
            "delta_y": float(raw_delta[1] / steps["y"]) if steps["y"] else 0.0,
            "delta_z": float(raw_delta[2] / steps["z"]) if steps["z"] else 0.0,
        }

    def get_teleop_events(self) -> dict:
        labels = self._keyboard.consume_episode_labels()

        # NOTE: this used to force switch_to_policy() here on every
        # success/failure label, to avoid leaving the follower stuck on
        # teleop_controller through a reset (Stage 5 finding). Removed --
        # it's now redundant with (and was actively harmful alongside) the
        # real home-position reset: ArmsResetterClient.reset_and_wait()
        # unconditionally releases follower_r_forward_position_controller
        # before every homing call regardless of how the episode ended,
        # and the post-homing baseline handoff (see
        # _on_reset_homed()/connect()) is what now establishes a clean,
        # known "policy drives by default" state every episode -- a single
        # source of truth instead of two independent, easily-desynced
        # give-back paths. Confirmed via a real-hardware incident: a user
        # instructed to "press space again" after this fired unknowingly
        # toggled the keyboard's raw intervention flag OFF (this code path
        # never touched it, only internal bookkeeping), handing the
        # untrained policy uncontested control -- removing this path
        # removes that whole failure class.

        return {
            TeleopEvents.IS_INTERVENTION: self._keyboard.is_intervention,
            TeleopEvents.SUCCESS: labels["success"],
            TeleopEvents.FAILURE: labels["failure"],
            TeleopEvents.RERECORD_EPISODE: labels["rerecord"],
            # `r` must END the episode too: lerobot's record loop only checks
            # RERECORD_EPISODE at the step the episode terminates, and this flag
            # is one-shot, so a bare `r` mid-episode was silently lost and the
            # botched demo got saved anyway. With r terminating, the same step
            # carries RERECORD_EPISODE and the record loop discards the episode.
            TeleopEvents.TERMINATE_EPISODE: labels["success"] or labels["failure"] or labels["rerecord"],
        }

    def send_feedback(self, feedback: dict) -> None:
        pass  # no haptic feedback device

    def disconnect(self) -> None:
        if not self._is_connected:
            return

        if _registered_leaders.get(self.config.ros_prefix) is self:
            del _registered_leaders[self.config.ros_prefix]

        # Real hardware finding (Stage 5): pressing 's'/'f' (success/failure)
        # ends the episode and, for gym_manipulator.py's num_episodes_to_record
        # bookkeeping, effectively ends the whole script -- but nothing about
        # those keys touches the space-bar intervention toggle. If the user
        # presses 's' WHILE STILL INTERVENING (without first pressing space
        # to give back), the script proceeds straight through
        # control_loop()'s mandatory post-episode reset wait and process
        # exit with teleop_controller still active the entire time, rather
        # than the intended "policy holds" resting state. Force a give-back
        # here (state-aware, so it's a safe no-op if we already gave back
        # properly) so the arm never ends a session parked in
        # human-intervention mode just because of keypress ordering. Must
        # run BEFORE stopping the spin thread below -- switch_to_policy()'s
        # call_async() needs it alive to ever resolve, same ordering lesson
        # as connect()'s baseline switch.
        #
        # EXCEPTION, found 2026-09-30 on real hardware: this must NOT fire
        # while a reset is mid-flight or just failed (self._reset_in_progress_
        # unsafe). perform_episode_reset_and_wait() deliberately leaves the
        # arm on teleop_controller and raises rather than re-engaging policy
        # control on an unhomed arm -- but that RuntimeError propagates
        # uncaught up through act_with_policy() to process exit, where THIS
        # atexit-registered disconnect() used to unconditionally call
        # switch_to_policy() anyway, silently overriding that safety decision
        # and handing an unverified, un-homed arm to policy control. Confirmed
        # live: an arms_resetter timeout left the arm exactly this way --
        # controller state looked "normal" (forward_position active) but the
        # arm was never actually homed.
        if self._switch_client is not None and not self._reset_in_progress_unsafe:
            self._switch_client.switch_to_policy()

        if self._owns_keyboard_lifecycle:
            self._keyboard.stop()
        self._spin_stop.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        self._is_connected = False
