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

from .keyboard_intervention import KeyboardIntervention
from .switch_service_client import SwitchServiceClient


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


class AnvilOpenArmLeader(Teleoperator):
    config_class = AnvilOpenArmLeaderConfig
    name = "anvil_openarm_leader"

    def __init__(self, config: AnvilOpenArmLeaderConfig):
        super().__init__(config)
        self.config = config
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

        self._keyboard = KeyboardIntervention()
        self._switch_client: SwitchServiceClient | None = None
        self._was_intervening = False
        self._leader_ee_ref: np.ndarray | None = None  # reset each time intervention (re)starts
        self._pending_switch_retries = 0  # see get_action()'s switch-failure handling

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

        self._keyboard.start()

        deadline = time.monotonic() + self.config.obs_timeout_s * 3
        while time.monotonic() < deadline:
            if self._has_leader_joint_data():
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"AnvilOpenArmLeader: no {self.config.ros_prefix} joint data on "
            f"{self.config.joint_state_topic} within {self.config.obs_timeout_s * 3}s "
            "-- is the devbox in a mode with leader hardware loaded?"
        )

    def _spin_loop(self) -> None:
        while not self._spin_stop.is_set() and rclpy.ok():
            self._executor.spin_once(timeout_sec=0.05)

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def _has_leader_joint_data(self) -> bool:
        if self._latest_joint_state is None:
            return False
        names = set(self._latest_joint_state.name)
        return all(j in names for j in self.joint_names)

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
            "-- will retry next tick, follower may be unheld until this succeeds"
        )
        if self._pending_switch_retries >= MAX_CONSECUTIVE_SWITCH_FAILURES:
            raise RuntimeError(
                f"AnvilOpenArmLeader: switch to "
                f"{'teleop' if target_was_intervening else 'policy'} failed "
                f"{MAX_CONSECUTIVE_SWITCH_FAILURES} consecutive times -- giving up "
                "rather than retrying forever with the follower in an unknown state."
            )

    def get_action(self) -> dict:
        if not self._is_connected:
            raise RuntimeError("AnvilOpenArmLeader is not connected")

        is_intervening = self._keyboard.is_intervention

        # Edge detection: handle the actual control-source switch here so it
        # happens exactly once per transition, not once per tick.
        if is_intervening and not self._was_intervening:
            self._handle_switch_edge(self._switch_client.switch_to_teleop, target_was_intervening=True)
        elif not is_intervening and self._was_intervening:
            self._handle_switch_edge(self._switch_client.switch_to_policy, target_was_intervening=False)

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
        return {
            TeleopEvents.IS_INTERVENTION: self._keyboard.is_intervention,
            TeleopEvents.SUCCESS: labels["success"],
            TeleopEvents.FAILURE: labels["failure"],
            TeleopEvents.RERECORD_EPISODE: labels["rerecord"],
            TeleopEvents.TERMINATE_EPISODE: labels["success"] or labels["failure"],
        }

    def send_feedback(self, feedback: dict) -> None:
        pass  # no haptic feedback device

    def disconnect(self) -> None:
        if not self._is_connected:
            return

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
        if self._switch_client is not None:
            self._switch_client.switch_to_policy()

        self._keyboard.stop()
        self._spin_stop.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        self._is_connected = False
