"""ROS2-native LeRobot Robot bridging to one Anvil OpenArm follower arm.

Drives the arm over the same ROS2 command topics inference_node.py uses
(std_msgs/Float64MultiArray on .../forward_position_controller/commands),
not lerobot's native CAN-based openarm_follower classes -- this project's
two-PC architecture has no CAN access from the GPU PC, and even on the
devbox those classes would contend with ros2_control for the same bus.
See the HIL-SERL plan for the full reasoning.
"""

import atexit
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float64MultiArray

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot
from lerobot_control.action_limiter import ActionLimiter
from lerobot_control.image_converter import ImageConverter
from lerobot_control.observation_manager import ObservationManager


@RobotConfig.register_subclass("anvil_openarm_follower")
@dataclass
class AnvilOpenArmFollowerConfig(RobotConfig):
    """Config for AnvilOpenArmFollower -- drives one OpenArm follower arm
    over its existing ROS2 command topic, not direct CAN."""

    ros_prefix: str = "follower_r"
    joint_state_topic: str = "/joint_states"
    command_topic: str = "/follower_r_forward_position_controller/commands"
    num_joints: int = 7
    # camera name -> ROS topic. Keep this small (one camera) for the first
    # push-bottle task; add more if the task needs multiple views.
    camera_topics: dict = field(
        default_factory=lambda: {"wrist_r": "/cam_wrist_r/image_raw/compressed"}
    )
    # HIL-SERL's reference configs use 10Hz, well under this rig's 30Hz
    # imitation-learning control_frequency -- deliberately not reusing that
    # default, see the plan's "control-frequency fit" risk note.
    control_frequency: float = 10.0
    # radians; matches configs/lerobot_control/inference_default.yaml's
    # safety.max_position_delta default. Second line of defense alongside
    # HIL-SERL's own end_effector_bounds, not a replacement for it.
    max_position_delta: float = 0.30
    # Per-tick change allowed between consecutive PUBLISHED commands. The
    # max_position_delta clamp above is around the MEASURED pose, so it can't
    # stop consecutive commands alternating up/down by ~0.2 rad (BC-policy
    # j6 wobble, 2026-09-21 trace). None disables.
    max_command_step_rad: float | None = 0.03
    obs_timeout_s: float = 2.0
    # Liveness watchdog window (ticks) -- see AnvilOpenArmFollower's class
    # docstring comment on _LIVENESS_WINDOW_SIZE for the default's original
    # reasoning and 2026-10-12's revision of it. Override per-arm via this
    # config field (e.g. a wider value for an arm/task combination known to
    # pass through a real, transient small-delta phase) rather than editing
    # the class constant, which stays as the dataclass default.
    liveness_window_size: int = 10
    # Gripper/finger joint control. The underlying ROS2 controller ALWAYS
    # expects 8 values (7 arm joints + finger_joint1 -- see
    # gripper_joint_name below) regardless of this flag; use_gripper only
    # controls whether that 8th value is a REAL commanded target (read from
    # the action dict) or a passive hold of the gripper's own current
    # position, and whether observation_features/action_features expose it
    # to the RL/dataset layer at all. False (default) reproduces this
    # class's original gripper-less behavior exactly -- every existing
    # single-arm/follower_l config is unaffected.
    use_gripper: bool = False
    # finger_joint1 is PRISMATIC (meters), not revolute -- confirmed via
    # configs/hilserl/urdf/openarm_teleop_kinematics.urdf's
    # follower_r_finger_joint1 joint (`type="prismatic"`, `limit lower="0.0"
    # upper="0.05"`). Values for this joint are therefore never run through
    # the rad2deg/deg2rad conversion the 7 revolute arm joints use below.
    gripper_min_m: float = 0.0
    gripper_max_m: float = 0.05
    # Per-tick change allowed on the gripper's PUBLISHED command -- same
    # "don't let consecutive commands slam between extremes" motivation as
    # max_command_step_rad above, sized independently since the gripper's
    # whole range (0.05m) is tiny next to an arm joint's. 0.01m/tick at
    # this task's 10Hz control_frequency opens/closes fully in ~0.5s. None
    # disables (matches max_command_step_rad's own convention).
    max_gripper_command_step_m: float | None = 0.01


class _BusShim:
    """Duck-typed stand-in for a lerobot MotorsBus. RobotEnv reaches
    directly into robot.bus.motors (gym_manipulator.py) -- only .keys() is
    ever called, values are never touched. Required or RobotEnv crashes at
    construction; not documented anywhere in the HIL-SERL docs."""

    def __init__(self, joint_names: list[str]):
        self.motors = dict.fromkeys(joint_names)


class AnvilOpenArmFollower(Robot):
    """Owns and spins its own single-threaded rclpy node + background spin
    thread. HIL-SERL's gym_manipulator.py expects a synchronous, blocking
    get_observation()/send_action() API called from a plain Python loop --
    a fundamentally different shape from inference_node.py's timer-driven
    MultiThreadedExecutor design, so this deliberately does not reuse that
    node or its executor.
    """

    config_class = AnvilOpenArmFollowerConfig
    name = "anvil_openarm_follower"

    # Liveness watchdog tuning -- see the docstring on
    # self._liveness_observed_window in __init__ for why this exists.
    # This is the DEFAULT, overridable per-arm via
    # AnvilOpenArmFollowerConfig.liveness_window_size (actual runtime value
    # lives in self._liveness_window_size, set from config in __init__ --
    # this class constant stays fixed at the original default so existing
    # tests that read it directly keep meaning the same thing).
    #
    # ORIGINAL reasoning (window ~1s at 10Hz): every real fault occurrence
    # at the time looked near-instantaneous (affected joints dropping
    # within ~2-7ms), so 1s of sustained "commanded but not moving" seemed
    # like a comfortable margin above normal jitter.
    #
    # REVISED, 2026-10-12: real-hardware trace data (devbox command-vs-
    # actual capture, bimanual push-bottle, follower_l) showed the freeze
    # is not instantaneous at all -- it begins exactly when the policy's
    # commanded deltas drop from a coarse sweep (~0.04-0.06 rad/step) into
    # a fine/contact-seeking correction phase (<0.002 rad/step), step-
    # aligned across repeated deterministic-policy attempts, not time-
    # aligned. Root cause suspected to be a magnitude-dependent dead-band
    # in the closed OpenArm_v10HW plugin or the Damiao motor firmware
    # itself (confirmed: no configurable deadband/threshold parameter
    # exists anywhere in the exposed ros2_control surface for this
    # pipeline -- checked exhaustively on the devbox side). If that's
    # right, the condition is plausibly transient: once the policy's own
    # trajectory moves back into larger deltas, the same joint may start
    # tracking again on its own. A 10-tick (~1-2s) window gives it no time
    # to do that before this watchdog kills the whole session. Per-arm
    # config override (not changed here) widens this for the affected
    # task/arm instead of removing the safety net outright.
    _LIVENESS_WINDOW_SIZE = 10
    # rad; a commanded delta below this is "holding still on purpose",
    # not a real move attempt, and shouldn't count toward the streak.
    _LIVENESS_COMMAND_THRESHOLD_RAD = 0.01
    # rad; the observed position's range (max-min) across the whole
    # window must stay below this for the watchdog to consider the arm
    # "not actually moving" despite sustained real commands.
    _LIVENESS_STILL_THRESHOLD_RAD = 0.002

    def __init__(self, config: AnvilOpenArmFollowerConfig):
        super().__init__(config)
        self.config = config
        self.joint_names = [f"{config.ros_prefix}_joint{i}" for i in range(1, config.num_joints + 1)]
        # Real hardware finding (Stage 5): follower_r_forward_position_controller's
        # actual command interface is 8-wide (7 arm joints + this gripper/finger
        # joint, matching teleop_controller's own claimed_interfaces order,
        # confirmed via Stage 1's list_controllers response), even though this
        # task's use_gripper=false at the lerobot level. Publishing only 7 values
        # gets the controller safety-deactivated by controller_manager
        # ("command size (7) does not match number of interfaces (8)") --
        # confirmed the hard way, this exact fault cascaded into two subsequent
        # switch_controller failures before being traced back to this.
        # Exposed via observation_features/action_features (and therefore
        # to the RL/dataset layer) only when config.use_gripper=True -- see
        # that field's docstring. With use_gripper=False (the original,
        # still-default behavior) this joint stays purely an internal
        # ROS2 command-vector correctness detail, invisible above this
        # class, exactly as before.
        self.gripper_joint_name = f"{config.ros_prefix}_finger_joint1"

        # bus.motors is what RobotEnv (lerobot.rl.gym_manipulator) actually
        # iterates to decide the action/observation vector's joint order --
        # see RobotEnv._get_observation()/step(), both of which read
        # robot.bus.motors.keys() directly, never self.joint_names. Keep
        # self.joint_names itself arm-only (7 revolute joints) everywhere
        # else in this class: the degree-based clamping/liveness math below
        # assumes a homogeneous revolute-joint vector and would be wrong if
        # the gripper (prismatic, meters) were mixed into it.
        bus_motor_names = list(self.joint_names)
        if config.use_gripper:
            bus_motor_names.append(self.gripper_joint_name)
        self.bus = _BusShim(bus_motor_names)
        self.cameras = dict.fromkeys(config.camera_topics)

        self._obs_manager = ObservationManager(device="cpu")
        self._action_limiter = ActionLimiter(
            max_delta=config.max_position_delta,
            model_joint_order=self.joint_names,
            controller_joint_order=self.joint_names,
        )
        self._image_converter = ImageConverter()

        self._node: Node | None = None
        self._executor: SingleThreadedExecutor | None = None
        self._spin_thread: threading.Thread | None = None
        self._spin_stop = threading.Event()
        self._pub = None
        self._is_connected = False
        # Diagnostic instrumentation (real-hardware investigation): the
        # "follower dead" pattern has recurred 5 times with inconsistent
        # timing relative to switch_controller/reset() events. Neither side
        # has ever had direct proof of whether OUR publishes to this
        # topic actually had a gap right before a given occurrence, or
        # whether they were flowing continuously and something else
        # (a fixed timer since controller activation?) is responsible.
        # Logs a WARNING (visible in the same terminal output already
        # being shared) whenever the gap between consecutive send_action()
        # publishes exceeds this threshold -- comfortably above the
        # expected ~100ms (10Hz) cadence, so it only fires on a real
        # anomaly, not normal jitter.
        self._last_publish_time: float | None = None
        self._publish_gap_warn_threshold_s = 0.25

        # Liveness watchdog: the recurring "follower dead" fault (8
        # real-hardware occurrences and counting) has no ROS2-visible
        # signature at all -- controller_manager reports the controller
        # as healthy/active throughout, and our own control loop keeps
        # publishing and logging normally, because nothing in this
        # pipeline ever checks whether the arm actually MOVED in response.
        # Confirmed directly on a real run: the arm died 14s into an
        # episode that still logged a clean "Episode reward: 0.0"
        # completion, because the actor has no feedback path for this at
        # all. This tracks commanded-vs-observed motion over a rolling
        # window and raises if we've been commanding real movement for a
        # sustained period with zero observed response -- turning a
        # silent, undetectable failure into a loud one, independent of
        # whatever the underlying root cause turns out to be.
        self._liveness_window_size = config.liveness_window_size
        self._liveness_observed_window: deque = deque(maxlen=self._liveness_window_size)
        self._liveness_commanded_streak = 0

        # Real hardware finding: reset()'s blocking sleep (already patched
        # in patches.py) was NOT the only source of multi-second publish
        # gaps -- dataset.save_episode()'s own encoding/write I/O, which
        # runs BEFORE reset() is even called, produces its own gap of up
        # to several seconds (confirmed via the warning above: 0.4-3.2s
        # gaps on every episode boundary in a clean, fault-free run,
        # staying comfortably under the ~5.3-5.5s threshold the fault has
        # consistently appeared at in prior occurrences). Rather than
        # patch every individual stock-lerobot blocking call that might
        # produce a gap (reset(), save_episode(), and whatever else might
        # exist), a dedicated heartbeat thread republishes the last-sent
        # command on a fixed cadence regardless of what the main control
        # thread is doing -- robust against gaps from ANY source, current
        # or future.
        self._publish_lock = threading.Lock()
        self._last_sent_msg: Float64MultiArray | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat_interval_s = 0.15

    # -- Robot ABC -----------------------------------------------------

    @property
    def observation_features(self) -> dict:
        feats: dict = {f"{j}.pos": float for j in self.joint_names}
        if self.config.use_gripper:
            feats[f"{self.gripper_joint_name}.pos"] = float
        for cam_name in self.config.camera_topics:
            feats[cam_name] = (480, 640, 3)  # (H, W, C) placeholder; actual shape confirmed at runtime
        return feats

    @property
    def action_features(self) -> dict:
        feats: dict = {f"{j}.pos": float for j in self.joint_names}
        if self.config.use_gripper:
            feats[f"{self.gripper_joint_name}.pos"] = float
        return feats

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        self._node = Node(f"anvil_hilserl_follower_{self.config.ros_prefix}")

        # Separate callback groups + a multi-threaded executor (below) so a
        # slow/backlogged image decode can never block joint_state
        # processing. Real-hardware finding (2026-09-30): a SingleThreadedExecutor
        # serializes ALL callbacks on this Node through one spin_once() loop --
        # publishing (send_action()'s self._pub.publish()) doesn't depend on
        # the executor at all (it's a direct call from the main thread), but
        # RECEIVING does. The wrist camera publishes full-resolution
        # (1080x1920) frames; compressed_imgmsg_to_numpy()'s JPEG decode runs
        # synchronously inside the same spin thread as the joint_state
        # callback -- a slow/backlogged decode can starve joint_state
        # processing for long enough to trip the liveness watchdog, while
        # publish (and therefore commanded motion) keeps working the whole
        # time. This exact asymmetry (publish fine, our own subscription
        # frozen, arm itself and /joint_states confirmed healthy on the
        # publisher side) is what a live occurrence showed via per-tick
        # trace data. Separate callback groups let rclpy schedule joint_state
        # and image callbacks on different threads instead of serializing them.
        # Real bimanual finding (2026-10-02): all cameras on one arm used to
        # share a SINGLE image_cbg -- harmless for single-arm's 1-camera
        # case (nothing to share with), but follower_r's 3 cameras sharing
        # one MutuallyExclusiveCallbackGroup meant only one of their
        # callbacks could ever be "in flight" at a time. Confirmed directly
        # via the connect()-wait diagnostic log: wrist_r and waist each got
        # their first frame within ~1s, while chest (the last-registered
        # subscription in the group) never received a single callback in a
        # full 15s window -- not a dead topic (independently confirmed
        # publishing at a healthy 60Hz), a real starvation/fairness issue
        # within the shared exclusive group under load (4 ROS2 nodes, one
        # publishing /joint_states at ~480Hz, all in one process). Fix:
        # ONE callback group PER camera, so they're scheduled independently
        # rather than competing for one exclusive slot.
        joint_state_cbg = MutuallyExclusiveCallbackGroup()

        self._node.create_subscription(
            JointState, self.config.joint_state_topic, self._on_joint_state, 10,
            callback_group=joint_state_cbg,
        )
        for cam_name, topic in self.config.camera_topics.items():
            self._node.create_subscription(
                CompressedImage, topic, self._make_image_cb(cam_name), 10,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )
        self._pub = self._node.create_publisher(Float64MultiArray, self.config.command_topic, 10)

        # A dedicated executor, not the bare rclpy.spin_once(self._node)
        # convenience call: that function falls back to a single, PROCESS-WIDE
        # global executor when none is given (confirmed via rclpy source:
        # "If no executor is provided ... the global executor is used").
        # AnvilOpenArmFollower and AnvilOpenArmLeader each own a separate
        # Node with their own background spin thread -- with both relying on
        # the implicit global executor, their two threads collide on it the
        # moment both run in the same process, raising "RuntimeError:
        # Executor is already spinning". Reproduced on real hardware at
        # Stage 5 (the first time both classes ever ran together -- Stage 2
        # only had the follower, Stage 4 bypasses the leader's connect()
        # entirely). A private per-node executor avoids this regardless of
        # how many other Robot/Teleoperator instances exist in the process.
        # MultiThreadedExecutor (not Single-) so the separate callback groups
        # above can actually run concurrently -- see their comment.
        # num_threads scales with callback-group count (1 joint_state + 1
        # per camera) + 1 spare, not a fixed 3 -- fixed sizing was fine when
        # every arm had exactly 1 camera (2 groups) but under-provisions
        # follower_r's 3-camera case (4 groups) now that each camera has its
        # own group instead of sharing one.
        self._executor = MultiThreadedExecutor(num_threads=len(self.config.camera_topics) + 2)
        self._executor.add_node(self._node)

        self._spin_stop.clear()
        self._spin_thread = threading.Thread(target=self._spin_loop, daemon=True)
        self._spin_thread.start()

        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        self._is_connected = True

        # gym_manipulator.py's main() never calls robot.disconnect() (no
        # try/finally, no env.close() -- confirmed by reading it directly).
        # Without this, the background spin thread above is still running
        # when the interpreter starts tearing down at process exit, and
        # rclpy's C extension aborts (SIGABRT, "terminate called without an
        # active exception") rather than exiting cleanly -- reproduced during
        # Stage 3 testing, happens identically whether the run succeeded or
        # crashed. atexit guarantees disconnect() runs before that teardown.
        atexit.register(self.disconnect)

        # Fail loudly if no data ever arrives, rather than hanging forever
        # the first time get_observation() is actually called.
        #
        # Diagnostic logging added 2026-10-02: a bimanual connect() attempt
        # with 4 ROS2 nodes (2 followers + 2 leaders) now running in one
        # process, follower_r alone subscribing to 3 cameras sharing one
        # MutuallyExclusiveCallbackGroup, timed out waiting for a complete
        # observation even at a generous 15s -- despite all 4 real topics
        # independently confirmed publishing at healthy rates moments
        # later. Unknown whether this is uneven throughput (all cameras
        # eventually arrive, just slower than before) or a harder
        # starvation/scheduling issue (some specific camera never gets a
        # single callback at all) -- this periodic log (every ~2s) plus
        # _make_image_cb's one-shot "first frame received" log make that
        # distinction visible in the next real attempt instead of guessing.
        deadline = time.monotonic() + self.config.obs_timeout_s * 3
        last_progress_log = 0.0
        while time.monotonic() < deadline:
            if self._obs_manager.has_complete_observation(list(self.config.camera_topics)):
                self._seed_heartbeat_from_current_position()
                return
            now = time.monotonic()
            if now - last_progress_log > 2.0:
                last_progress_log = now
                missing_cameras = [c for c in self.config.camera_topics if c not in self._obs_manager.latest_images]
                self._node.get_logger().info(
                    f"AnvilOpenArmFollower: still waiting for complete observation -- "
                    f"joint_state={'OK' if self._obs_manager.latest_joint_state is not None else 'MISSING'}, "
                    f"missing_cameras={missing_cameras}"
                )
            time.sleep(0.05)
        raise RuntimeError(
            f"AnvilOpenArmFollower: no complete observation within "
            f"{self.config.obs_timeout_s * 3}s of connecting -- check that "
            f"{self.config.joint_state_topic} and {list(self.config.camera_topics.values())} "
            "are actually being published (ROS_DOMAIN_ID / CycloneDDS peer config?)."
        )

    def _spin_loop(self) -> None:
        """Real-hardware finding (2026-09-22): a single corrupted/truncated
        compressed camera frame (transient network issue, not a code bug --
        cv2.imdecode() returning None) raised inside _cb()'s image-decode
        call with no try/except anywhere in the callback chain. Since
        nothing here caught it either, the exception propagated out of
        spin_once() and killed this ENTIRE background thread silently
        (Python threads print a traceback and just stop, no crash of the
        main process). Every subsequent joint_state/image callback then
        went unprocessed -- observations froze -- and ~7s later the
        (correctly-behaving) liveness watchdog tripped on what looked like
        an unresponsive arm, when the real cause was one bad camera frame
        several seconds earlier. Guard broadly here (any callback, not
        just images) so a single bad message degrades gracefully (that one
        message is dropped, spinning continues) instead of silently
        starving the whole observation pipeline.
        """
        while not self._spin_stop.is_set() and rclpy.ok():
            try:
                self._executor.spin_once(timeout_sec=0.05)
            except Exception:  # noqa: BLE001 -- see docstring: must not let this thread die
                # rclpy's logger (unlike stdlib logging) has no exc_info=
                # kwarg -- passing one raises TypeError from INSIDE this
                # handler, which would silently kill the thread anyway
                # (confirmed by _stage7b's own test catching this).
                # Format the traceback into the message string instead.
                self._node.get_logger().error(
                    "AnvilOpenArmFollower: exception in a ROS callback -- dropping this "
                    f"message and continuing rather than silently killing the spin thread.\n"
                    f"{traceback.format_exc()}"
                )

    def _heartbeat_loop(self) -> None:
        """Republishes the last-sent command if send_action() hasn't been
        called recently, so the command topic is never silent for longer
        than self._heartbeat_interval_s regardless of what the main
        control thread is doing (dataset I/O, video encoding, a blocking
        reset() wait, etc.). See __init__'s comment for the full story."""
        while not self._heartbeat_stop.is_set() and rclpy.ok():
            time.sleep(self._heartbeat_interval_s)
            with self._publish_lock:
                if self._last_sent_msg is None or self._last_publish_time is None:
                    continue
                if time.monotonic() - self._last_publish_time < self._heartbeat_interval_s:
                    continue  # send_action() itself published recently enough
                # rclpy.shutdown() can be triggered from elsewhere in the
                # process (lerobot's own SIGINT handling in
                # lerobot.rl.process, unrelated to our own disconnect()) --
                # reproduced on real hardware: this thread and
                # AnvilOpenArmLeader.disconnect()'s give-back both hit
                # "publisher's context is invalid" during actor.py's
                # shutdown. Only a shutdown-time cosmetic error (the real
                # control window is already over by then), but check
                # rclpy.ok() again right before publishing to avoid it.
                if not rclpy.ok():
                    return
                self._pub.publish(self._last_sent_msg)
                self._last_publish_time = time.monotonic()

    def _seed_heartbeat_from_current_position(self) -> None:
        """Real hardware finding (Stage 6): the heartbeat above only
        republishes `self._last_sent_msg`, which stayed `None` until the
        first successful `send_action()` call -- meaning it provided ZERO
        protection during the gap between connect() finishing and that
        first call. The actor path's first real policy inference is a cold
        start (model/GPU warmup) that took ~6s in practice ("Policy FPS 1.1
        below required 10 at step 0"), well past the fault's observed
        ~5.3-5.5s threshold -- and the devbox's motor-status log confirmed
        the fault landed in exactly this window, with no reset()/switch
        anywhere nearby (ruling out every previously-suspected gap source
        and independently ruling out a fixed activation-timer explanation
        too, via a separate 39.5-minute idle-with-no-fault data point).

        Fix: seed `_last_sent_msg` here, immediately once a real
        observation is available (before connect() ever returns), using
        the robot's OWN current actual position as the initial hold target
        -- the same "echo back reality, command no real motion" principle
        already used for the gripper-hold value and the bumpless-transfer
        pre-publish. Closes the one remaining unprotected window: from
        connect() completing to whatever calls send_action() first,
        however long that turns out to take.
        """
        msg = self._obs_manager.latest_joint_state
        if msg is None:
            return
        name_to_pos = dict(zip(msg.name, msg.position, strict=False))
        if not all(j in name_to_pos for j in [*self.joint_names, self.gripper_joint_name]):
            return

        hold_msg = Float64MultiArray()
        hold_msg.data = [name_to_pos[j] for j in self.joint_names] + [name_to_pos[self.gripper_joint_name]]

        with self._publish_lock:
            self._last_sent_msg = hold_msg
            self._last_publish_time = time.monotonic()

    def clear_heartbeat(self) -> None:
        """Stop republishing any remembered command. Called at the start
        of a homing reset: the arm is about to be moved by the devbox's
        arms_resetter, so whatever was last commanded is about to become
        stale, and must never be republished afterwards."""
        with self._publish_lock:
            self._last_sent_msg = None

    def reseed_heartbeat_after_reset(self) -> None:
        """Re-seed the heartbeat with the arm's ACTUAL post-homing pose.

        Real hardware finding (first full session with homing): the
        heartbeat kept republishing the last pre-reset command (the arm's
        pose at the end of the previous episode) throughout homing and
        right through re-activation of follower_r_forward_position_controller
        -- overwriting the one bumpless home-position message
        switch_to_policy() publishes, and commanding a large step from
        home back toward the old pose at high gain the instant the
        controller came up. The arm stopped responding within ~1.5s. Must
        run after homing completes and BEFORE the policy controller is
        re-activated. If the current pose can't be read, the heartbeat is
        left empty (silent) rather than stale -- silence is safe, a stale
        target is not.
        """
        self.clear_heartbeat()
        self._seed_heartbeat_from_current_position()

    def _on_joint_state(self, msg: JointState) -> None:
        self._obs_manager.update_joint_state(msg)

    def _make_image_cb(self, cam_name: str):
        def _cb(msg: CompressedImage) -> None:
            if cam_name not in self._obs_manager.latest_images:
                self._node.get_logger().info(
                    f"AnvilOpenArmFollower: first frame received for camera '{cam_name}'"
                )
            img = self._image_converter.compressed_imgmsg_to_numpy(msg, desired_encoding="rgb8")
            self._obs_manager.update_image(cam_name, img)

        return _cb

    @property
    def is_calibrated(self) -> bool:
        return True  # devbox owns real motor calibration; nothing to do here

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_observation(self) -> dict:
        if not self._is_connected:
            raise RuntimeError("AnvilOpenArmFollower is not connected")

        deadline = time.monotonic() + self.config.obs_timeout_s
        while time.monotonic() < deadline:
            if self._obs_manager.has_complete_observation(list(self.config.camera_topics)):
                break
            time.sleep(0.005)
        else:
            raise TimeoutError(
                f"AnvilOpenArmFollower.get_observation(): no complete observation "
                f"within {self.config.obs_timeout_s}s"
            )

        msg = self._obs_manager.latest_joint_state
        name_to_pos = dict(zip(msg.name, msg.position, strict=False))

        obs: dict = {}
        for j in self.joint_names:
            if j not in name_to_pos:
                raise KeyError(
                    f"AnvilOpenArmFollower: joint '{j}' not found in /joint_states "
                    f"(saw: {list(name_to_pos)[:5]}...). Is the devbox in a mode "
                    "that loads this arm's hardware?"
                )
            # RobotKinematics/RobotEnv convention is degrees (verified against
            # lerobot/model/kinematics.py's deg2rad/rad2deg calls).
            obs[f"{j}.pos"] = float(np.rad2deg(name_to_pos[j]))

        if self.config.use_gripper:
            if self.gripper_joint_name not in name_to_pos:
                raise KeyError(
                    f"AnvilOpenArmFollower: use_gripper=True but '{self.gripper_joint_name}' not "
                    f"found in /joint_states (saw: {list(name_to_pos)[:5]}...)."
                )
            # Raw meters, NOT rad2deg -- finger_joint1 is prismatic (see
            # gripper_min_m/gripper_max_m's docstring in the config above).
            obs[f"{self.gripper_joint_name}.pos"] = float(name_to_pos[self.gripper_joint_name])

        for cam_name in self.config.camera_topics:
            obs[cam_name] = self._obs_manager.latest_images[cam_name]

        return obs

    @staticmethod
    def _scale_to_cap(delta: np.ndarray, cap: float) -> np.ndarray:
        """Shrink `delta` uniformly so its largest joint component is <= cap,
        keeping the direction (ratios between joints) unchanged."""
        largest = float(np.max(np.abs(delta)))
        if largest <= cap or largest == 0.0:
            return delta
        return delta * (cap / largest)

    def _check_liveness(self, commanded_rad: np.ndarray, observed_rad: np.ndarray | None) -> None:
        """Raise if we've been commanding real movement for a full window
        with zero observed response -- see the watchdog docstring in
        __init__ for why this exists and what it's protecting against.

        Deliberately ARM-ONLY: commanded_rad/observed_rad never include the
        gripper, by construction (callers only ever pass the 7-joint
        arrays). A grasp holding against a real object is EXPECTED to stop
        moving while still being commanded -- that's a legitimate contact
        state, not a fault, and including the gripper here would make this
        watchdog trip on every successful grasp.
        """
        if observed_rad is None:
            # Can't evaluate this tick (joint state incomplete) -- don't
            # let a transient observation gap reset a streak in progress.
            return

        # While a human is intervening, teleop_controller drives the arm and
        # follower_r_forward_position_controller is INACTIVE, so our commands
        # are ignored by design -- a still leader arm then looks exactly like
        # "commanded movement, no response". Confirmed false positive: the
        # actor died ~8s after the user pressed space (2026-09-21 07:13).
        # is_any_leader_intervening() (not a single get_registered_leader()
        # lookup): the registry became keyed by ros_prefix so a second
        # AnvilOpenArmLeader (bimanual, one per arm) can coexist without
        # overwriting the first's registration -- see that module's
        # docstring for the overwrite bug this replaced. For exactly one
        # registered leader (the single-arm case this watchdog was
        # originally written for) this is byte-for-byte the same check as
        # before.
        from lerobot_control.hilserl.teleop_anvil_openarm_leader import is_any_leader_intervening

        if is_any_leader_intervening():
            self._liveness_commanded_streak = 0
            self._liveness_observed_window.clear()
            return

        commanded_delta = float(np.max(np.abs(commanded_rad - observed_rad)))
        if commanded_delta > self._LIVENESS_COMMAND_THRESHOLD_RAD:
            self._liveness_commanded_streak += 1
        else:
            self._liveness_commanded_streak = 0

        self._liveness_observed_window.append(observed_rad)

        if (
            self._liveness_commanded_streak < self._liveness_window_size
            or len(self._liveness_observed_window) < self._liveness_window_size
        ):
            return

        observed_stack = np.stack(self._liveness_observed_window)
        observed_range = float(np.max(observed_stack.max(axis=0) - observed_stack.min(axis=0)))
        if observed_range < self._LIVENESS_STILL_THRESHOLD_RAD:
            raise RuntimeError(
                f"AnvilOpenArmFollower({self.config.ros_prefix}): liveness watchdog "
                f"tripped -- commanded real movement "
                f"(>{self._LIVENESS_COMMAND_THRESHOLD_RAD} rad delta) "
                f"for {self._liveness_window_size} consecutive ticks but observed "
                f"joint position only moved {observed_range:.5f} rad over that same "
                f"window (threshold {self._LIVENESS_STILL_THRESHOLD_RAD} rad). The "
                f"{self.config.ros_prefix} arm appears unresponsive -- see the "
                "recurring 'follower dead' fault investigation in the HIL-SERL plan. "
                "(2026-10-02: this message used to omit which arm, forcing physical "
                "observation to tell which one faulted in a bimanual run -- fixed "
                "after the first live bimanual actor session needed exactly that.)"
            )

    def send_action(self, action: dict) -> dict:
        if not self._is_connected:
            raise RuntimeError("AnvilOpenArmFollower is not connected")

        target_deg = np.array([action[f"{j}.pos"] for j in self.joint_names], dtype=np.float64)
        target_rad = np.deg2rad(target_deg)

        msg = self._obs_manager.latest_joint_state
        name_to_pos = dict(zip(msg.name, msg.position, strict=False)) if msg is not None else {}

        current_rad = None
        if all(j in name_to_pos for j in self.joint_names):
            current_rad = np.array([name_to_pos[j] for j in self.joint_names], dtype=np.float64)

        # Both limits below scale the WHOLE joint-delta vector together
        # (direction-preserving) instead of clipping each joint independently.
        # Independent per-joint clipping (what ActionLimiter.apply_delta_limit
        # and the earlier version of the step cap here did) changes the RATIO
        # between joints, so the IK's intended end-effector motion is bent
        # into a different direction whenever more than one joint saturates.
        # Reproduced offline 2026-09-24 (simulated real pipeline): from the
        # homed pose, the z-ceiling clip demands a ~5 cm descent, the joints
        # (1, 4, 7) saturate at different ratios, and the hand ROSE from
        # z=0.357 to 0.45 with joint 7 swinging 31 -> 89 deg -- matching the
        # real recorded episodes (all 13 rose, joint 7 to ~70 deg). With no
        # limiter, or scaling as below, the same demand descends to the
        # ceiling and holds. The shared ActionLimiter is left alone (it also
        # serves production inference); only this class stops using it.
        if current_rad is not None:
            clamped_rad = current_rad + self._scale_to_cap(
                target_rad - current_rad, self.config.max_position_delta
            )
        else:
            clamped_rad = target_rad

        step = self.config.max_command_step_rad
        if step is not None:
            with self._publish_lock:
                last = self._last_sent_msg
            if last is not None:
                last_arm = np.asarray(last.data[: len(clamped_rad)], dtype=np.float64)
                clamped_rad = last_arm + self._scale_to_cap(clamped_rad - last_arm, step)

        self._check_liveness(clamped_rad, current_rad)

        if self.gripper_joint_name not in name_to_pos:
            raise RuntimeError(
                f"AnvilOpenArmFollower.send_action(): '{self.gripper_joint_name}' not "
                "found in latest /joint_states -- can't build the full 8-value command "
                "vector this controller expects (see gripper_joint_name's docstring in "
                "__init__ for why 7 values isn't enough)."
            )

        if self.config.use_gripper:
            gripper_key = f"{self.gripper_joint_name}.pos"
            if gripper_key not in action:
                raise KeyError(
                    f"AnvilOpenArmFollower.send_action(): use_gripper=True but '{gripper_key}' "
                    "missing from action -- action_features declares this key, so every caller "
                    "(RobotEnv.step(), the composite's send_action(), any test) must provide it."
                )
            gripper_cmd_m = float(
                np.clip(action[gripper_key], self.config.gripper_min_m, self.config.gripper_max_m)
            )
            gripper_step = self.config.max_gripper_command_step_m
            if gripper_step is not None:
                with self._publish_lock:
                    last = self._last_sent_msg
                if last is not None:
                    last_gripper_m = float(last.data[-1])
                    delta = gripper_cmd_m - last_gripper_m
                    if abs(delta) > gripper_step:
                        gripper_cmd_m = last_gripper_m + gripper_step * (1.0 if delta > 0 else -1.0)
        else:
            # Hold the gripper at its own current position -- unchanged
            # from this class's original gripper-less behavior.
            gripper_cmd_m = float(name_to_pos[self.gripper_joint_name])

        out_msg = Float64MultiArray()
        out_msg.data = clamped_rad.tolist() + [gripper_cmd_m]

        with self._publish_lock:
            now = time.monotonic()
            if self._last_publish_time is not None:
                gap = now - self._last_publish_time
                if gap > self._publish_gap_warn_threshold_s:
                    self._node.get_logger().warning(
                        f"AnvilOpenArmFollower: {gap:.3f}s gap since last publish to "
                        f"{self.config.command_topic} (expected ~0.1s at 10Hz) -- "
                        "the heartbeat thread should now prevent this from ever "
                        "exceeding ~0.15s in practice; a large gap here would mean "
                        "even the heartbeat itself got starved."
                    )
            self._last_publish_time = now
            self._last_sent_msg = out_msg
            # rclpy.shutdown() can be triggered elsewhere in the process
            # (lerobot's own SIGINT handling) between this lock being
            # acquired and the publish call below -- same race already
            # guarded in _heartbeat_loop()/switch_to_policy(), but this
            # call site (send_action()'s own direct publish, also reused
            # by the reset-keepalive patch) was missed. A shutdown-time
            # publish attempt here is a no-op, not a real failure.
            if rclpy.ok():
                self._pub.publish(out_msg)

        sent = {f"{j}.pos": float(np.rad2deg(v)) for j, v in zip(self.joint_names, clamped_rad, strict=False)}
        if self.config.use_gripper:
            sent[f"{self.gripper_joint_name}.pos"] = gripper_cmd_m
        return sent

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)
        self._spin_stop.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
        self._is_connected = False
