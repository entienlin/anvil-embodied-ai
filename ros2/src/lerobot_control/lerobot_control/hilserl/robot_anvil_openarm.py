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
from rclpy.executors import SingleThreadedExecutor
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
    # Window size ~1s at this task's 10Hz control_frequency: every real
    # fault occurrence so far has been near-instantaneous (all affected
    # joints drop within ~2-7ms), so 1s of sustained "commanded but not
    # moving" is a comfortable margin above any normal settle/latency
    # jitter while still catching a real fault within one second of it
    # happening, not many seconds later.
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
        # switch_controller failures before being traced back to this. Not
        # exposed via observation_features/action_features -- purely an
        # internal ROS2 command-vector correctness detail, invisible to the
        # RL/dataset layer, which stays 7-dim per the task's actual design.
        self.gripper_joint_name = f"{config.ros_prefix}_finger_joint1"

        self.bus = _BusShim(self.joint_names)
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
        self._liveness_observed_window: deque = deque(maxlen=self._LIVENESS_WINDOW_SIZE)
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
        for cam_name in self.config.camera_topics:
            feats[cam_name] = (480, 640, 3)  # (H, W, C) placeholder; actual shape confirmed at runtime
        return feats

    @property
    def action_features(self) -> dict:
        return {f"{j}.pos": float for j in self.joint_names}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return

        if not rclpy.ok():
            rclpy.init(args=None)

        self._node = Node(f"anvil_hilserl_follower_{self.config.ros_prefix}")

        self._node.create_subscription(JointState, self.config.joint_state_topic, self._on_joint_state, 10)
        for cam_name, topic in self.config.camera_topics.items():
            self._node.create_subscription(CompressedImage, topic, self._make_image_cb(cam_name), 10)
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
        self._executor = SingleThreadedExecutor()
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
        deadline = time.monotonic() + self.config.obs_timeout_s * 3
        while time.monotonic() < deadline:
            if self._obs_manager.has_complete_observation(list(self.config.camera_topics)):
                self._seed_heartbeat_from_current_position()
                return
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

        for cam_name in self.config.camera_topics:
            obs[cam_name] = self._obs_manager.latest_images[cam_name]

        return obs

    def _check_liveness(self, commanded_rad: np.ndarray, observed_rad: np.ndarray | None) -> None:
        """Raise if we've been commanding real movement for a full window
        with zero observed response -- see the watchdog docstring in
        __init__ for why this exists and what it's protecting against.
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
        from lerobot_control.hilserl.teleop_anvil_openarm_leader import get_registered_leader

        leader = get_registered_leader()
        if leader is not None and (leader._was_intervening or leader._keyboard.is_intervention):
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
            self._liveness_commanded_streak < self._LIVENESS_WINDOW_SIZE
            or len(self._liveness_observed_window) < self._LIVENESS_WINDOW_SIZE
        ):
            return

        observed_stack = np.stack(self._liveness_observed_window)
        observed_range = float(np.max(observed_stack.max(axis=0) - observed_stack.min(axis=0)))
        if observed_range < self._LIVENESS_STILL_THRESHOLD_RAD:
            raise RuntimeError(
                f"AnvilOpenArmFollower: liveness watchdog tripped -- commanded "
                f"real movement (>{self._LIVENESS_COMMAND_THRESHOLD_RAD} rad delta) "
                f"for {self._LIVENESS_WINDOW_SIZE} consecutive ticks but observed "
                f"joint position only moved {observed_range:.5f} rad over that same "
                f"window (threshold {self._LIVENESS_STILL_THRESHOLD_RAD} rad). The "
                "arm appears unresponsive -- see the recurring 'follower dead' "
                "fault investigation in the HIL-SERL plan."
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

        clamped_rad = self._action_limiter.process(target_rad, current_positions=current_rad)

        step = self.config.max_command_step_rad
        if step is not None:
            with self._publish_lock:
                last = self._last_sent_msg
            if last is not None:
                last_arm = np.asarray(last.data[: len(clamped_rad)], dtype=np.float64)
                clamped_rad = last_arm + np.clip(clamped_rad - last_arm, -step, step)

        self._check_liveness(clamped_rad, current_rad)

        if self.gripper_joint_name not in name_to_pos:
            raise RuntimeError(
                f"AnvilOpenArmFollower.send_action(): '{self.gripper_joint_name}' not "
                "found in latest /joint_states -- can't build the full 8-value command "
                "vector this controller expects (see gripper_joint_name's docstring in "
                "__init__ for why 7 values isn't enough)."
            )
        # Hold the gripper at its own current position -- this task never
        # commands it, so echo it back rather than moving it.
        gripper_hold_rad = float(name_to_pos[self.gripper_joint_name])

        out_msg = Float64MultiArray()
        out_msg.data = clamped_rad.tolist() + [gripper_hold_rad]

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

        return {f"{j}.pos": float(np.rad2deg(v)) for j, v in zip(self.joint_names, clamped_rad, strict=False)}

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
