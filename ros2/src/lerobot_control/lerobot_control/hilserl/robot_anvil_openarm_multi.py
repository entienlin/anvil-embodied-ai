"""Composite ROS2-native LeRobot Robot bridging to 1-or-N Anvil OpenArm
follower arms at once (single-arm or bimanual, same class).

Why a composite rather than a separate bimanual-specific class: lerobot's
RobotEnv (lerobot/rl/gym_manipulator.py) expects exactly ONE `robot` object
implementing the Robot ABC -- it reaches directly into `robot.bus.motors`
and `robot.cameras`, and calls a single `get_observation()`/`send_action()`
pair once per control tick. Bimanual control (2 arms, 4 cameras) needs to
present as that one object while actually driving two independent ROS2
arms underneath.

Design choice (see the HIL-SERL bimanual task notes for the full
tradeoff): this class is parameterized by `arms: list[...]`, a list of
AnvilOpenArmFollowerConfig-shaped entries, length 1 for single-arm or 2 for
bimanual (nothing structurally stops a longer list, but only 1-2 are
exercised by this project's tests/configs so far). It is NOT a new
bimanual-specific reimplementation of the ROS2/watchdog/heartbeat/liveness
logic in robot_anvil_openarm.py -- it literally constructs one
AnvilOpenArmFollower per list entry and delegates every real operation to
them, concatenating/fanning-out at the edges only. This means:

  - AnvilOpenArmFollower itself is COMPLETELY UNCHANGED and untouched by
    this file (other than the _check_liveness/is_any_leader_intervening
    wiring already covered in that module) -- every single-arm config
    that uses `type: anvil_openarm_follower` directly keeps behaving
    exactly as before, independent of whether this composite exists.
  - Each arm gets its OWN liveness watchdog with its own window/streak
    state (an unresponsive left arm must not be masked by a responsive
    right arm, and vice versa -- these are two physically independent
    arms with two independent fault histories), simply because each
    sub-controller is a real, separate AnvilOpenArmFollower instance with
    its own watchdog state, not shared aggregate state invented here.
  - This same class IS the single-arm class too, when given a 1-element
    `arms` list -- see the _stage test asserting single-arm-via-composite
    produces identical observation_features/action_features/get_observation
    shapes to a bare AnvilOpenArmFollower constructed directly, which is
    the behavioral-equivalence property that makes "pick the composite for
    everything, make arm count a config knob" a safe, trivial future merge
    rather than a second code path to keep in sync.
"""

from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.robots.robot import Robot

from .robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig, _BusShim


def _coerce_arm_config(entry) -> AnvilOpenArmFollowerConfig:
    """Accepts either an already-built AnvilOpenArmFollowerConfig (the
    normal case when constructing this in Python, e.g. from tests or
    another config's __post_init__) or a plain dict (the normal case when
    this config is loaded from YAML -- `arms` is typed as a bare `list`,
    not `list[AnvilOpenArmFollowerConfig]`, deliberately mirroring this
    package's existing AnvilOpenArmFollowerConfig.camera_topics: dict
    field, which is also a bare, untyped-generic container for the same
    reason: this project could not confirm draccus reliably reconstructs
    nested dataclasses inside a list/dict field, so nested arm configs are
    parsed as plain dicts and converted here instead, where it's trivial
    to test without needing draccus at all)."""
    if isinstance(entry, AnvilOpenArmFollowerConfig):
        return entry
    if isinstance(entry, dict):
        return AnvilOpenArmFollowerConfig(**entry)
    raise TypeError(
        f"AnvilOpenArmMultiFollowerConfig.arms entries must be an "
        f"AnvilOpenArmFollowerConfig or a dict of its fields, got {type(entry)!r}"
    )


@RobotConfig.register_subclass("anvil_openarm_multi_follower")
@dataclass
class AnvilOpenArmMultiFollowerConfig(RobotConfig):
    """One entry per arm. Defaults to a single follower_r arm (so this
    config type is itself usable as a drop-in single-arm replacement, not
    just a bimanual-only type) -- real bimanual configs override `arms`
    with two entries. See configs/hilserl/train_rl_push_bottle_bimanual_jointspace.yaml
    for the real 2-arm/4-camera example this project uses.
    """

    arms: list = field(default_factory=lambda: [AnvilOpenArmFollowerConfig()])


class AnvilOpenArmMultiFollower(Robot):
    """Owns N AnvilOpenArmFollower sub-controllers (N = len(config.arms)),
    each a fully independent ROS2 node/spin-thread/heartbeat/watchdog, and
    presents them as one combined Robot to RobotEnv.

    Joint/camera ordering in observation_features/action_features/
    get_observation()/send_action() follows config.arms' list order,
    arm-by-arm, each arm's own joint_names order within itself (matching
    AnvilOpenArmFollower's own `f"{ros_prefix}_joint{i}"` order) -- this
    must match whatever order the env config's `features`/`features_map`
    declare, exactly as the single-arm case already requires for its own
    7 joints.
    """

    config_class = AnvilOpenArmMultiFollowerConfig
    name = "anvil_openarm_multi_follower"

    def __init__(self, config: AnvilOpenArmMultiFollowerConfig):
        super().__init__(config)
        self.config = config

        arm_configs = [_coerce_arm_config(a) for a in config.arms]
        if not arm_configs:
            raise ValueError("AnvilOpenArmMultiFollowerConfig.arms must have at least one entry")

        self._arms: list[AnvilOpenArmFollower] = [AnvilOpenArmFollower(c) for c in arm_configs]

        all_joint_names: list[str] = []
        all_camera_names: list[str] = []
        for arm in self._arms:
            # arm.bus.motors.keys(), NOT arm.joint_names: includes the
            # gripper/finger joint when that arm's use_gripper=True (see
            # AnvilOpenArmFollower.__init__'s bus_motor_names) -- this
            # composite's own self.joint_names/self.bus must reflect
            # EXACTLY what RobotEnv (gym_manipulator.py) iterates over for
            # action/observation vector order, which reads
            # robot.bus.motors.keys() directly, never .joint_names. For any
            # arm with use_gripper=False (the default), this is identical
            # to arm.joint_names -- no behavior change for follower_l or
            # any other gripper-less arm.
            arm_bus_names = list(arm.bus.motors.keys())
            dup_joints = set(all_joint_names) & set(arm_bus_names)
            if dup_joints:
                raise ValueError(
                    f"AnvilOpenArmMultiFollower: duplicate joint name(s) {dup_joints} across "
                    "arms -- each arm's ros_prefix must be unique"
                )
            dup_cams = set(all_camera_names) & set(arm.config.camera_topics)
            if dup_cams:
                raise ValueError(
                    f"AnvilOpenArmMultiFollower: duplicate camera name(s) {dup_cams} across "
                    "arms -- every camera name (across ALL arms) must be unique"
                )
            all_joint_names.extend(arm_bus_names)
            all_camera_names.extend(arm.config.camera_topics)

        self.joint_names = all_joint_names
        self.bus = _BusShim(all_joint_names)
        self.cameras = dict.fromkeys(all_camera_names)

    # -- Robot ABC -----------------------------------------------------

    @property
    def observation_features(self) -> dict:
        feats: dict = {}
        for arm in self._arms:
            feats.update(arm.observation_features)
        return feats

    @property
    def action_features(self) -> dict:
        feats: dict = {}
        for arm in self._arms:
            feats.update(arm.action_features)
        return feats

    @property
    def is_connected(self) -> bool:
        return all(arm.is_connected for arm in self._arms)

    @property
    def is_calibrated(self) -> bool:
        return all(arm.is_calibrated for arm in self._arms)

    def calibrate(self) -> None:
        for arm in self._arms:
            arm.calibrate()

    def configure(self) -> None:
        for arm in self._arms:
            arm.configure()

    def connect(self, calibrate: bool = True) -> None:
        for arm in self._arms:
            arm.connect(calibrate=calibrate)

    def get_observation(self) -> dict:
        obs: dict = {}
        for arm in self._arms:
            obs.update(arm.get_observation())
        return obs

    def send_action(self, action: dict) -> dict:
        """Splits the combined action dict by each arm's own
        action_features keys, sends each slice to its own arm, and
        re-combines the (possibly clamped) echoed-back action -- mirrors
        AnvilOpenArmFollower.send_action()'s own return convention
        (actual, post-clamp commanded positions, not the raw input)."""
        sent: dict = {}
        for arm in self._arms:
            arm_keys = arm.action_features.keys()
            sub_action = {k: action[k] for k in arm_keys}
            sent.update(arm.send_action(sub_action))
        return sent

    def clear_heartbeat(self) -> None:
        """Fans out to every arm -- see AnvilOpenArmFollower.clear_heartbeat()'s
        docstring. Called by patches.py's reset patch via
        `self.robot.clear_heartbeat` with no knowledge of how many arms
        `self.robot` actually wraps, so this name/signature must match
        exactly what a bare AnvilOpenArmFollower exposes."""
        for arm in self._arms:
            arm.clear_heartbeat()

    def reseed_heartbeat_after_reset(self) -> None:
        """Fans out to every arm -- see clear_heartbeat()'s docstring for
        why this name/signature must match AnvilOpenArmFollower's."""
        for arm in self._arms:
            arm.reseed_heartbeat_after_reset()

    def disconnect(self) -> None:
        for arm in self._arms:
            arm.disconnect()
