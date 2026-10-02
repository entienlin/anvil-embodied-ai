"""HIL-SERL integration: ROS2-native Robot/Teleoperator bridge for the
Anvil OpenArm rig, driving the follower over the existing ROS2 command
topics rather than lerobot's native CAN-based openarm_follower classes
(which this project's two-PC architecture can't use directly).

Re-exports AnvilOpenArmFollower/AnvilOpenArmLeader here deliberately, not
just for convenience: lerobot.utils.import_utils.make_device_from_device_class
(the fallback path robots/utils.py::make_robot_from_config uses for any
config.type not in its own hardcoded if/elif chain) resolves the actual
Robot/Teleoperator class by deriving a device name from the Config class
name (stripping "Config") and searching a short list of candidate modules
-- the FIRST candidate it tries is this package's own __init__ namespace.
Registering the *Config* classes via @RobotConfig.register_subclass(...)
(done in the respective files, for draccus's config-parsing registry) is a
completely separate mechanism from this one and does not by itself make the
Robot/Teleoperator classes discoverable here -- confirmed the hard way via
an ImportError during Stage 3 testing before this re-export was added.
"""

from .robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig  # noqa: F401
from .robot_anvil_openarm_multi import AnvilOpenArmMultiFollower, AnvilOpenArmMultiFollowerConfig  # noqa: F401
from .teleop_anvil_openarm_leader import AnvilOpenArmLeader, AnvilOpenArmLeaderConfig  # noqa: F401
from .teleop_anvil_openarm_multi_leader import AnvilOpenArmMultiLeader, AnvilOpenArmMultiLeaderConfig  # noqa: F401
