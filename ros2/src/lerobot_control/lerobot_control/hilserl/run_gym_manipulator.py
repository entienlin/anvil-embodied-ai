"""Entrypoint for HIL-SERL's gym_manipulator, wired up with Anvil's
ROS2-native Robot/Teleoperator classes.

Registration approach: explicit imports for their @RobotConfig/
@TeleoperatorConfig.register_subclass(...) side effects, rather than
relying on lerobot's register_third_party_plugins() distribution-name
autodiscovery -- whether a colcon/ament_python --symlink-install package
surfaces correctly to importlib.metadata.distributions() is unverified and
not worth risking. This means these classes are only reachable through
this entrypoint, not stock lerobot_record/lerobot_teleoperate CLIs, which
is fine since nothing here uses those.

Usage (matches gym_manipulator.py's own CLI exactly, since main() is a
plain draccus @parser.wrap() function underneath):
    ros2 run lerobot_control hilserl_gym_manipulator --config_path <path>
"""

from lerobot_control.hilserl import robot_anvil_openarm  # noqa: F401 -- registration side effect
from lerobot_control.hilserl import teleop_anvil_openarm_leader  # noqa: F401 -- registration side effect


def main() -> None:
    from lerobot.rl.gym_manipulator import main as gym_manipulator_main

    from lerobot_control.hilserl.patches import (
        patch_ee_bounds_and_safety_to_clamp_not_crash,
        patch_inverse_kinematics_to_always_reseed_from_real_joints,
        patch_map_delta_action_for_gripperless_robots,
        patch_robot_env_reset_to_keep_publishing,
    )

    patch_map_delta_action_for_gripperless_robots()
    patch_ee_bounds_and_safety_to_clamp_not_crash()
    patch_inverse_kinematics_to_always_reseed_from_real_joints()
    patch_robot_env_reset_to_keep_publishing()
    gym_manipulator_main()


if __name__ == "__main__":
    main()
