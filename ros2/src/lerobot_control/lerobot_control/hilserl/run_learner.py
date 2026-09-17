"""Entrypoint for HIL-SERL's learner process, wired up with Anvil's
ROS2-native Robot/Teleoperator classes -- see run_actor.py's docstring for
the full story on why this matters even here.

The learner itself trains the SAC policy from transitions streamed over
gRPC from the actor -- it does NOT construct/connect to the real robot
(make_policy(env_cfg=cfg.env) only reads the env config's declared
feature shapes, it doesn't instantiate the environment). But it still
draccus-parses the SAME config file as the actor (the full
TrainRLServerPipelineConfig, including env.robot.type/env.teleop.type),
and draccus needs AnvilOpenArmFollowerConfig/AnvilOpenArmLeaderConfig
registered via @RobotConfig.register_subclass(...)/
@TeleoperatorConfig.register_subclass(...) to resolve those `type:`
discriminators at PARSE time -- without this import, --config_path would
fail to parse at all, even though the learner never touches real hardware.
Applying the same patches here too for consistency, even though most of
them (EEBoundsAndSafety, InverseKinematicsRLStep, RobotEnv.reset) target
code paths the learner process itself never executes -- harmless either
way (idempotent, no-ops if the patched classes are never used) and keeps
this entrypoint symmetric with run_actor.py rather than requiring anyone
to reason about which patches are "actually needed" here.

Usage:
    ros2 run lerobot_control hilserl_learner --config_path <path>
"""

from lerobot_control.hilserl import robot_anvil_openarm  # noqa: F401 -- registration side effect
from lerobot_control.hilserl import teleop_anvil_openarm_leader  # noqa: F401 -- registration side effect


def main() -> None:
    from lerobot.rl.learner import train_cli

    from lerobot_control.hilserl.patches import apply_all_patches

    apply_all_patches()
    train_cli()


if __name__ == "__main__":
    main()
