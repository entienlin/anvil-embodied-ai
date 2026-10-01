"""Entrypoint for HIL-SERL's actor process, wired up with Anvil's
ROS2-native Robot/Teleoperator classes -- same registration/patch story as
run_gym_manipulator.py (see that file's docstring), applied here too
because lerobot.rl.actor reuses gym_manipulator.py's make_robot_env()/
make_processors() pipeline directly (confirmed via source read: actor.py
imports those functions from gym_manipulator itself, not a separate
implementation), so every patch that matters for recording matters here
too -- more so, since the actor publishes an untrained policy's raw output
to the real robot from its very first control step.

actor.py/learner.py have no console-script entry points in lerobot itself
(unlike lerobot-record etc.) -- they're invoked as `python -m
lerobot.rl.actor`/`python -m lerobot.rl.learner`. This wraps that same
invocation with our registration + patches applied first.

Config: a TrainRLServerPipelineConfig, NOT the GymManipulatorConfig shape
hilserl_gym_manipulator uses -- env.type: gym_manipulator must be set
explicitly (TrainRLServerPipelineConfig.env is a polymorphic EnvConfig
field, unlike GymManipulatorConfig.env's concrete HILSerlRobotEnvConfig).
See configs/hilserl/train_rl_push_bottle_smoketest.yaml.

Usage:
    ros2 run lerobot_control hilserl_actor --config_path <path>
"""

from lerobot_control.hilserl import robot_anvil_openarm  # noqa: F401 -- registration side effect
from lerobot_control.hilserl import teleop_anvil_openarm_leader  # noqa: F401 -- registration side effect


def main() -> None:
    from lerobot.rl.actor import actor_cli

    from lerobot_control.hilserl.patches import apply_all_patches
    from lerobot_control.hilserl.session_provenance import log_session_provenance

    log_session_provenance("actor")
    apply_all_patches()
    actor_cli()


if __name__ == "__main__":
    main()
