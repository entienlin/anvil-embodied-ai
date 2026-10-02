"""Catches the exact bug found 2026-10-02 on real hardware: all three
entrypoints (run_gym_manipulator.py, run_actor.py, run_learner.py) had
module-level registration-side-effect imports for the single-arm
robot_anvil_openarm/teleop_anvil_openarm_leader modules, but NOT for
robot_anvil_openarm_multi/teleop_anvil_openarm_multi_leader -- so
draccus.parse() failed with "Couldn't find a choice class for
'anvil_openarm_multi_follower'" the first time anyone actually ran a
bimanual config through a real entrypoint.

Every other _bimanualN_*_test.py constructs AnvilOpenArmMultiFollower/
AnvilOpenArmMultiLeader directly in Python, which works regardless of
whether @RobotConfig.register_subclass(...)'s import-time side effect ever
ran -- none of them would ever have caught this. This test instead
imports each entrypoint MODULE (exactly as `ros2 run lerobot_control
hilserl_actor` etc. would) and confirms the registries resolve the multi
types afterward -- the same path the real bug was in, not a construction
check.

No ROS2 daemon needed (importing these modules doesn't connect to
anything), but does need the real lerobot/rclpy environment (same as every
other test in this package) -- run via run_regression_tests.sh inside the
built image, not on the bare host.
"""

import sys


def _check(label: str, cond: bool) -> bool:
    print(f"[{'OK' if cond else 'FAIL'}] {label}", flush=True)
    return cond


def main() -> int:
    import importlib

    from lerobot.robots.config import RobotConfig
    from lerobot.teleoperators.config import TeleoperatorConfig

    ok = True
    print("[bimanual4] each entrypoint module registers both single-arm and multi types...", flush=True)

    for module_name in [
        "lerobot_control.hilserl.run_gym_manipulator",
        "lerobot_control.hilserl.run_actor",
        "lerobot_control.hilserl.run_learner",
    ]:
        importlib.import_module(module_name)

        robot_cls = RobotConfig.get_choice_class("anvil_openarm_multi_follower")
        ok &= _check(f"{module_name}: anvil_openarm_multi_follower resolves", robot_cls is not None)

        teleop_cls = TeleoperatorConfig.get_choice_class("anvil_openarm_multi_leader")
        ok &= _check(f"{module_name}: anvil_openarm_multi_leader resolves", teleop_cls is not None)

        # Also confirm the single-arm types weren't accidentally dropped --
        # the fix adds imports, it must not remove or shadow the existing ones.
        single_robot_cls = RobotConfig.get_choice_class("anvil_openarm_follower")
        ok &= _check(f"{module_name}: anvil_openarm_follower (single-arm) still resolves", single_robot_cls is not None)

        single_teleop_cls = TeleoperatorConfig.get_choice_class("anvil_openarm_leader")
        ok &= _check(
            f"{module_name}: anvil_openarm_leader (single-arm) still resolves", single_teleop_cls is not None
        )

    print("\n[bimanual4] real bimanual recording config parses end-to-end through draccus...", flush=True)
    # Mirrors the exact failure mode found on real hardware: draccus.parse()
    # against the real YAML, not a hand-built config object.
    sys.argv = ["x", "--config_path", "/workspace/configs/hilserl/env_push_bottle_bimanual_jointspace.yaml"]
    from lerobot.configs import parser
    from lerobot.rl.gym_manipulator import GymManipulatorConfig

    parsed = {}

    @parser.wrap()
    def _parse(cfg: GymManipulatorConfig) -> None:
        parsed["cfg"] = cfg

    _parse()
    cfg = parsed["cfg"]
    ok &= _check("real config: robot.type == anvil_openarm_multi_follower", cfg.env.robot.type == "anvil_openarm_multi_follower")
    ok &= _check("real config: teleop.type == anvil_openarm_multi_leader", cfg.env.teleop.type == "anvil_openarm_multi_leader")
    ok &= _check("real config: robot has 2 arms", len(cfg.env.robot.arms) == 2)
    ok &= _check("real config: teleop has 2 arms", len(cfg.env.teleop.arms) == 2)

    if not ok:
        print("\n[bimanual4] FAILED", flush=True)
        return 1
    print("\n[bimanual4] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
