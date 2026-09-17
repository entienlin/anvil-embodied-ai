"""Isolated test for patch_robot_env_reset_to_keep_publishing(), added
after the 4th real "follower dead" occurrence was localized (via the
devbox's motor-status logging) to ~5.3s after
follower_r_forward_position_controller activation -- squarely inside
RobotEnv.reset()'s blocking sleep window, during which zero commands are
published to the follower's command topic. Leading hypothesis: a
command-timeout watchdog in the motor firmware.

No real hardware needed -- constructs a bare RobotEnv instance via
__new__ (skips __init__, which needs actual hardware) with just the
attributes RobotEnv.reset() actually touches, and calls reset() directly
against it, both before and after patching.

Run: python3 -m lerobot_control.hilserl._stage5h_reset_keepalive_patch_test
"""

import sys
import time

from lerobot.rl.gym_manipulator import RobotEnv

from lerobot_control.hilserl.patches import patch_robot_env_reset_to_keep_publishing


class _StubRobot:
    def __init__(self):
        self.send_action_calls: list[dict] = []
        self._pos = {"j1.pos": 12.0, "j2.pos": -5.0}

    def get_observation(self) -> dict:
        return dict(self._pos)

    def send_action(self, action: dict) -> dict:
        self.send_action_calls.append(dict(action))
        return action


def _make_fake_env(reset_time_s: float) -> RobotEnv:
    """A real RobotEnv instance, built via __new__ to skip __init__ (which
    needs actual hardware) -- passes isinstance checks so the original
    method's super().reset(...) call resolves correctly, unlike a plain
    duck-typed stub. Only sets the attributes RobotEnv.reset() actually
    reads/writes."""
    env = RobotEnv.__new__(RobotEnv)
    env.reset_pose = None
    env.reset_time_s = reset_time_s
    env._joint_names = ["j1", "j2"]
    env.robot = _StubRobot()
    env.current_step = 1
    env.episode_data = "stale"
    env._get_observation = lambda: dict(env.robot._pos)
    return env


def main() -> int:
    reset_time_s = 0.5  # short, so the test runs fast

    print("[stage5h] UNPATCHED: confirming the original reset() really "
          "does publish zero commands during the sleep (sanity-checks "
          "the diagnosis)...", flush=True)
    env = _make_fake_env(reset_time_s)
    t0 = time.perf_counter()
    obs, info = RobotEnv.reset(env)
    elapsed = time.perf_counter() - t0
    if env.robot.send_action_calls:
        print(f"[FAIL] unpatched reset() should publish nothing, got "
              f"{len(env.robot.send_action_calls)} calls", flush=True)
        return 1
    if elapsed < reset_time_s * 0.8:
        print(f"[FAIL] unpatched reset() should still take ~{reset_time_s}s, took {elapsed:.2f}s", flush=True)
        return 1
    print(f"[OK] unpatched: 0 send_action() calls during a {elapsed:.2f}s reset "
          "-- confirms the silent gap", flush=True)

    print("[stage5h] applying patch...", flush=True)
    patch_robot_env_reset_to_keep_publishing()

    print("[stage5h] PATCHED: reset() must keep publishing throughout, "
          "same total duration...", flush=True)
    env2 = _make_fake_env(reset_time_s)
    t0 = time.perf_counter()
    obs2, info2 = RobotEnv.reset(env2)
    elapsed2 = time.perf_counter() - t0

    n_calls = len(env2.robot.send_action_calls)
    expected_min_calls = int(reset_time_s / 0.1) - 1  # ~0.1s cadence, allow slop
    if n_calls < expected_min_calls:
        print(f"[FAIL] expected at least ~{expected_min_calls} send_action() calls "
              f"over {reset_time_s}s at 100ms cadence, got {n_calls}", flush=True)
        return 1
    if elapsed2 < reset_time_s * 0.8:
        print(f"[FAIL] patched reset() should still take ~{reset_time_s}s, took {elapsed2:.2f}s", flush=True)
        return 1
    # Every call must be the SAME hold position (not drifting/changing).
    first_call = env2.robot.send_action_calls[0]
    if not all(c == first_call for c in env2.robot.send_action_calls):
        print(f"[FAIL] hold action changed between calls: {env2.robot.send_action_calls}", flush=True)
        return 1
    if first_call != {"j1.pos": 12.0, "j2.pos": -5.0}:
        print(f"[FAIL] hold action doesn't match the robot's real observed position: {first_call}", flush=True)
        return 1
    print(f"[OK] patched: {n_calls} send_action() calls over {elapsed2:.2f}s, "
          f"all publishing the same real hold position {first_call} -- no more silent gap", flush=True)

    # Return-value/state-reset behavior must be unchanged.
    if env2.current_step != 0 or env2.episode_data is not None:
        print("[FAIL] episode tracking state wasn't reset correctly", flush=True)
        return 1
    if obs2 != {"j1.pos": 12.0, "j2.pos": -5.0}:
        print(f"[FAIL] unexpected observation returned: {obs2}", flush=True)
        return 1
    print("[OK] episode tracking reset and returned observation unchanged from original behavior", flush=True)

    print("[stage5h] re-applying patch must be a no-op (idempotent)...", flush=True)
    before = RobotEnv.reset
    patch_robot_env_reset_to_keep_publishing()
    if RobotEnv.reset is not before:
        print("[FAIL] re-applying the patch replaced the method again", flush=True)
        return 1
    print("[OK] idempotent", flush=True)

    print("[stage5h] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
