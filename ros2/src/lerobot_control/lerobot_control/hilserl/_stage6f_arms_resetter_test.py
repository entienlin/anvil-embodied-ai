"""Isolated tests for the real home-position reset added after a real
fault: RobotEnv.reset()'s old hold-in-place approach never touched
controller state or intervention bookkeeping, structurally leaving the
untrained policy in sole, unsupervised control of the real arm for
reset_time_s after every episode boundary.

Two things tested, both pure logic, no ROS2/hardware needed:
1. ArmsResetterClient.reset_and_wait()'s polling state machine (accepted
   -> homed, accepted -> failed, not accepted, timeout) -- bypasses
   __init__ (which needs a real rclpy Node) and injects stub
   client/status objects directly.
2. AnvilOpenArmLeader.perform_episode_reset_and_wait()'s orchestration
   (switch-to-teleop -> resetter -> switch-to-policy-and-rebaseline, or
   raise loudly on any failure without rebaselining) -- same
   bypass-connect()-entirely pattern already established in
   _stage5e_switch_retry_test.py.

Run: python3 -m lerobot_control.hilserl._stage6f_arms_resetter_test
"""

import sys
import time

from lerobot_control.hilserl.arms_resetter_client import ArmsResetterClient
from lerobot_control.hilserl.teleop_anvil_openarm_leader import AnvilOpenArmLeader, AnvilOpenArmLeaderConfig


class _FakeFuture:
    def __init__(self, result, done_after_calls: int = 0):
        self._result = result
        self._done_after_calls = done_after_calls
        self._calls = 0

    def done(self) -> bool:
        self._calls += 1
        return self._calls > self._done_after_calls

    def result(self):
        return self._result


class _FakeResetResult:
    def __init__(self, accepted: bool, message: str = ""):
        self.accepted = accepted
        self.message = message


class _FakeRequest:
    dehome = None


class _FakeResetArmsCls:
    Request = _FakeRequest


class _FakeNode:
    def get_logger(self):
        return self

    def warning(self, *_a, **_kw):
        pass

    def error(self, *_a, **_kw):
        pass


class _FakeClient:
    def __init__(self, accepted: bool, owner: ArmsResetterClient, status_on_call: str | None):
        self._accepted = accepted
        self._owner = owner
        self._status_on_call = status_on_call
        self.calls = 0

    def call_async(self, _req):
        self.calls += 1
        # Simulates the status topic's async callback having already
        # fired by the time we start polling -- real reset_and_wait()
        # clears _latest_status right before this call (correctly, to
        # avoid a stale reading from a prior cycle), so a test-only status
        # set BEFORE call_async() would be silently wiped; setting it as
        # this call's own side effect mirrors "the update arrives during
        # the call" without needing a real spinning executor.
        if self._status_on_call is not None:
            self._owner._latest_status = self._status_on_call
        return _FakeFuture(_FakeResetResult(self._accepted, "test"))


def _make_resetter_client(accepted: bool, status_on_call: str | None = None) -> ArmsResetterClient:
    client = ArmsResetterClient.__new__(ArmsResetterClient)
    client._node = _FakeNode()
    client._call_timeout_s = 2.0
    client._latest_status = None
    client._client = _FakeClient(accepted, client, status_on_call)
    client._reset_arms_cls = _FakeResetArmsCls
    return client


def main() -> int:
    print("[stage6f] ArmsResetterClient: accepted -> homed...", flush=True)
    client = _make_resetter_client(accepted=True, status_on_call="homed")
    if client.reset_and_wait(timeout_s=1.0) is not True:
        print("[FAIL] expected True on accepted+homed", flush=True)
        return 1
    print("[OK] accepted+homed -> True", flush=True)

    print("[stage6f] ArmsResetterClient: accepted -> failed...", flush=True)
    client = _make_resetter_client(accepted=True, status_on_call="failed")
    if client.reset_and_wait(timeout_s=1.0) is not False:
        print("[FAIL] expected False on accepted+failed", flush=True)
        return 1
    print("[OK] accepted+failed -> False", flush=True)

    print("[stage6f] ArmsResetterClient: not accepted...", flush=True)
    client = _make_resetter_client(accepted=False)
    if client.reset_and_wait(timeout_s=1.0) is not False:
        print("[FAIL] expected False when the request itself wasn't accepted", flush=True)
        return 1
    if client._client.calls != 1:
        print("[FAIL] expected exactly one service call attempt", flush=True)
        return 1
    print("[OK] not-accepted -> False, no pointless status polling", flush=True)

    print("[stage6f] ArmsResetterClient: accepted but status never reaches a terminal "
          "state -- must time out, not hang...", flush=True)
    client = _make_resetter_client(accepted=True, status_on_call="homing_arms")  # stuck mid-sequence
    start = time.monotonic()
    result = client.reset_and_wait(timeout_s=0.3)
    elapsed = time.monotonic() - start
    if result is not False:
        print("[FAIL] expected False on timeout", flush=True)
        return 1
    if elapsed > 2.0:
        print(f"[FAIL] took {elapsed:.2f}s, expected to bail out near the 0.3s timeout", flush=True)
        return 1
    print(f"[OK] timed out correctly after {elapsed:.2f}s, did not hang", flush=True)

    print("[stage6f] AnvilOpenArmLeader.perform_episode_reset_and_wait(): "
          "success path...", flush=True)

    class _StubSwitchClient:
        def __init__(self):
            self.teleop_calls = 0
            self.policy_calls = 0

        def switch_to_teleop(self):
            self.teleop_calls += 1
            return True

        def switch_to_policy(self):
            self.policy_calls += 1
            return True

    class _StubResetter:
        def __init__(self, ok: bool):
            self.ok = ok
            self.calls = 0

        def reset_and_wait(self, timeout_s):
            self.calls += 1
            return self.ok

    config = AnvilOpenArmLeaderConfig(urdf_path="/workspace/configs/hilserl/urdf/openarm_teleop_kinematics.urdf")
    leader = AnvilOpenArmLeader(config)
    leader._is_connected = True
    leader._switch_client = _StubSwitchClient()
    leader._arms_resetter = _StubResetter(ok=True)
    leader._was_intervening = True  # simulate mid-episode intervention state
    leader._leader_ee_ref = object()
    leader._keyboard._is_intervention = True  # human intervened this episode

    leader.perform_episode_reset_and_wait(timeout_s=5.0)

    if leader._switch_client.teleop_calls != 1:
        print("[FAIL] expected exactly one switch_to_teleop() call before homing", flush=True)
        return 1
    if leader._arms_resetter.calls != 1:
        print("[FAIL] expected exactly one reset_and_wait() call", flush=True)
        return 1
    if leader._switch_client.policy_calls != 1:
        print("[FAIL] expected exactly one switch_to_policy() call after homing succeeded", flush=True)
        return 1
    if leader._was_intervening is not False:
        print("[FAIL] expected _was_intervening reset to False on success", flush=True)
        return 1
    if leader._leader_ee_ref is not None:
        print("[FAIL] expected _leader_ee_ref cleared on success", flush=True)
        return 1
    if leader._keyboard.is_intervention is not False:
        print("[FAIL] expected the keyboard toggle cleared on success, so the next "
              "episode starts as 'policy drives' instead of silently inheriting "
              "the human's intervention", flush=True)
        return 1
    print("[OK] success path: teleop released -> homed -> policy baseline reestablished, "
          "state and keyboard toggle correctly reset", flush=True)

    print("[stage6f] perform_episode_reset_and_wait(): homing failure must raise, "
          "NOT reestablish the policy baseline...", flush=True)
    leader2 = AnvilOpenArmLeader(config)
    leader2._is_connected = True
    leader2._switch_client = _StubSwitchClient()
    leader2._arms_resetter = _StubResetter(ok=False)
    leader2._was_intervening = True
    leader2._keyboard._is_intervention = True

    try:
        leader2.perform_episode_reset_and_wait(timeout_s=5.0)
        print("[FAIL] expected a RuntimeError when homing fails", flush=True)
        return 1
    except RuntimeError as e:
        print(f"[OK] raised as expected: {e}", flush=True)

    if leader2._switch_client.policy_calls != 0:
        print("[FAIL] must NOT reestablish the policy baseline after a failed reset", flush=True)
        return 1
    if leader2._was_intervening is not True:
        print("[FAIL] _was_intervening must be left unchanged on failure, not silently reset", flush=True)
        return 1
    if leader2._keyboard.is_intervention is not True:
        print("[FAIL] keyboard toggle must be left alone on a failed reset", flush=True)
        return 1
    print("[OK] correctly left the arm on teleop_controller, did not silently continue", flush=True)

    print("[stage6f] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
