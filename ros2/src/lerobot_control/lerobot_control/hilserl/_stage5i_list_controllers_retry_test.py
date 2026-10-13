"""Isolated test for SwitchServiceClient._get_controller_states()'s retry
behavior, added after a real list_controllers timeout (2s, previously zero
retry) killed 3 actor sessions in a row at a real episode-boundary reset
-- alongside live nvidia-smi evidence of other, unrelated users' jobs
actively competing for the same GPU this pipeline uses. Whatever causes
any ONE slow response, a single miss should not cost a whole session.

No real ROS2 service/hardware needed -- a fake _list_client whose
call_async() returns futures we control directly, so this exercises the
real retry loop in _get_controller_states() without needing controller_manager
or a devbox.

Run: python3 -m lerobot_control.hilserl._stage5i_list_controllers_retry_test
"""

import sys

import rclpy
from rclpy.node import Node

from lerobot_control.hilserl.switch_service_client import SwitchServiceClient


class _FakeFuture:
    def __init__(self, result=None, never_completes=False):
        self._result = result
        self._never_completes = never_completes

    def done(self) -> bool:
        return not self._never_completes

    def result(self):
        return self._result


class _FakeController:
    def __init__(self, name: str, state: str):
        self.name = name
        self.state = state


class _FakeListResult:
    def __init__(self, controllers):
        self.controller = controllers


class _FlakyListClient:
    def __init__(self, fail_count: int):
        """Times out (never-completing future) the first `fail_count`
        calls, then returns a real result forever after. fail_count=-1
        means always time out."""
        self.fail_count = fail_count
        self.calls = 0

    def call_async(self, request):
        self.calls += 1
        if self.fail_count < 0 or self.calls <= self.fail_count:
            return _FakeFuture(never_completes=True)
        return _FakeFuture(result=_FakeListResult([_FakeController("teleop_controller", "active")]))


def main() -> int:
    rclpy.init(args=None)
    node = Node("stage5i_test_node")

    print("[stage5i] 2 timeouts then success -- must retry and succeed on "
          "the 3rd attempt, not give up after the first...", flush=True)
    client = SwitchServiceClient(node, call_timeout_s=0.05)
    client._list_client = _FlakyListClient(fail_count=2)
    states = client._get_controller_states()
    assert states == {"teleop_controller": "active"}, states
    assert client._list_client.calls == 3, client._list_client.calls
    print("[OK] retried through 2 timeouts, succeeded on attempt 3", flush=True)

    print("[stage5i] always times out -- must give up after exactly "
          "_LIST_CONTROLLERS_MAX_ATTEMPTS attempts, returning None...", flush=True)
    client2 = SwitchServiceClient(node, call_timeout_s=0.05)
    client2._list_client = _FlakyListClient(fail_count=-1)
    states2 = client2._get_controller_states()
    assert states2 is None, states2
    assert client2._list_client.calls == SwitchServiceClient._LIST_CONTROLLERS_MAX_ATTEMPTS, (
        client2._list_client.calls
    )
    print(f"[OK] gave up after exactly {SwitchServiceClient._LIST_CONTROLLERS_MAX_ATTEMPTS} attempts", flush=True)

    print("[stage5i] first attempt succeeds -- must not retry unnecessarily...", flush=True)
    client3 = SwitchServiceClient(node, call_timeout_s=0.05)
    client3._list_client = _FlakyListClient(fail_count=0)
    states3 = client3._get_controller_states()
    assert states3 == {"teleop_controller": "active"}, states3
    assert client3._list_client.calls == 1, client3._list_client.calls
    print("[OK] succeeded on the first attempt, no unnecessary retry", flush=True)

    rclpy.shutdown()
    print("\n[stage5i] PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
