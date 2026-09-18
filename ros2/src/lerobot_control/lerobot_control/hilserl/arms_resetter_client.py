"""Client for the devbox's /arms_resetter/reset service -- drives a real
home-position reset for the leader_r/follower_r arm pair between HIL-SERL
episodes, replacing the old hold-in-place reset.

anvil_msgs (ResetArms.srv, ArmsResetStatus.msg) is a local, minimal
reproduction of the devbox's real package -- see
ros2/src/anvil_msgs/package.xml for why this gives real cross-machine
type-hash interop, not just a lookalike.

Devbox-confirmed facts this client is built against (not guessed):
- The service call itself is a fire-and-forget accept/reject ack, NOT a
  completion signal -- real completion is reported asynchronously on
  /arms_resetter/reset_status (state: "deactivating_controllers" ->
  "homing_arms" -> "setting_runtime_gains" -> "activating_controllers"
  -> "homed", or "failed" if it can't claim the interfaces it needs).
- follower_r_forward_position_controller MUST be released (not merely
  "expected to be") before every call -- the resetter only ever
  deactivates teleop_controller itself, has zero awareness of our
  manually-loaded forward_position_controller. If it's active, the
  resetter's own JTC activation fails, reset_status goes to "failed",
  and no motion is attempted (fails safe) -- this client does NOT do
  that release itself (no controller-switching capability of its own),
  see AnvilOpenArmLeader.perform_episode_reset_and_wait() for the full
  orchestration.
- Always ends with teleop_controller active regardless of prior state
  (fixed by the profile's control_mode) -- reestablishing
  forward_position_controller afterward is the caller's job too.
- Typical duration ~4-5s (3-waypoint trajectory + gains ramp), not a
  guaranteed constant.
"""

import time

from rclpy.node import Node


class ArmsResetterClient:
    def __init__(
        self,
        node: Node,
        reset_service_name: str = "/arms_resetter/reset",
        status_topic: str = "/arms_resetter/reset_status",
        call_timeout_s: float = 5.0,
    ):
        self._node = node
        self._call_timeout_s = call_timeout_s
        self._latest_status: str | None = None

        from anvil_msgs.msg import ArmsResetStatus  # deferred: ROS2-only import
        from anvil_msgs.srv import ResetArms  # deferred: ROS2-only import

        self._reset_arms_cls = ResetArms
        self._client = node.create_client(ResetArms, reset_service_name)
        node.create_subscription(ArmsResetStatus, status_topic, self._on_status, 10)

    def _on_status(self, msg) -> None:
        self._latest_status = msg.state

    def wait_for_service(self, timeout_s: float = 10.0) -> bool:
        return self._client.wait_for_service(timeout_sec=timeout_s)

    def reset_and_wait(self, timeout_s: float = 15.0) -> bool:
        """Calls /arms_resetter/reset (dehome=false) and blocks until
        reset_status reports "homed" or "failed", or timeout_s elapses.
        Returns True only on a confirmed "homed" -- everything else
        (rejected request, "failed" status, timeout) returns False, never
        raises, so the caller (which knows the full episode-reset context)
        decides how loud to be about it.

        Relies on something else already spinning `node`'s executor (this
        class creates no thread of its own) -- matches
        SwitchServiceClient's exact pattern, safe here because this is
        always constructed with AnvilOpenArmLeader's own node, already
        spun by its background _spin_loop thread.
        """
        self._latest_status = None  # clear any stale reading from a previous cycle

        req = self._reset_arms_cls.Request()
        req.dehome = False
        future = self._client.call_async(req)
        deadline = time.monotonic() + self._call_timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            self._node.get_logger().error("ArmsResetterClient: reset service call timed out")
            return False

        result = future.result()
        if not result.accepted:
            self._node.get_logger().error(
                f"ArmsResetterClient: reset request not accepted: {result.message}"
            )
            return False

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._latest_status == "homed":
                return True
            if self._latest_status == "failed":
                self._node.get_logger().error(
                    "ArmsResetterClient: reset_status reported 'failed' -- most likely "
                    "follower_r_forward_position_controller wasn't released before this "
                    "call (see this class's docstring)"
                )
                return False
            time.sleep(0.05)

        self._node.get_logger().error(
            f"ArmsResetterClient: timed out after {timeout_s}s waiting for 'homed' "
            f"(last status: {self._latest_status!r})"
        )
        return False
