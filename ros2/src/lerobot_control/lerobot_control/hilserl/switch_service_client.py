"""Client for controller_manager/switch_controller, used to hand control of
the follower_r arm between the GPU-side policy (forward-position
controller) and the physical leader arm (teleop mirror controller).

Calls the standard ros2_control service directly -- no new devbox-hosted
service, per the plan's decision to reuse what already exists rather than
add devbox-side surface area. controls_owner_manager is NOT used here; it's
an unrelated, fixed 3-way arbitration with no slot for this.

Field names and the STRICT=2 constant were verified directly against
controller_manager_msgs/srv/SwitchController at Stage 1 (`ros2 interface
show`) -- all matched what was originally assumed from the stable,
cross-distro ros2_control API. Cross-machine service calls themselves were
also confirmed working against real hardware at Stage 1
(list_controllers). What Stage 1 could NOT catch (read-only, no actual
switch attempted): the real first switch attempt (Stage 5) revealed that
hardcoding a fixed "always activate X, deactivate Y" request breaks
STRICT switching the moment actual controller state doesn't match that
assumption -- see _switch_to()'s docstring.
"""

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

STRICT = 2  # controller_manager_msgs/srv/SwitchController.STRICT -- verified at Stage 1


class SwitchServiceClient:
    """Switches follower_r between its forward-position controller (policy
    control) and its teleop mirror controller (leader-arm control), via the
    devbox's standard controller_manager/switch_controller service.
    """

    def __init__(
        self,
        node: Node,
        policy_controller: str = "follower_r_forward_position_controller",
        teleop_controller: str = "teleop_controller",
        command_topic: str = "/follower_r_forward_position_controller/commands",
        joint_state_topic: str = "/joint_states",
        joint_names: list[str] | None = None,
        service_name: str = "/controller_manager/switch_controller",
        call_timeout_s: float = 2.0,
    ):
        self._node = node
        self._policy_controller = policy_controller
        self._teleop_controller = teleop_controller
        self._joint_names = joint_names or []
        self._call_timeout_s = call_timeout_s

        self._pub = node.create_publisher(Float64MultiArray, command_topic, 10)
        self._latest_joint_state: JointState | None = None
        node.create_subscription(JointState, joint_state_topic, self._on_joint_state, 10)

        from controller_manager_msgs.srv import ListControllers, SwitchController  # deferred: ROS2-only import

        self._client = node.create_client(SwitchController, service_name)
        self._switch_controller_cls = SwitchController

        list_service_name = service_name.rsplit("/", 1)[0] + "/list_controllers"
        self._list_client = node.create_client(ListControllers, list_service_name)

    def _on_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg

    def wait_for_service(self, timeout_s: float = 10.0) -> bool:
        return self._client.wait_for_service(timeout_sec=timeout_s) and self._list_client.wait_for_service(
            timeout_sec=timeout_s
        )

    def _get_controller_states(self) -> dict[str, str] | None:
        """{controller_name: state} via list_controllers, or None on failure."""
        from controller_manager_msgs.srv import ListControllers  # deferred: ROS2-only import

        future = self._list_client.call_async(ListControllers.Request())
        deadline = time.monotonic() + self._call_timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not future.done():
            self._node.get_logger().error("SwitchServiceClient: list_controllers call timed out")
            return None
        return {c.name: c.state for c in future.result().controller}

    def has_follower_joint_state(self) -> bool:
        """Whether at least one /joint_states message has arrived yet.

        switch_to_policy()'s bumpless-transfer pre-publish checks
        _latest_joint_state synchronously, with no retry -- callers doing a
        connect()-time baseline switch should wait for this to be True
        first (requires the node to already be spinning), or the very
        first switch silently skips bumpless transfer.
        """
        return self._latest_joint_state is not None

    def switch_to_policy(self) -> bool:
        """Hand control of follower_r back to the GPU-side policy.

        Bumpless transfer: publish the arm's current position once on the
        forward-position command topic BEFORE activating that controller,
        so it doesn't snap to whatever stale target was last commanded
        before the teleop interlude.
        """
        # rclpy.shutdown() can be triggered from elsewhere in the process
        # (lerobot's own SIGINT handling, unrelated to our own
        # disconnect()) -- reproduced on real hardware: this call, invoked
        # from AnvilOpenArmLeader.disconnect()'s give-back, hit "publisher's
        # context is invalid" mid-shutdown. The give-back is moot anyway if
        # rclpy is already down (nothing can be published/serviced), so
        # fail cleanly instead of raising into an atexit callback.
        if not rclpy.ok():
            return False

        if self._latest_joint_state is not None and self._joint_names:
            name_to_pos = dict(zip(self._latest_joint_state.name, self._latest_joint_state.position, strict=False))
            if all(j in name_to_pos for j in self._joint_names):
                msg = Float64MultiArray()
                msg.data = [name_to_pos[j] for j in self._joint_names]
                self._pub.publish(msg)
            else:
                self._node.get_logger().warning(
                    "SwitchServiceClient.switch_to_policy(): missing joints in latest "
                    "/joint_states, skipping bumpless-transfer pre-publish"
                )
        return self._switch_to(want_active=self._policy_controller, want_inactive=self._teleop_controller)

    def switch_to_teleop(self) -> bool:
        """Hand control of follower_r to the leader arm."""
        return self._switch_to(want_active=self._teleop_controller, want_inactive=self._policy_controller)

    def _switch_to(self, want_active: str, want_inactive: str) -> bool:
        """Build a minimal activate/deactivate request from ACTUAL current
        state, rather than assuming a fixed prior state.

        Real-hardware finding (Stage 5): a hardcoded "always activate X,
        always deactivate Y" request fails STRICT switching outright the
        moment reality doesn't match that assumption -- confirmed directly:
        "Controller with name 'teleop_controller' can not be deactivated
        since it is not active" when it had (unexpectedly) already gone
        inactive before this call. Querying real state first and omitting
        already-satisfied transitions makes this idempotent against
        whatever the actual state turns out to be.
        """
        states = self._get_controller_states()
        if states is None:
            self._node.get_logger().error(
                f"SwitchServiceClient: could not determine current controller "
                f"states before switching (want_active={want_active}, "
                f"want_inactive={want_inactive})"
            )
            return False

        activate = [] if states.get(want_active) == "active" else [want_active]
        deactivate = [] if states.get(want_inactive) != "active" else [want_inactive]

        if not activate and not deactivate:
            self._node.get_logger().info(
                f"SwitchServiceClient: {want_active} already active, {want_inactive} "
                "already inactive -- nothing to switch"
            )
            return True

        return self._call_switch(activate=activate, deactivate=deactivate)

    def _call_switch(self, activate: list[str], deactivate: list[str]) -> bool:
        req = self._switch_controller_cls.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = STRICT
        req.activate_asap = True

        future = self._client.call_async(req)
        deadline = time.monotonic() + self._call_timeout_s
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.01)

        if not future.done():
            self._node.get_logger().error(
                f"SwitchServiceClient: switch_controller call timed out "
                f"(activate={activate}, deactivate={deactivate})"
            )
            return False

        result = future.result()
        ok = bool(getattr(result, "ok", False))
        if not ok:
            self._node.get_logger().error(
                f"SwitchServiceClient: switch_controller reported failure "
                f"(activate={activate}, deactivate={deactivate}): "
                f"{getattr(result, 'message', '<no message field>')}"
            )
        return ok
