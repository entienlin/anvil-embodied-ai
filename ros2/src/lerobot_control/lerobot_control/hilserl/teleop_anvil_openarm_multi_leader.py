"""Composite ROS2-native LeRobot Teleoperator driving 1-or-N Anvil OpenArm
leader arms at once (single-arm or bimanual, same class) -- the
Teleoperator-side counterpart to robot_anvil_openarm_multi.py's
AnvilOpenArmMultiFollower. See that module's docstring for the general
"one composite object, N real sub-controllers, N=1 or N=2 today" rationale
and why RobotEnv needs exactly one of these.

Intervention is ONE shared human decision covering every arm at once, not
independently toggled per arm: there is one human at one leader rig moving
both arms with both hands simultaneously, so there is exactly one
KeyboardIntervention instance (this class owns it and shares it with every
sub-AnvilOpenArmLeader it constructs), and "press space" takes over (or
gives back) every arm together. This is a deliberate project decision, not
a limitation worked around here -- see the HIL-SERL bimanual task notes.

This is also the module that makes the teleop_anvil_openarm_leader.py
registry fix (keyed by ros_prefix, see that module's docstring on
_registered_leaders) actually load-bearing: each sub-AnvilOpenArmLeader
still calls its own real connect()/disconnect(), which still registers
itself under its own ros_prefix exactly as a standalone single-arm leader
would -- two of them coexisting in one process is exactly the scenario the
old singleton couldn't handle. This class ADDITIONALLY registers ITSELF
under teleop_anvil_openarm_leader.MULTI_LEADER_REGISTRY_KEY once both
sub-leaders are ready, so that the two existing no-argument
get_registered_leader() call sites (robot_anvil_openarm.py's liveness
check now goes through is_any_leader_intervening() instead, but
patches.py's patch_robot_env_reset_to_home_via_arms_resetter() still calls
get_registered_leader() with no args) transparently get THIS composite
back instead of either individual sub-leader -- no change to patches.py
needed at all, because get_registered_leader() already prefers a
registered composite over any ambiguous multi-entry guess.
"""

from dataclasses import dataclass, field

from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators.utils import TeleopEvents

from .keyboard_intervention import KeyboardIntervention
from .teleop_anvil_openarm_leader import (
    MULTI_LEADER_REGISTRY_KEY,
    AnvilOpenArmLeader,
    AnvilOpenArmLeaderConfig,
    _registered_leaders,
)


def _coerce_arm_config(entry) -> AnvilOpenArmLeaderConfig:
    """See robot_anvil_openarm_multi.py's _coerce_arm_config() -- same
    reasoning: `arms` is a bare `list` (not `list[AnvilOpenArmLeaderConfig]`)
    so YAML-loaded dict entries and directly-constructed config objects
    (the common case in tests) both work without relying on draccus to
    reconstruct nested dataclasses inside a list field."""
    if isinstance(entry, AnvilOpenArmLeaderConfig):
        return entry
    if isinstance(entry, dict):
        return AnvilOpenArmLeaderConfig(**entry)
    raise TypeError(
        f"AnvilOpenArmMultiLeaderConfig.arms entries must be an "
        f"AnvilOpenArmLeaderConfig or a dict of its fields, got {type(entry)!r}"
    )


@TeleoperatorConfig.register_subclass("anvil_openarm_multi_leader")
@dataclass
class AnvilOpenArmMultiLeaderConfig(TeleoperatorConfig):
    """One entry per arm, same list-of-configs convention as
    AnvilOpenArmMultiFollowerConfig.arms. Every entry's action_mode MUST be
    "joint_space" -- see AnvilOpenArmMultiLeader's docstring for why
    "ee_delta" fundamentally cannot generalize to multiple arms (it's tied
    to a single hardcoded end-effector's IK chain) and is rejected at
    construction time rather than silently misbehaving.
    """

    arms: list = field(default_factory=lambda: [AnvilOpenArmLeaderConfig(action_mode="joint_space")])


class AnvilOpenArmMultiLeader(Teleoperator):
    """Owns N AnvilOpenArmLeader sub-controllers (N = len(config.arms)) and
    ONE shared KeyboardIntervention, presenting them as one combined
    Teleoperator to RobotEnv.

    Why joint_space-only: AnvilOpenArmLeaderConfig's "ee_delta" mode
    returns a 3-element {"delta_x","delta_y","delta_z"} dict consumed by
    lerobot's InterventionActionProcessorStep's dict branch, which is
    hardcoded to exactly those 3 keys feeding a single end-effector's
    IK chain (see this task's architecture notes: namespacing EE-delta/IK
    for two independent end-effectors would be real, nontrivial new code
    this project's bimanual joint-space design deliberately avoids).
    "joint_space" mode instead returns a plain ndarray -- the generic
    branch InterventionActionProcessorStep already supports for ANY
    length -- which is exactly what lets this class just concatenate N
    sub-leaders' ndarrays together with no IK involved at all, matching
    the single-arm joint-space design this project already validated on
    real hardware (see teleop_anvil_openarm_leader.py's
    _get_action_joint_space() docstring).
    """

    config_class = AnvilOpenArmMultiLeaderConfig
    name = "anvil_openarm_multi_leader"

    def __init__(self, config: AnvilOpenArmMultiLeaderConfig):
        super().__init__(config)
        self.config = config

        arm_configs = [_coerce_arm_config(a) for a in config.arms]
        if not arm_configs:
            raise ValueError("AnvilOpenArmMultiLeaderConfig.arms must have at least one entry")
        for c in arm_configs:
            if c.action_mode != "joint_space":
                raise ValueError(
                    f"AnvilOpenArmMultiLeader requires every arm's action_mode to be "
                    f"'joint_space' (got {c.action_mode!r} for ros_prefix={c.ros_prefix!r}) -- "
                    "'ee_delta' cannot generalize to multiple arms, see this class's docstring"
                )

        self._keyboard = KeyboardIntervention()
        self._arms: list[AnvilOpenArmLeader] = [
            AnvilOpenArmLeader(c, shared_keyboard=self._keyboard) for c in arm_configs
        ]

        seen_prefixes: set[str] = set()
        for c in arm_configs:
            if c.ros_prefix in seen_prefixes:
                raise ValueError(f"AnvilOpenArmMultiLeader: duplicate ros_prefix {c.ros_prefix!r} across arms")
            seen_prefixes.add(c.ros_prefix)

        self._num_joints_per_arm = [c.num_joints for c in arm_configs]
        self._is_connected = False

    # -- Teleoperator ABC ------------------------------------------------

    @property
    def action_features(self) -> dict:
        return {"dtype": "float32", "shape": (sum(self._num_joints_per_arm),), "names": None}

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return all(arm.is_calibrated for arm in self._arms)

    def calibrate(self) -> None:
        for arm in self._arms:
            arm.calibrate()

    def configure(self) -> None:
        for arm in self._arms:
            arm.configure()

    @property
    def _was_intervening(self) -> bool:
        """True if any sub-arm's own edge-detected intervention state is
        currently True -- see is_any_leader_intervening() in
        teleop_anvil_openarm_leader.py, which reads this attribute on
        whatever is registered (a plain AnvilOpenArmLeader or, here, this
        composite) and ORs it with `._keyboard.is_intervention`. Exposed
        as a property (not a plain attribute set by get_action()) because
        this composite doesn't do its own edge detection -- each
        sub-leader already does its own, correctly, against the one
        shared keyboard -- so this just reports their combined state
        rather than duplicating that bookkeeping."""
        return any(arm._was_intervening for arm in self._arms)

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return
        # This instance (not any individual sub-leader) owns the shared
        # keyboard's lifecycle -- each sub-leader was constructed with
        # shared_keyboard=self._keyboard, which makes its own connect()/
        # disconnect() skip start()/stop() (see AnvilOpenArmLeader.__init__'s
        # _owns_keyboard_lifecycle). Started ONCE here, before any
        # sub-leader connects, so every sub-leader's connect()-time
        # baseline-switch logic can already read a real (idle) keyboard
        # state rather than racing its own startup.
        self._keyboard.start()
        for arm in self._arms:
            arm.connect(calibrate=calibrate)
        _registered_leaders[MULTI_LEADER_REGISTRY_KEY] = self
        self._is_connected = True

    def get_action(self):
        """Concatenates every sub-arm's joint-space action (degrees, each
        arm's own joint order) in config.arms' list order -- must match
        whatever order the env config's `features`/`features_map` declare
        the combined action in (document this in the env YAML, same as
        the single-arm case already documents its 7-joint order).

        Real-hardware finding (2026-10-02): each sub-arm's OWN get_action()
        does its own edge detection and calls ITS OWN SwitchServiceClient
        independently -- the exact same shared-teleop_controller conflict
        perform_episode_reset_and_wait() already hit and fixed
        (_combined_switch()), but THIS is the live per-tick path (every
        control-loop tick, not just between episodes), confirmed directly:
        "Unable to activate controller 'teleop_controller' since the
        command interface 'follower_r_joint1/position' is currently
        claimed by another controller" when leader_l's get_action() tried
        to switch while follower_r_forward_position_controller was still
        active -- 20 consecutive failures, then a hard crash. Since every
        sub-arm shares the SAME KeyboardIntervention (one human, one
        decision), their edge fires on the same tick -- so the fix is the
        same pattern as the reset path: do ONE combined switch here first,
        then pre-sync each sub-arm's internal _was_intervening/
        _leader_ee_ref state to match so ITS OWN get_action() sees no edge
        and never attempts its own independent switch call.
        """
        import numpy as np

        if not self._is_connected:
            raise RuntimeError("AnvilOpenArmMultiLeader is not connected")

        is_intervening = self._keyboard.is_intervention
        any_arm_needs_switch = any(arm._was_intervening != is_intervening for arm in self._arms)
        if any_arm_needs_switch:
            if self._combined_switch(want_teleop=is_intervening):
                for arm in self._arms:
                    arm._pending_switch_retries = 0
                    arm._was_intervening = is_intervening
                    arm._leader_ee_ref = arm._leader_ee_position() if is_intervening else None
            else:
                for arm in self._arms:
                    arm._pending_switch_retries += 1
                    arm._node.get_logger().warning(
                        f"AnvilOpenArmMultiLeader: combined switch to "
                        f"{'teleop' if is_intervening else 'policy'} failed "
                        f"(attempt {arm._pending_switch_retries}/20) -- will retry next tick, "
                        "followers may be unheld until this succeeds"
                    )
                if any(arm._pending_switch_retries >= 20 for arm in self._arms):
                    raise RuntimeError(
                        f"AnvilOpenArmMultiLeader: combined switch to "
                        f"{'teleop' if is_intervening else 'policy'} failed 20 consecutive "
                        "times -- giving up rather than retrying forever with the followers "
                        "in an unknown state."
                    )

        parts = [arm.get_action() for arm in self._arms]
        return np.concatenate([np.asarray(p, dtype=np.float32) for p in parts])

    def get_teleop_events(self) -> dict:
        """Reads the ONE shared keyboard directly rather than delegating to
        any sub-leader's own get_teleop_events(): KeyboardIntervention.
        consume_episode_labels() is a one-shot destructive read (see its
        docstring) -- calling it once per sub-leader per tick would make
        every call after the first silently see already-cleared flags.
        Must be called exactly once per tick, which calling it here
        (instead of forwarding to a sub-leader) guarantees regardless of
        how many arms are configured.
        """
        labels = self._keyboard.consume_episode_labels()
        return {
            TeleopEvents.IS_INTERVENTION: self._keyboard.is_intervention,
            TeleopEvents.SUCCESS: labels["success"],
            TeleopEvents.FAILURE: labels["failure"],
            TeleopEvents.RERECORD_EPISODE: labels["rerecord"],
            TeleopEvents.TERMINATE_EPISODE: labels["success"] or labels["failure"] or labels["rerecord"],
        }

    def send_feedback(self, feedback: dict) -> None:
        pass

    def _combined_switch(self, want_teleop: bool) -> bool:
        """Single switch_controller call covering every arm's own policy
        controller together with the ONE shared teleop_controller.

        Real-hardware finding (2026-10-02): teleop_controller is a single
        controller instance spanning every configured arm slot (confirmed
        by the devbox directly) -- activating it requires ALL its arms'
        command interfaces to be free at that moment, not just the one
        arm we're nominally switching. Per-arm independent switch_to_teleop()
        calls (looping arm by arm, each a separate switch_controller call)
        hit this directly: switching leader_l to teleop while
        follower_r_forward_position_controller is still active fails with
        "command interface 'follower_r_joint1/position' is currently
        claimed by another controller", even though the call never
        mentions follower_r at all -- activating the shared controller
        implicitly needs its interfaces too. The reverse direction
        (give-back) happens to work via a per-arm loop today, since
        deactivating the shared controller on the FIRST arm's call already
        frees every arm's interfaces for the rest -- but this method
        handles both directions the same, explicit way rather than relying
        on that asymmetry.

        Uses arm[0]'s SwitchServiceClient for the actual service call
        (any arm's client works, all point at the same controller_manager)
        -- same "any arm's client will do" reasoning already used for the
        shared ArmsResetterClient call below.
        """
        primary = self._arms[0]._switch_client
        states = primary._get_controller_states()
        if states is None:
            return False

        teleop_name = primary._teleop_controller
        policy_names = [arm._switch_client._policy_controller for arm in self._arms]

        if want_teleop:
            activate = [] if states.get(teleop_name) == "active" else [teleop_name]
            deactivate = [p for p in policy_names if states.get(p) == "active"]
        else:
            activate = [p for p in policy_names if states.get(p) != "active"]
            deactivate = [] if states.get(teleop_name) != "active" else [teleop_name]

        if not activate and not deactivate:
            return True
        return primary._call_switch(activate=activate, deactivate=deactivate)

    def perform_episode_reset_and_wait(
        self,
        timeout_s: float = 15.0,
        before_homing=None,
        after_homed=None,
    ) -> None:
        """Bimanual/multi-arm episode reset, called by patches.py's
        patch_robot_env_reset_to_home_via_arms_resetter() via
        get_registered_leader() -- same orchestration AnvilOpenArmLeader.
        perform_episode_reset_and_wait() does for one arm, generalized to
        every registered arm, with ONE real difference that matters:

        The devbox's real /arms_resetter/reset service homes whatever arms
        are in the ACTIVE PROFILE, with no per-arm selection in its own
        API (confirmed in this project's earlier single-arm work, carried
        forward here as a design assumption for the bimanual profile,
        since no per-arm resetter API exists to even ask for one arm
        only). So this calls reset_and_wait() exactly ONCE (via the first
        arm's ArmsResetterClient -- any arm's client would do, since they
        all point at the same service), not once per arm: calling it twice
        would either double-home the whole profile or race two concurrent
        homing sequences against the same hardware, neither of which is
        anything the resetter's API is built to support.

        Both switches (to teleop before homing, back to policy after) go
        through _combined_switch() -- a single switch_controller call
        covering every arm's own policy controller plus the one shared
        teleop_controller -- not independent per-arm calls. See
        _combined_switch()'s own docstring for the real conflict this
        avoids.
        """
        if before_homing is not None:
            before_homing()

        for arm in self._arms:
            arm._reset_in_progress_unsafe = True
        if not self._combined_switch(want_teleop=True):
            raise RuntimeError(
                "AnvilOpenArmMultiLeader: failed to switch to teleop_controller before "
                "homing -- aborting before calling the real arms_resetter."
            )

        if not self._arms[0]._arms_resetter.reset_and_wait(timeout_s=timeout_s):
            raise RuntimeError(
                "AnvilOpenArmMultiLeader: episode reset failed -- arms_resetter did not "
                "report 'homed' for the active (bimanual) profile. Leaving every arm on "
                "teleop_controller rather than re-engaging policy control in an unknown state."
            )

        if after_homed is not None:
            after_homed()

        if not self._combined_switch(want_teleop=False):
            raise RuntimeError(
                "AnvilOpenArmMultiLeader: homing succeeded but switching back to "
                "policy control afterward failed -- see the error above."
            )

        for arm in self._arms:
            arm._reset_in_progress_unsafe = False
            arm._was_intervening = False
            arm._leader_ee_ref = None
        self._keyboard.reset_intervention()

    def disconnect(self) -> None:
        if not self._is_connected:
            return
        if _registered_leaders.get(MULTI_LEADER_REGISTRY_KEY) is self:
            del _registered_leaders[MULTI_LEADER_REGISTRY_KEY]
        for arm in self._arms:
            arm.disconnect()
        # This instance owns the shared keyboard's lifecycle (see connect()) --
        # stop it only after every sub-leader (which may still read
        # self._keyboard.is_intervention during its own disconnect()-time
        # give-back logic) has fully disconnected.
        self._keyboard.stop()
        self._is_connected = False
