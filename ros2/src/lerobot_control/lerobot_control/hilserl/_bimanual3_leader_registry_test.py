"""Offline test for the teleop_anvil_openarm_leader.py registry fix: the
old `_registered_leader` module-level singleton became `_registered_leaders`,
a dict keyed by ros_prefix, specifically so two AnvilOpenArmLeader
instances (one per arm, for bimanual) can coexist in one process without
the second-to-connect() silently overwriting the first's registration --
see that module's docstring for the real gap this closes and
_check_liveness()'s own history of a confirmed false-positive this
registry feeds into.

Same pure-Python, no-connect(), no-ROS2-daemon pattern as every other
_stageN_*_test.py / _bimanualN_*_test.py file in this package -- this
test pokes the module-level dict directly rather than going through real
connect() calls (which need live ROS2 services), matching
_stage6h_liveness_skip_when_intervening_test.py's own established
approach to testing this exact registry's effect on _check_liveness().

Validates:
  - two leaders registered under different ros_prefix keys both remain
    independently retrievable (the overwrite bug this replaces would have
    silently dropped the first one).
  - get_registered_leader() with no argument is correctly ambiguous
    (returns None) when more than one PLAIN leader is registered and no
    composite is present -- callers that need a specific arm's leader
    must now pass ros_prefix explicitly.
  - get_registered_leader() with no argument prefers a registered
    composite (AnvilOpenArmMultiLeader) over any ambiguous multi-entry
    guess -- this is what makes patches.py's
    patch_robot_env_reset_to_home_via_arms_resetter() (unmodified, still
    calling get_registered_leader() with no args) correctly find the
    bimanual composite's own perform_episode_reset_and_wait() instead of
    either individual sub-leader's.
  - is_any_leader_intervening() correctly reflects a shared/global
    intervention state across however many leaders are registered
    (0, 1, or several) -- the single-arm case (exactly one registered
    leader) must produce byte-for-byte the same answer the old
    `leader._was_intervening or leader._keyboard.is_intervention` check
    gave directly, which _stage6h already covers end-to-end through
    AnvilOpenArmFollower._check_liveness(); this test isolates just the
    registry/helper function itself, plus the new multi-leader scenarios
    _stage6h predates.

Run: python3 -m lerobot_control.hilserl._bimanual3_leader_registry_test
"""

import sys

from lerobot_control.hilserl.teleop_anvil_openarm_leader import (
    MULTI_LEADER_REGISTRY_KEY,
    AnvilOpenArmLeader,
    AnvilOpenArmLeaderConfig,
    _registered_leaders,
    get_registered_leader,
    is_any_leader_intervening,
)


def _check(label: str, cond: bool) -> bool:
    print(f"[{'OK' if cond else 'FAIL'}] {label}", flush=True)
    return cond


def _make_leader(ros_prefix: str) -> AnvilOpenArmLeader:
    leader = AnvilOpenArmLeader(AnvilOpenArmLeaderConfig(ros_prefix=ros_prefix, action_mode="joint_space"))
    leader._is_connected = True
    return leader


class _FakeComposite:
    """Minimal stand-in exercising only what is_any_leader_intervening()/
    get_registered_leader() touch -- AnvilOpenArmMultiLeader itself is
    covered end-to-end by _bimanual2_composite_leader_test.py; this keeps
    the registry test focused on the registry's own dict/lookup logic."""

    def __init__(self):
        self._was_intervening = False

        class _Kb:
            is_intervention = False

        self._keyboard = _Kb()


def main() -> int:
    ok = True
    _registered_leaders.clear()

    print("[bimanual3] two leaders register under distinct ros_prefix keys without colliding...", flush=True)
    leader_l = _make_leader("leader_l")
    _registered_leaders["leader_l"] = leader_l
    leader_r = _make_leader("leader_r")
    _registered_leaders["leader_r"] = leader_r
    ok &= _check("leader_l retrievable by its own prefix", get_registered_leader("leader_l") is leader_l)
    ok &= _check("leader_r retrievable by its own prefix", get_registered_leader("leader_r") is leader_r)
    ok &= _check("both still present (no overwrite)", len(_registered_leaders) == 2)

    print("\n[bimanual3] no-arg lookup is ambiguous (None) with 2 plain leaders and no composite...", flush=True)
    ok &= _check("get_registered_leader() returns None", get_registered_leader() is None)

    print("\n[bimanual3] a registered composite is preferred by the no-arg lookup...", flush=True)
    composite = _FakeComposite()
    _registered_leaders[MULTI_LEADER_REGISTRY_KEY] = composite
    ok &= _check("get_registered_leader() now returns the composite", get_registered_leader() is composite)
    del _registered_leaders[MULTI_LEADER_REGISTRY_KEY]

    print("\n[bimanual3] is_any_leader_intervening(): shared/global intervention across N leaders...", flush=True)
    ok &= _check("nobody intervening -> False", is_any_leader_intervening() is False)
    leader_l._keyboard._is_intervention = True
    ok &= _check("leader_l intervening -> True (global OR)", is_any_leader_intervening() is True)
    leader_l._keyboard._is_intervention = False
    leader_r._was_intervening = True
    ok &= _check(
        "leader_r mid-switch (_was_intervening, keyboard already back to False) -> still True",
        is_any_leader_intervening() is True,
    )
    leader_r._was_intervening = False
    ok &= _check("back to nobody intervening -> False", is_any_leader_intervening() is False)

    print("\n[bimanual3] single-arm case: exactly one registered leader behaves exactly as the old singleton did...", flush=True)
    _registered_leaders.clear()
    single = _make_leader("leader_r")
    _registered_leaders["leader_r"] = single
    ok &= _check("no-arg lookup finds the one leader", get_registered_leader() is single)
    ok &= _check("not intervening -> False", is_any_leader_intervening() is False)
    single._keyboard._is_intervention = True
    ok &= _check("intervening -> True", is_any_leader_intervening() is True)
    single._keyboard._is_intervention = False

    print("\n[bimanual3] no registered leaders at all -> safe empty-registry defaults...", flush=True)
    _registered_leaders.clear()
    ok &= _check("get_registered_leader() -> None", get_registered_leader() is None)
    ok &= _check("get_registered_leader('leader_r') -> None", get_registered_leader("leader_r") is None)
    ok &= _check("is_any_leader_intervening() -> False", is_any_leader_intervening() is False)

    if not ok:
        print("\n[bimanual3] FAILED", flush=True)
        return 1
    print("\n[bimanual3] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
