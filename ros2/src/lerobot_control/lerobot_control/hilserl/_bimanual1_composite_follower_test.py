"""Offline test for AnvilOpenArmMultiFollower (robot_anvil_openarm_multi.py),
the composite Robot that drives 1-or-N AnvilOpenArmFollower sub-arms at
once for single-arm or bimanual control.

Mirrors the established pattern in this package's other _stageN_*_test.py
files (e.g. _stage6e_liveness_watchdog_test.py, _stage4_leader_test.py):
pure Python construction, no connect(), no rclpy.init(), no ROS2 daemon,
no real/fake hardware. Only real dependency is this package's own
imports (lerobot, rclpy, numpy) -- same requirement every existing stage
test already has.

Validates:
  - a 1-arm AnvilOpenArmMultiFollowerConfig produces observation_features/
    action_features/bus.motors/cameras IDENTICAL to a bare
    AnvilOpenArmFollower constructed directly -- the behavioral-equivalence
    property that makes this composite class a safe superset of the
    single-arm class rather than a second, divergent implementation (see
    this module's own docstring, and robot_anvil_openarm_multi.py's, for
    why that matters for the "make arm count a config knob later" plan).
  - a 2-arm (bimanual) config produces the correct 14-dim joint
    observation/action features and 4 camera observation features, in
    left-then-right joint order.
  - duplicate camera names or duplicate ros_prefixes across arms raise
    loudly at construction instead of silently colliding in the merged
    observation/action dicts.
  - get_observation()/send_action()/clear_heartbeat()/
    reseed_heartbeat_after_reset() correctly fan out to (and merge results
    from) every sub-arm, using lightweight stub sub-arms so this stays a
    pure computation test independent of AnvilOpenArmFollower's own ROS2
    plumbing (already covered by that class's own existing tests).

Run: python3 -m lerobot_control.hilserl._bimanual1_composite_follower_test
"""

import sys

from lerobot_control.hilserl.robot_anvil_openarm import AnvilOpenArmFollower, AnvilOpenArmFollowerConfig
from lerobot_control.hilserl.robot_anvil_openarm_multi import (
    AnvilOpenArmMultiFollower,
    AnvilOpenArmMultiFollowerConfig,
)


def _check(label: str, cond: bool) -> bool:
    print(f"[{'OK' if cond else 'FAIL'}] {label}", flush=True)
    return cond


class _StubArm:
    """Lightweight stand-in for AnvilOpenArmFollower -- used only for the
    fan-out/merge tests below, which care about AnvilOpenArmMultiFollower's
    OWN logic (splitting/merging dicts across sub-arms), not about
    AnvilOpenArmFollower's ROS2/clamping/watchdog internals (those already
    have their own dedicated tests, e.g. _stage6e/_stage7e)."""

    def __init__(self, prefix: str, cams: dict):
        self.joint_names = [f"{prefix}_joint{i}" for i in range(1, 8)]
        self.config = type("C", (), {"camera_topics": cams})()
        self.sent = None
        self.cleared = False
        self.reseeded = False

    @property
    def observation_features(self):
        feats = {f"{j}.pos": float for j in self.joint_names}
        for c in self.config.camera_topics:
            feats[c] = (480, 640, 3)
        return feats

    @property
    def action_features(self):
        return {f"{j}.pos": float for j in self.joint_names}

    def get_observation(self):
        obs = {f"{j}.pos": 1.0 for j in self.joint_names}
        for c in self.config.camera_topics:
            obs[c] = f"<img:{c}>"
        return obs

    def send_action(self, action):
        self.sent = dict(action)
        return {k: v + 100 for k, v in action.items()}

    def clear_heartbeat(self):
        self.cleared = True

    def reseed_heartbeat_after_reset(self):
        self.reseeded = True


def main() -> int:
    ok = True

    print("[bimanual1] 1-arm composite matches a bare AnvilOpenArmFollower exactly...", flush=True)
    plain_cfg = AnvilOpenArmFollowerConfig(
        ros_prefix="follower_r", camera_topics={"wrist_r": "/cam_wrist_r/image_raw/compressed"}
    )
    plain = AnvilOpenArmFollower(plain_cfg)
    multi = AnvilOpenArmMultiFollower(AnvilOpenArmMultiFollowerConfig(arms=[plain_cfg]))
    ok &= _check("observation_features identical", plain.observation_features == multi.observation_features)
    ok &= _check("action_features identical", plain.action_features == multi.action_features)
    ok &= _check("bus.motors keys identical", list(plain.bus.motors.keys()) == list(multi.bus.motors.keys()))
    ok &= _check("cameras keys identical", list(plain.cameras.keys()) == list(multi.cameras.keys()))

    print("\n[bimanual1] 2-arm (bimanual) composite: 14 joints + 4 cameras, left-then-right order...", flush=True)
    bi = AnvilOpenArmMultiFollower(
        AnvilOpenArmMultiFollowerConfig(
            arms=[
                AnvilOpenArmFollowerConfig(
                    ros_prefix="follower_l", camera_topics={"wrist_l": "/cam_wrist_l/image_raw/compressed"}
                ),
                AnvilOpenArmFollowerConfig(
                    ros_prefix="follower_r",
                    camera_topics={
                        "wrist_r": "/cam_wrist_r/image_raw/compressed",
                        "waist": "/cam_waist/image_raw/compressed",
                        "chest": "/cam_chest/image_raw/compressed",
                    },
                ),
            ]
        )
    )
    joint_feats = [k for k in bi.action_features if k.endswith(".pos")]
    ok &= _check("14 joint action features", len(joint_feats) == 14)
    ok &= _check(
        "left arm's joints precede right arm's",
        joint_feats[0].startswith("follower_l_joint1") and joint_feats[7].startswith("follower_r_joint1"),
    )
    cam_feats = [k for k in bi.observation_features if k not in joint_feats]
    ok &= _check("4 camera observation features", len(cam_feats) == 4)
    ok &= _check("camera names as configured", set(cam_feats) == {"wrist_l", "wrist_r", "waist", "chest"})
    ok &= _check("bus.motors has 14 joints", len(bi.bus.motors) == 14)

    print("\n[bimanual1] dict-shaped `arms` entries (as YAML would load them) also construct correctly...", flush=True)
    bi_from_dicts = AnvilOpenArmMultiFollower(
        AnvilOpenArmMultiFollowerConfig(
            arms=[
                {"ros_prefix": "follower_l", "camera_topics": {"wrist_l": "/cam_wrist_l/image_raw/compressed"}},
                {"ros_prefix": "follower_r", "camera_topics": {"wrist_r": "/cam_wrist_r/image_raw/compressed"}},
            ]
        )
    )
    ok &= _check("dict-based arms produce 14 joints", len(bi_from_dicts.bus.motors) == 14)

    print("\n[bimanual1] duplicate camera name across arms raises at construction...", flush=True)
    try:
        AnvilOpenArmMultiFollower(
            AnvilOpenArmMultiFollowerConfig(
                arms=[
                    AnvilOpenArmFollowerConfig(ros_prefix="follower_l", camera_topics={"wrist_r": "/x"}),
                    AnvilOpenArmFollowerConfig(ros_prefix="follower_r", camera_topics={"wrist_r": "/y"}),
                ]
            )
        )
        ok &= _check("duplicate camera name raises ValueError", False)
    except ValueError:
        ok &= _check("duplicate camera name raises ValueError", True)

    print("\n[bimanual1] duplicate ros_prefix across arms raises at construction...", flush=True)
    try:
        AnvilOpenArmMultiFollower(
            AnvilOpenArmMultiFollowerConfig(
                arms=[
                    AnvilOpenArmFollowerConfig(ros_prefix="follower_r"),
                    AnvilOpenArmFollowerConfig(ros_prefix="follower_r"),
                ]
            )
        )
        ok &= _check("duplicate ros_prefix raises ValueError", False)
    except ValueError:
        ok &= _check("duplicate ros_prefix raises ValueError", True)

    print("\n[bimanual1] get_observation()/send_action() fan out to and merge from every sub-arm...", flush=True)
    fan = AnvilOpenArmMultiFollower.__new__(AnvilOpenArmMultiFollower)
    left, right = _StubArm("follower_l", {"wrist_l": "x"}), _StubArm("follower_r", {"wrist_r": "x"})
    fan._arms = [left, right]

    obs = fan.get_observation()
    ok &= _check(
        "merged observation has both arms' joints", "follower_l_joint1.pos" in obs and "follower_r_joint1.pos" in obs
    )
    ok &= _check("merged observation has both cameras", "wrist_l" in obs and "wrist_r" in obs)

    combined_action = {**left.action_features, **right.action_features}
    in_action = dict.fromkeys(combined_action, 5.0)
    out = fan.send_action(in_action)
    ok &= _check(
        "send_action split the combined action correctly per arm",
        left.sent == dict.fromkeys(left.action_features, 5.0) and right.sent == dict.fromkeys(right.action_features, 5.0),
    )
    ok &= _check("send_action merged each sub-arm's return value", all(v == 105.0 for v in out.values()))

    fan.clear_heartbeat()
    fan.reseed_heartbeat_after_reset()
    ok &= _check("clear_heartbeat() fanned out to every arm", left.cleared and right.cleared)
    ok &= _check("reseed_heartbeat_after_reset() fanned out to every arm", left.reseeded and right.reseeded)

    if not ok:
        print("\n[bimanual1] FAILED", flush=True)
        return 1
    print("\n[bimanual1] PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
