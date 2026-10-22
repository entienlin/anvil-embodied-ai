#!/usr/bin/env python3
"""Generate a self-contained HTML viewer for HIL-SERL actor-episode .npz trajectories.

Reads one or more episode .npz files saved by ``save_actor_trajectory()`` in
``ros2/src/lerobot_control/lerobot_control/hilserl/patches.py`` and emits a single
offline-viewable .html file with, per episode: a frame scrubber, per-camera image
display, time-series charts (action, state, reward) with intervention periods
shaded, and summary stats. With multiple input files the episodes are laid out
side by side for comparison (e.g. early-good vs. later-degraded episode in the
same session).

No non-stdlib dependency beyond numpy (and Pillow if present, for cheap JPEG
encoding; falls back to a tiny hand-rolled PNG encoder if Pillow is missing).
Does NOT require torch, ROS2, or lerobot.

Usage:
    python3 scripts/hilserl_episode_viewer.py ep1.npz [ep2.npz ...] -o output.html

Design note (CLI vs. server): this module is deliberately kept as the single
source of truth for loading episodes and rendering the viewer HTML/JS/CSS
(load_episode, compute_fk_positions, render_skeleton_frames, build_html,
render_episode_html, encode_frame_data_uri, ...). The CLI entry point below
(main()) is one thin consumer of those functions, writing a static
self-contained .html file for one-off/offline use. A second consumer,
scripts/hilserl_episode_viewer_server.py, imports this module and calls the
exact same functions to serve the same viewer markup dynamically for an
arbitrary episode picked from a dropdown, with an in-memory LRU cache instead
of writing a file. Both stay in sync by construction -- there is no
duplicated rendering logic to drift between the two. The CLI mode is kept
working (not replaced) because it's still the simplest path for a true
one-off "give me a single static HTML to send/open later" use case; the
server is the better choice for anything involving comparing/browsing
multiple runs/episodes without regenerating files by hand.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import zlib
from typing import Any

import numpy as np

try:
    from PIL import Image

    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False


# --------------------------------------------------------------------------
# forward-kinematics skeleton pane (bimanual push-bottle jointspace task)
# --------------------------------------------------------------------------
#
# state_deg / action layout for this task is 16-dim, LEFT-then-RIGHT, each
# arm contributing its own 8-wide block of [joint1..joint7 (deg), gripper
# (raw meters, NOT degrees -- finger_joint1 is prismatic)]:
#
#   [follower_l_joint1..7, follower_l_gripper, follower_r_joint1..7, follower_r_gripper]
#
# Confirmed via:
#   - configs/hilserl/env_push_bottle_bimanual_jointspace.yaml's `arms:` list
#     (follower_l entry before follower_r) and its header comment:
#     "LEFT-then-RIGHT 16-dim state/action: [7 left joints, left gripper,
#     7 right joints, right gripper]".
#   - ros2/src/.../hilserl/robot_anvil_openarm.py:159
#     (`self.joint_names = [f"{ros_prefix}_joint{i}" for i in range(1, num_joints+1)]`)
#     and :185-188 (bus_motor_names = joint_names, then gripper joint
#     appended at the end IFF use_gripper) -- so each arm's own 8-wide
#     block is [joint1..joint7, gripper], gripper last.
#   - ros2/src/.../hilserl/robot_anvil_openarm_multi.py:112-141
#     (AnvilOpenArmMultiFollower.__init__ concatenates arm_bus_names for
#     each arm in `config.arms` list order into self.joint_names, which
#     RobotEnv/get_observation() follow directly) -- so the overall vector
#     is arm-blocks concatenated in config.arms list order, i.e. left
#     block then right block given the yaml's declared order above.
#   - robot_anvil_openarm.py:589 (`obs[f"{j}.pos"] = float(np.rad2deg(...))`
#     for the 7 revolute joints) and :597 ("Raw meters, NOT rad2deg --
#     finger_joint1 is prismatic") confirm per-dimension units: the 14 arm
#     joints are in degrees (hence "state_deg"), the 2 gripper dims are in
#     raw meters, 0.0-0.05m range.
_ARM_PREFIXES = ("follower_l", "follower_r")
_N_ARM_JOINTS = 7
_ARM_BLOCK_WIDTH = _N_ARM_JOINTS + 1  # 7 joints + 1 gripper, per arm
_ARM_COLORS = {"follower_l": "#5fd3ff", "follower_r": "#ffb15f"}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF_PATH = os.path.join(
    _REPO_ROOT, "configs", "hilserl", "urdf", "openarm_teleop_kinematics.urdf"
)

# --------------------------------------------------------------------------
# real control frequency, for the playback-speed control
# --------------------------------------------------------------------------
#
# Confirmed directly from this task's own configs (not assumed):
#   configs/hilserl/env_push_bottle_bimanual_jointspace.yaml:41   fps: 10
#   configs/hilserl/env_push_bottle_bimanual_jointspace.yaml:79,92  control_frequency: 10.0
#   configs/hilserl/train_rl_push_bottle_bimanual_jointspace.yaml:94  fps: 10
#   configs/hilserl/train_rl_push_bottle_bimanual_jointspace.yaml:132,158  control_frequency: 10.0
# i.e. one saved (state_deg/action/...) step IS one real 100ms control tick.
# "1x / real-time" playback therefore means 100ms between frames, not the
# previously-hardcoded 90ms (~11.1 Hz, ~1.11x real speed) the player used.
CONTROL_FREQUENCY_HZ = 10.0
REAL_FRAME_INTERVAL_MS = 1000.0 / CONTROL_FREQUENCY_HZ  # 100.0


def _try_import_placo():
    """placo (the same FK/IK library this project already uses via
    lerobot.model.kinematics.RobotKinematics, see
    ros2/src/.../hilserl/live_ee_position.py) is a compiled C++ extension
    with Python bindings. It's directly importable in the host .venv here
    (confirmed: no Docker/ROS2/rclpy needed) -- so we use it directly for
    FK rather than shelling out to the hilserl-test Docker image or
    hand-rolling a custom URDF-chain FK implementation."""
    try:
        import placo  # type: ignore[import-not-found]

        return placo
    except Exception:
        return None


def _fk_chain_names(prefix: str) -> list[str]:
    """Frame names for one arm's chain, base to tip: link0..link7, then the
    fixed hand_tcp frame (confirmed present in the URDF for both follower
    arms, independent of gripper/finger joint values -- see
    configs/hilserl/urdf/openarm_teleop_kinematics.urdf's
    follower_{l,r}_hand_tcp_joint, a fixed joint off of follower_{l,r}_hand,
    itself fixed off link7; finger joints only affect the two finger links,
    not hand_tcp)."""
    return [f"{prefix}_link{i}" for i in range(_N_ARM_JOINTS + 1)] + [f"{prefix}_hand_tcp"]


def compute_fk_positions(state: np.ndarray, urdf_path: str = DEFAULT_URDF_PATH) -> dict[str, np.ndarray] | None:
    """FK over every frame of one episode's state_deg (N,16) array.

    Returns {"follower_l": (N, 9, 3), "follower_r": (N, 9, 3)} world-frame
    xyz (meters) for each arm's link0..link7 + hand_tcp chain, or None if
    placo isn't importable or the state array doesn't look like the
    16-dim bimanual layout this function assumes.
    """
    placo = _try_import_placo()
    if placo is None:
        return None
    if state.ndim != 2 or state.shape[1] < 2 * _ARM_BLOCK_WIDTH:
        return None
    if not os.path.isfile(urdf_path):
        return None

    robot = placo.RobotWrapper(urdf_path)
    chains = {p: _fk_chain_names(p) for p in _ARM_PREFIXES}
    n = state.shape[0]
    out = {p: np.zeros((n, len(chains[p]), 3), dtype=np.float32) for p in _ARM_PREFIXES}

    for t in range(n):
        for i in range(_N_ARM_JOINTS):
            left_deg = float(state[t, i])
            right_deg = float(state[t, _ARM_BLOCK_WIDTH + i])
            robot.set_joint(f"follower_l_joint{i + 1}", float(np.deg2rad(left_deg)))
            robot.set_joint(f"follower_r_joint{i + 1}", float(np.deg2rad(right_deg)))
        robot.update_kinematics()
        for p in _ARM_PREFIXES:
            for i, frame_name in enumerate(chains[p]):
                out[p][t, i] = robot.get_T_world_frame(frame_name)[:3, 3]

    return out


def render_skeleton_frames(fk: dict[str, np.ndarray], dpi: int = 92, figsize: tuple[float, float] = (2.6, 2.6)) -> list[str]:
    """Render each frame's two-arm stick-figure skeleton (FK output) as a
    small PNG data URI, one fixed oblique 3D viewing angle, axis limits
    held constant across the whole episode (computed once from the global
    min/max over every frame) so the pane doesn't rescale/jitter frame to
    frame -- only the arms themselves should appear to move."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = next(iter(fk.values())).shape[0]
    all_pts = np.concatenate([arr.reshape(-1, 3) for arr in fk.values()], axis=0)
    center = all_pts.mean(axis=0)
    radius = float(np.max(np.abs(all_pts - center))) * 1.15 + 1e-3

    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#11131a")
    ax = fig.add_subplot(111, projection="3d")

    data_uris: list[str] = []
    for t in range(n):
        ax.clear()
        ax.set_facecolor("#11131a")
        for p in _ARM_PREFIXES:
            pts = fk[p][t]
            ax.plot(
                pts[:, 0], pts[:, 1], pts[:, 2],
                "-o", markersize=3, linewidth=1.8, color=_ARM_COLORS[p], label=p,
            )
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.view_init(elev=18, azim=-60)
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="#11131a")
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        data_uris.append(f"data:image/png;base64,{b64}")

    plt.close(fig)
    return data_uris


# --------------------------------------------------------------------------
# npz loading
# --------------------------------------------------------------------------


def load_episode(path: str) -> dict[str, Any]:
    """Load one episode .npz and return arrays + derived camera/stat info.

    Schema (see save_actor_trajectory): state_deg (N,S), action (N,A),
    reward (N,), done (N,), is_intervention (N,), and one or more camera
    arrays. Camera arrays are either a single legacy key "image" (N,3,128,128)
    from the single-arm era, or "image_<camera_name>" keys (bimanual, "all
    cameras" fix) -- both forms are handled generically here.
    """
    with np.load(path) as npz:
        keys = list(npz.files)
        state = np.asarray(npz["state_deg"], dtype=np.float32) if "state_deg" in keys else np.zeros((0, 0))
        action = np.asarray(npz["action"], dtype=np.float32) if "action" in keys else np.zeros((0, 0))
        reward = np.asarray(npz["reward"], dtype=np.float32) if "reward" in keys else np.zeros((0,))
        done = np.asarray(npz["done"], dtype=np.float32) if "done" in keys else np.zeros((0,))
        is_intervention = (
            np.asarray(npz["is_intervention"], dtype=np.float32)
            if "is_intervention" in keys
            else np.zeros((len(reward),))
        )

        cameras: dict[str, np.ndarray] = {}
        for k in keys:
            if k == "image":
                cameras["cam"] = np.asarray(npz[k])
            elif k.startswith("image_"):
                cameras[k[len("image_") :]] = np.asarray(npz[k])

    n = int(reward.shape[0]) if reward.ndim >= 1 else 0

    stats = {
        "n_steps": n,
        "action_dim": int(action.shape[1]) if action.ndim == 2 and action.size else 0,
        "state_dim": int(state.shape[1]) if state.ndim == 2 and state.size else 0,
        "n_cameras": len(cameras),
        "camera_names": sorted(cameras.keys()),
        "intervention_rate": float(np.mean(is_intervention)) if n else 0.0,
        "n_intervention_steps": int(np.sum(is_intervention >= 0.5)) if n else 0,
        "final_reward": float(reward[-1]) if n else None,
        "final_done": float(done[-1]) if n else None,
        "success": bool(n and done[-1] >= 0.5 and reward[-1] >= 0.5),
        "any_reward_fired": bool(n and np.any(reward >= 0.5)),
        "reward_sum": float(np.sum(reward)) if n else 0.0,
        "action_mean_abs": float(np.mean(np.abs(action))) if action.size else 0.0,
        "action_std": float(np.std(action)) if action.size else 0.0,
        "state_mean_abs": float(np.mean(np.abs(state))) if state.size else 0.0,
        "state_std": float(np.std(state)) if state.size else 0.0,
    }

    return {
        "path": path,
        "basename": os.path.basename(path),
        "state": state,
        "action": action,
        "reward": reward,
        "done": done,
        "is_intervention": is_intervention,
        "cameras": cameras,
        "stats": stats,
        "n_steps": n,
    }


# --------------------------------------------------------------------------
# image encoding: CHW float16 (range ~[0,1]) -> base64 data URI
# --------------------------------------------------------------------------


def _frame_to_uint8_hwc(frame_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) float -> (H,W,3) uint8, clipped to [0,1] then scaled to 255."""
    arr = np.asarray(frame_chw, dtype=np.float32)
    arr = np.transpose(arr, (1, 2, 0))  # CHW -> HWC
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0 + 0.5).astype(np.uint8)


def _encode_png_fallback(hwc_uint8: np.ndarray) -> bytes:
    """Minimal, dependency-free PNG encoder (used only if Pillow is absent)."""
    h, w, _ = hwc_uint8.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type: none
        raw.extend(hwc_uint8[y].tobytes())
    compressed = zlib.compress(bytes(raw), level=6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        return (
            len(data).to_bytes(4, "big")
            + c
            + zlib.crc32(c).to_bytes(4, "big")
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = (
        w.to_bytes(4, "big")
        + h.to_bytes(4, "big")
        + bytes([8, 2, 0, 0, 0])  # bit depth 8, color type 2 (RGB)
    )
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", compressed) + chunk(b"IEND", b"")


def encode_frame_data_uri(frame_chw: np.ndarray, jpeg_quality: int = 80) -> str:
    hwc = _frame_to_uint8_hwc(frame_chw)
    if _HAVE_PIL:
        buf = io.BytesIO()
        Image.fromarray(hwc, mode="RGB").save(buf, format="JPEG", quality=jpeg_quality)
        mime = "image/jpeg"
        data = buf.getvalue()
    else:
        data = _encode_png_fallback(hwc)
        mime = "image/png"
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


# --------------------------------------------------------------------------
# HTML assembly
# --------------------------------------------------------------------------

CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 16px; background: #14161a; color: #e7e9ee;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 13px;
}
h1 { font-size: 18px; margin: 0 0 12px 0; }
.episodes {
  display: flex; flex-direction: row; gap: 16px; align-items: flex-start;
  overflow-x: auto; padding-bottom: 24px;
}
.episode {
  flex: 0 0 auto; width: 560px; background: #1c1f26; border: 1px solid #2d313b;
  border-radius: 8px; padding: 12px;
}
.episode h2 {
  font-size: 13px; font-weight: 600; margin: 0 0 8px 0; word-break: break-all;
  color: #9fd3ff;
}
.stats-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 4px 12px;
  background: #11131700; border: 1px solid #2d313b; border-radius: 6px;
  padding: 8px 10px; margin-bottom: 10px; font-size: 12px;
}
.stats-grid .k { color: #9aa3b2; }
.stats-grid .v { color: #e7e9ee; font-weight: 600; text-align: right; }
.success-yes { color: #5fd38a !important; }
.success-no { color: #e2726b !important; }
.controls {
  display: flex; align-items: center; gap: 6px; margin-bottom: 8px; flex-wrap: wrap;
}
.controls button {
  background: #2a2e38; color: #e7e9ee; border: 1px solid #3a3f4b; border-radius: 4px;
  padding: 4px 10px; cursor: pointer; font-size: 12px;
}
.controls button:hover { background: #343945; }
.controls select {
  background: #2a2e38; color: #e7e9ee; border: 1px solid #3a3f4b; border-radius: 4px;
  padding: 3px 6px; cursor: pointer; font-size: 12px;
}
.controls input[type=range] { flex: 1 1 auto; min-width: 120px; }
.frame-label { font-size: 12px; color: #9aa3b2; white-space: nowrap; min-width: 86px; text-align: right; }
.cams { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
.cam-box { text-align: center; }
.cam-box img {
  width: 128px; height: 128px; image-rendering: pixelated; display: block;
  background: #000; border: 1px solid #333; border-radius: 4px;
}
.cam-box .cam-label { font-size: 11px; color: #9aa3b2; margin-top: 2px; }
.skel-box { text-align: center; margin-bottom: 10px; }
.skel-box img {
  width: 234px; height: 234px; display: block; background: #11131a;
  border: 1px solid #333; border-radius: 4px; margin: 0 auto;
}
.skel-box .skel-label { font-size: 11px; color: #9aa3b2; margin-top: 2px; }
.skel-legend { display: flex; justify-content: center; gap: 14px; margin-top: 4px; font-size: 10px; color: #c3c9d4; }
.skel-legend .chip { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 3px; vertical-align: middle; }
.skel-unavailable { font-size: 11px; color: #e2a65f; padding: 10px; text-align: center; }
.chart-block { margin-bottom: 10px; }
.chart-title { font-size: 11px; color: #9aa3b2; margin-bottom: 2px; }
.chart-wrap { position: relative; width: 100%; }
.chart-wrap canvas { width: 100%; display: block; background: #11131a; border-radius: 4px; }
.cursor-line {
  position: absolute; top: 0; bottom: 0; width: 1px; background: #ffd966;
  pointer-events: none; left: 0;
}
.legend { display: flex; flex-wrap: wrap; gap: 6px 10px; margin-top: 3px; font-size: 10px; color: #c3c9d4; }
.legend .chip { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 3px; vertical-align: middle; }
.readout { font-size: 11px; color: #c3c9d4; margin-top: 2px; }
.intervention-badge {
  display: inline-block; padding: 1px 6px; border-radius: 3px; font-size: 11px; margin-left: 6px;
}
.intervention-on { background: #5a2d2d; color: #ffb3a7; }
.intervention-off { background: #273a2a; color: #a9e6b4; }
"""

JS = r"""
function hslColor(i, n) {
  const hue = Math.round((i * 360) / Math.max(n, 1));
  return `hsl(${hue}, 70%, 60%)`;
}

function normalizeSeries(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const span = hi - lo;
  return { norm: values.map(v => (v - lo) / span), lo, hi };
}

function drawChart(canvas, seriesMatrix, labels, interventionMask) {
  // seriesMatrix: array of arrays (one per line), all same length N
  const n = (seriesMatrix[0] || []).length;
  const w = canvas.width, h = canvas.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  // intervention shading (applies across full chart height)
  if (interventionMask && n > 0) {
    ctx.fillStyle = "rgba(220, 90, 70, 0.22)";
    const stepW = w / n;
    let runStart = -1;
    for (let i = 0; i < n; i++) {
      const on = interventionMask[i] >= 0.5;
      if (on && runStart === -1) runStart = i;
      if ((!on || i === n - 1) && runStart !== -1) {
        const endIdx = on ? i + 1 : i;
        ctx.fillRect(runStart * stepW, 0, (endIdx - runStart) * stepW, h);
        runStart = -1;
      }
    }
  }

  const pad = 4;
  seriesMatrix.forEach((series, idx) => {
    if (series.length < 2) return;
    const { norm } = normalizeSeries(series);
    ctx.strokeStyle = hslColor(idx, seriesMatrix.length);
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    for (let i = 0; i < norm.length; i++) {
      const x = (i / (norm.length - 1)) * (w - 2 * pad) + pad;
      const y = h - pad - norm[i] * (h - 2 * pad);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  });
}

function buildLegend(container, labels, nSeries) {
  container.innerHTML = "";
  for (let i = 0; i < nSeries; i++) {
    const span = document.createElement("span");
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.style.background = hslColor(i, nSeries);
    span.appendChild(chip);
    span.appendChild(document.createTextNode(labels[i] || ("dim " + i)));
    container.appendChild(span);
  }
}

function colsOf(matrix2d) {
  // matrix2d: N x D -> D arrays of length N
  if (!matrix2d || matrix2d.length === 0) return [];
  const d = matrix2d[0].length;
  const out = [];
  for (let j = 0; j < d; j++) out.push(matrix2d.map(row => row[j]));
  return out;
}

function setupEpisode(epId) {
  const dataEl = document.getElementById("data-" + epId);
  const data = JSON.parse(dataEl.textContent);
  const n = data.n_steps;

  const slider = document.getElementById("slider-" + epId);
  const frameLabel = document.getElementById("framelabel-" + epId);
  const playBtn = document.getElementById("play-" + epId);
  const prevBtn = document.getElementById("prev-" + epId);
  const nextBtn = document.getElementById("next-" + epId);
  const speedSelect = document.getElementById("speed-" + epId);
  const readout = document.getElementById("readout-" + epId);
  const camImgs = {};
  data.camera_names.forEach(cam => {
    camImgs[cam] = document.getElementById("img-" + epId + "-" + cam);
  });
  const skelImg = document.getElementById("skel-" + epId);
  const cursors = [
    document.getElementById("cursor-action-" + epId),
    document.getElementById("cursor-state-" + epId),
    document.getElementById("cursor-reward-" + epId),
  ].filter(Boolean);

  const actionCols = colsOf(data.action);
  const stateCols = colsOf(data.state);
  const actionLabels = actionCols.map((_, i) => "a" + i);
  const stateLabels = stateCols.map((_, i) => "s" + i);

  const actionCanvas = document.getElementById("chart-action-" + epId);
  const stateCanvas = document.getElementById("chart-state-" + epId);
  const rewardCanvas = document.getElementById("chart-reward-" + epId);

  if (actionCanvas && actionCols.length) {
    drawChart(actionCanvas, actionCols, actionLabels, data.is_intervention);
    buildLegend(document.getElementById("legend-action-" + epId), actionLabels, actionCols.length);
  }
  if (stateCanvas && stateCols.length) {
    drawChart(stateCanvas, stateCols, stateLabels, data.is_intervention);
    buildLegend(document.getElementById("legend-state-" + epId), stateLabels, stateCols.length);
  }
  if (rewardCanvas) {
    drawChart(rewardCanvas, [data.reward, data.done], ["reward", "done"], data.is_intervention);
    buildLegend(document.getElementById("legend-reward-" + epId), ["reward", "done"], 2);
  }

  let frameIdx = 0;
  let playing = false;
  let timer = null;

  function render() {
    data.camera_names.forEach(cam => {
      const img = camImgs[cam];
      if (img) img.src = data.images[cam][frameIdx];
    });
    if (skelImg && data.skeleton_available) {
      skelImg.src = data.skeleton_images[frameIdx];
    }
    frameLabel.textContent = "Frame " + frameIdx + " / " + (n - 1);
    slider.value = frameIdx;
    const frac = n > 1 ? frameIdx / (n - 1) : 0;
    cursors.forEach(c => { c.style.left = (frac * 100) + "%"; });
    const interv = data.is_intervention[frameIdx] >= 0.5;
    const r = data.reward[frameIdx];
    const d = data.done[frameIdx];
    readout.innerHTML =
      "reward=" + r.toFixed(2) + "  done=" + d.toFixed(0) +
      "<span class='intervention-badge " + (interv ? "intervention-on" : "intervention-off") + "'>" +
      (interv ? "HUMAN" : "policy") + "</span>";
  }

  function stopPlay() {
    playing = false;
    if (timer) { clearInterval(timer); timer = null; }
    playBtn.textContent = "Play";
  }

  // Playback speed: REAL_FRAME_INTERVAL_MS is the true real-world control
  // tick (100ms @ 10Hz for this task, see Python REAL_FRAME_INTERVAL_MS /
  // CONTROL_FREQUENCY_HZ). "speed" is a multiplier on top of that real
  // interval -- 1x reproduces real-time playback exactly, 2x/4x speed it
  // up, 0.25x/0.5x slow it down. The previous player had no speed control
  // at all and a hardcoded 90ms tick (~1.11x real speed), kept below as an
  // explicit option so old recordings/behavior can still be reproduced.
  const REAL_FRAME_INTERVAL_MS = window.__realFrameIntervalMs || 100;

  function currentIntervalMs() {
    const speed = speedSelect ? parseFloat(speedSelect.value) : 1;
    return REAL_FRAME_INTERVAL_MS / (speed > 0 ? speed : 1);
  }

  function tick() {
    frameIdx += 1;
    if (frameIdx >= n) { stopPlay(); frameIdx = n - 1; render(); return; }
    render();
  }

  function restartTimerIfPlaying() {
    if (!playing) return;
    if (timer) clearInterval(timer);
    timer = setInterval(tick, currentIntervalMs());
  }

  slider.addEventListener("input", () => { stopPlay(); frameIdx = parseInt(slider.value, 10); render(); });
  prevBtn.addEventListener("click", () => { stopPlay(); frameIdx = Math.max(0, frameIdx - 1); render(); });
  nextBtn.addEventListener("click", () => { stopPlay(); frameIdx = Math.min(n - 1, frameIdx + 1); render(); });
  playBtn.addEventListener("click", () => {
    if (playing) { stopPlay(); return; }
    playing = true;
    playBtn.textContent = "Pause";
    timer = setInterval(tick, currentIntervalMs());
  });
  if (speedSelect) {
    speedSelect.addEventListener("change", restartTimerIfPlaying);
  }

  render();
}

window.addEventListener("DOMContentLoaded", () => {
  (window.__episodeIds || []).forEach(setupEpisode);
});
"""


def _stat_rows(stats: dict[str, Any]) -> list[tuple[str, str, str]]:
    success_cls = "success-yes" if stats["success"] else "success-no"
    rows = [
        ("Steps (N)", str(stats["n_steps"]), ""),
        ("Intervention rate", f"{stats['intervention_rate']:.1%}", ""),
        ("Intervention steps", f"{stats['n_intervention_steps']} / {stats['n_steps']}", ""),
        ("Final reward", f"{stats['final_reward']:.3f}" if stats["final_reward"] is not None else "n/a", ""),
        ("Final done", f"{stats['final_done']:.0f}" if stats["final_done"] is not None else "n/a", ""),
        ("Success (done & reward)", "YES" if stats["success"] else "no", success_cls),
        ("Reward fired at all", "yes" if stats["any_reward_fired"] else "no", ""),
        ("Reward sum", f"{stats['reward_sum']:.3f}", ""),
        ("Action |mean|", f"{stats['action_mean_abs']:.3f}", ""),
        ("Action std", f"{stats['action_std']:.3f}", ""),
        ("State |mean|", f"{stats['state_mean_abs']:.3f}", ""),
        ("State std", f"{stats['state_std']:.3f}", ""),
        ("Action dim / State dim", f"{stats['action_dim']} / {stats['state_dim']}", ""),
        ("Cameras", ", ".join(stats["camera_names"]) or "none", ""),
    ]
    return rows


def render_episode_html(
    ep: dict[str, Any],
    ep_id: str,
    jpeg_quality: int,
    skeleton_images: list[str] | None = None,
    skeleton_error: str | None = None,
) -> tuple[str, str]:
    """Returns (html_fragment, json_data_script)."""
    n = ep["n_steps"]
    stats = ep["stats"]

    images: dict[str, list[str]] = {}
    for cam, arr in ep["cameras"].items():
        images[cam] = [encode_frame_data_uri(arr[i], jpeg_quality=jpeg_quality) for i in range(n)]

    data_payload = {
        "n_steps": n,
        "camera_names": stats["camera_names"],
        "action": ep["action"].tolist(),
        "state": ep["state"].tolist(),
        "reward": ep["reward"].tolist(),
        "done": ep["done"].tolist(),
        "is_intervention": ep["is_intervention"].tolist(),
        "images": images,
        "skeleton_available": skeleton_images is not None,
        "skeleton_images": skeleton_images or [],
    }
    data_json = json.dumps(data_payload)

    stat_rows_html = "\n".join(
        f'<div class="k">{k}</div><div class="v {cls}">{v}</div>' for k, v, cls in _stat_rows(stats)
    )

    cam_html = "\n".join(
        f'<div class="cam-box"><img id="img-{ep_id}-{cam}" alt="{cam}">'
        f'<div class="cam-label">{cam}</div></div>'
        for cam in stats["camera_names"]
    )
    if not cam_html:
        cam_html = '<div class="cam-box"><div class="cam-label">no camera arrays found</div></div>'

    if skeleton_images is not None:
        skel_html = f"""
        <div class="skel-box">
          <img id="skel-{ep_id}" alt="FK skeleton">
          <div class="skel-label">FK skeleton (both arms, from state_deg)</div>
          <div class="skel-legend">
            <span><span class="chip" style="background:{_ARM_COLORS["follower_l"]}"></span>follower_l</span>
            <span><span class="chip" style="background:{_ARM_COLORS["follower_r"]}"></span>follower_r</span>
          </div>
        </div>"""
    else:
        reason = skeleton_error or "skeleton disabled"
        skel_html = f'<div class="skel-box"><div class="skel-unavailable">FK skeleton unavailable: {reason}</div></div>'

    def chart_block(title: str, chart_key: str) -> str:
        return f"""
        <div class="chart-block">
          <div class="chart-title">{title}</div>
          <div class="chart-wrap">
            <canvas id="chart-{chart_key}-{ep_id}" width="1040" height="130"></canvas>
            <div class="cursor-line" id="cursor-{chart_key}-{ep_id}"></div>
          </div>
          <div class="legend" id="legend-{chart_key}-{ep_id}"></div>
        </div>"""

    fragment = f"""
    <div class="episode">
      <h2>{ep["basename"]}</h2>
      <div class="stats-grid">{stat_rows_html}</div>
      <div class="controls">
        <button id="prev-{ep_id}">&laquo; prev</button>
        <button id="play-{ep_id}">Play</button>
        <button id="next-{ep_id}">next &raquo;</button>
        <input type="range" id="slider-{ep_id}" min="0" max="{max(n - 1, 0)}" value="0" step="1">
        <span class="frame-label" id="framelabel-{ep_id}">Frame 0 / {max(n - 1, 0)}</span>
        <select id="speed-{ep_id}" title="Playback speed (1x = real time, {CONTROL_FREQUENCY_HZ:g}Hz = {REAL_FRAME_INTERVAL_MS:g}ms/frame)">
          <option value="0.25">0.25x</option>
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x (real-time, {CONTROL_FREQUENCY_HZ:g} Hz)</option>
          <option value="1.1111">old default (~11.1 Hz)</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>
      </div>
      <div class="readout" id="readout-{ep_id}"></div>
      <div class="cams">{cam_html}</div>
      {skel_html}
      {chart_block("Action (all dims, each line min/max-normalized)", "action")}
      {chart_block("State (deg, all dims, each line min/max-normalized)", "state")}
      {chart_block("Reward / Done", "reward")}
      <div class="readout">Red shaded bands = human intervention (is_intervention=1).</div>
    </div>"""

    data_script = f'<script type="application/json" id="data-{ep_id}">{data_json}</script>'
    return fragment, data_script


def build_html(
    episodes: list[dict[str, Any]],
    jpeg_quality: int,
    skeleton_results: list[tuple[list[str] | None, str | None]] | None = None,
) -> str:
    ep_ids = [f"ep{i}" for i in range(len(episodes))]
    if skeleton_results is None:
        skeleton_results = [(None, "skeleton disabled (--no-skeleton)")] * len(episodes)
    fragments = []
    data_scripts = []
    for ep, ep_id, (skel_images, skel_error) in zip(episodes, ep_ids, skeleton_results):
        frag, data_script = render_episode_html(ep, ep_id, jpeg_quality, skel_images, skel_error)
        fragments.append(frag)
        data_scripts.append(data_script)

    title = "HIL-SERL Episode Viewer"
    if len(episodes) == 1:
        title += f" — {episodes[0]['basename']}"
    else:
        title += f" — comparing {len(episodes)} episodes"

    ep_ids_json = json.dumps(ep_ids)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<h1>{title}</h1>
<div class="episodes">
{"".join(fragments)}
</div>
{"".join(data_scripts)}
<script>
window.__episodeIds = {ep_ids_json};
window.__realFrameIntervalMs = {REAL_FRAME_INTERVAL_MS!r};  // 1x/real-time, from CONTROL_FREQUENCY_HZ={CONTROL_FREQUENCY_HZ!r}
{JS}
</script>
</body>
</html>"""


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a self-contained HTML viewer for HIL-SERL actor episode .npz trajectories."
    )
    parser.add_argument("npz_files", nargs="+", help="One or more episode .npz files (single = viewer, multiple = side-by-side comparison).")
    parser.add_argument("-o", "--output", default="episode_viewer.html", help="Output .html path (default: episode_viewer.html).")
    parser.add_argument(
        "--jpeg-quality", type=int, default=80, help="JPEG quality for embedded frames (1-95, default 80). Ignored if Pillow is unavailable (falls back to PNG)."
    )
    parser.add_argument(
        "--skeleton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include a forward-kinematics stick-figure skeleton pane per episode, computed from "
            "state_deg via placo FK against the project's openarm_teleop_kinematics.urdf (default: "
            "on; FK itself is fast but per-frame matplotlib rendering adds ~5-15ms/frame -- use "
            "--no-skeleton to skip it, e.g. for quick iteration on other parts of the viewer)."
        ),
    )
    parser.add_argument(
        "--urdf-path",
        default=DEFAULT_URDF_PATH,
        help=f"URDF used for the FK skeleton pane (default: {DEFAULT_URDF_PATH}).",
    )
    args = parser.parse_args(argv)

    episodes = []
    for path in args.npz_files:
        if not os.path.isfile(path):
            print(f"error: file not found: {path}", file=sys.stderr)
            return 1
        print(f"Loading {path} ...", file=sys.stderr)
        ep = load_episode(path)
        print(
            f"  N={ep['n_steps']} action_dim={ep['stats']['action_dim']} state_dim={ep['stats']['state_dim']} "
            f"cameras={ep['stats']['camera_names']} intervention_rate={ep['stats']['intervention_rate']:.1%}",
            file=sys.stderr,
        )
        episodes.append(ep)

    skeleton_results: list[tuple[list[str] | None, str | None]] = []
    if args.skeleton:
        for ep in episodes:
            print(f"Computing FK skeleton for {ep['basename']} ...", file=sys.stderr)
            fk = compute_fk_positions(ep["state"], urdf_path=args.urdf_path)
            if fk is None:
                msg = (
                    "placo not importable"
                    if _try_import_placo() is None
                    else f"state_dim={ep['stats']['state_dim']} (need 16-dim bimanual layout) or URDF not found"
                )
                print(f"  skeleton skipped: {msg}", file=sys.stderr)
                skeleton_results.append((None, msg))
                continue
            images = render_skeleton_frames(fk)
            print(f"  rendered {len(images)} skeleton frames", file=sys.stderr)
            skeleton_results.append((images, None))
    else:
        skeleton_results = [(None, "skeleton disabled (--no-skeleton)")] * len(episodes)

    html = build_html(episodes, jpeg_quality=args.jpeg_quality, skeleton_results=skeleton_results)

    with open(args.output, "w") as f:
        f.write(html)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote {args.output} ({size_mb:.2f} MB) for {len(episodes)} episode(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
