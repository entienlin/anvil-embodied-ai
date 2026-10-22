#!/usr/bin/env python3
"""Small local HTTP server for browsing/comparing HIL-SERL actor-episode .npz
trajectories, without regenerating a static HTML file by hand for every
run/episode combination.

Rationale for a server instead of a bigger static file (see
scripts/hilserl_episode_viewer.py's module docstring for the full design
note): the CLI tool bakes every requested episode's camera frames + FK
skeleton frames into one offline .html, which is great for a one-off but
doesn't scale to "pick any episode from any run, including runs that don't
exist yet, via a dropdown" -- that would mean either regenerating a file per
click or pre-baking every run's every episode up front (expensive, and bad
for disk usage on top of the ~83GB already used by other layer caches on
this box). This server instead renders on demand and keeps only a small
in-memory LRU of the most recently viewed episodes (FK/skeleton rendering is
the expensive part, a couple of seconds per episode) -- nothing is ever
written to disk by this script. Restarting the server clears the cache; no
cache directory accumulates on disk.

It reuses -- does not duplicate -- hilserl_episode_viewer.py's loading and
HTML-rendering functions (load_episode, compute_fk_positions,
render_skeleton_frames, build_html, REAL_FRAME_INTERVAL_MS, ...): this
script only adds directory scanning, the index/compare page, routing, and
the in-memory cache.

Usage:
    /home/b50317/openarm/anvil-embodied-ai/.venv/bin/python3 \\
        scripts/hilserl_episode_viewer_server.py \\
        --root outputs/hilserl_bimanual_runs --port 8787

Then forward port 8787 (e.g. VS Code's Ports panel) and open it in a
browser. Plain HTTP, no TLS/auth -- intended for local single-user use only
(same trust model as `python3 -m http.server`, which this project already
used this session for the static-HTML output).
"""

from __future__ import annotations

import argparse
import collections
import glob
import html
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import hilserl_episode_viewer as viewer

# --------------------------------------------------------------------------
# run/episode discovery
# --------------------------------------------------------------------------


def scan_runs(root: str) -> dict[str, list[str]]:
    """Scan `root` for `*/actor_trajectories/*.npz`, grouped by run (the
    directory name directly under `root` that contains actor_trajectories/).
    Returns {run_name: [episode_basename, ...]} sorted so the most recent
    run name sorts first and episode basenames sort ascending (both are
    timestamp-prefixed, so this is also chronological). Re-scanned on every
    index page load (cheap: a glob), so new runs/episodes created after the
    server started show up without a restart -- no index needs to be
    rebuilt or cached on disk for that to work.
    """
    pattern = os.path.join(root, "*", "actor_trajectories", "*.npz")
    runs: dict[str, list[str]] = collections.defaultdict(list)
    for path in glob.glob(pattern):
        run_name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        runs[run_name].append(os.path.basename(path))
    return {
        run: sorted(files)
        for run, files in sorted(runs.items(), key=lambda kv: kv[0], reverse=True)
    }


def episode_path(root: str, run: str, filename: str) -> str:
    return os.path.join(root, run, "actor_trajectories", filename)


# --------------------------------------------------------------------------
# in-memory LRU cache (process lifetime only -- nothing touches disk)
# --------------------------------------------------------------------------


class EpisodeCache:
    """Caches loaded episode + rendered FK skeleton frames, keyed by
    (run, filename). Pure in-memory, bounded to `maxsize` entries, evicting
    least-recently-used. No disk footprint whatsoever -- this intentionally
    does NOT persist across server restarts, per the project's constraint
    against accumulating new on-disk cache trash."""

    def __init__(self, root: str, maxsize: int, skeleton: bool, urdf_path: str, jpeg_quality: int):
        self._root = root
        self._maxsize = maxsize
        self._skeleton = skeleton
        self._urdf_path = urdf_path
        self._jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._data: "collections.OrderedDict[tuple[str, str], tuple[dict, list[str] | None, str | None]]" = (
            collections.OrderedDict()
        )

    def get(self, run: str, filename: str) -> tuple[dict, list[str] | None, str | None]:
        key = (run, filename)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return self._data[key]

        path = episode_path(self._root, run, filename)
        print(f"[cache miss] loading {run}/{filename} ...", file=sys.stderr)
        t0 = time.time()
        ep = viewer.load_episode(path)

        skel_images: list[str] | None = None
        skel_error: str | None = None
        if self._skeleton:
            fk = viewer.compute_fk_positions(ep["state"], urdf_path=self._urdf_path)
            if fk is None:
                skel_error = (
                    "placo not importable"
                    if viewer._try_import_placo() is None
                    else f"state_dim={ep['stats']['state_dim']} (need 16-dim bimanual layout) or URDF not found"
                )
            else:
                skel_images = viewer.render_skeleton_frames(fk)
        else:
            skel_error = "skeleton disabled (--no-skeleton)"

        print(f"  ... loaded+rendered in {time.time() - t0:.2f}s (N={ep['n_steps']})", file=sys.stderr)

        with self._lock:
            self._data[key] = (ep, skel_images, skel_error)
            self._data.move_to_end(key)
            while len(self._data) > self._maxsize:
                evicted_key, _ = self._data.popitem(last=False)
                print(f"[cache evict] {evicted_key[0]}/{evicted_key[1]}", file=sys.stderr)
        return self._data[key]

    @property
    def jpeg_quality(self) -> int:
        return self._jpeg_quality


# --------------------------------------------------------------------------
# page rendering
# --------------------------------------------------------------------------

INDEX_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 24px; background: #14161a; color: #e7e9ee;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 14px;
}
h1 { font-size: 19px; margin: 0 0 4px 0; }
.sub { color: #9aa3b2; font-size: 12px; margin-bottom: 20px; }
.card {
  background: #1c1f26; border: 1px solid #2d313b; border-radius: 8px;
  padding: 18px 20px; max-width: 560px;
}
label { display: block; font-size: 12px; color: #9aa3b2; margin: 14px 0 4px 0; }
select {
  width: 100%; background: #2a2e38; color: #e7e9ee; border: 1px solid #3a3f4b;
  border-radius: 4px; padding: 7px 8px; font-size: 13px;
}
button.submit {
  margin-top: 18px; background: #3c6ea5; color: #fff; border: none; border-radius: 5px;
  padding: 9px 18px; font-size: 13px; cursor: pointer;
}
button.submit:hover { background: #4a7fbb; }
.hint { font-size: 11px; color: #747d8c; margin-top: 6px; }
.counts { font-size: 11px; color: #747d8c; margin-top: 14px; }
"""

INDEX_JS = r"""
function onRunChange() {
  const run = document.getElementById("run").value;
  const episodes = (window.__runIndex[run] || []);
  for (const id of ["a", "b"]) {
    const sel = document.getElementById(id);
    const keepNone = id === "b";
    sel.innerHTML = "";
    if (keepNone) {
      const opt = document.createElement("option");
      opt.value = ""; opt.textContent = "(none -- single episode)";
      sel.appendChild(opt);
    }
    episodes.forEach(ep => {
      const opt = document.createElement("option");
      opt.value = ep; opt.textContent = ep;
      sel.appendChild(opt);
    });
  }
}
window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("run").addEventListener("change", onRunChange);
  onRunChange();
});
"""


def render_index_html(runs: dict[str, list[str]], root: str) -> str:
    run_options = "\n".join(f'<option value="{html.escape(r)}">{html.escape(r)} ({len(fs)} episodes)</option>' for r, fs in runs.items())
    total_eps = sum(len(fs) for fs in runs.values())
    run_index_json = json.dumps(runs)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HIL-SERL Episode Viewer</title>
<style>{INDEX_CSS}</style>
</head>
<body>
<h1>HIL-SERL Episode Viewer</h1>
<div class="sub">Root: {html.escape(root)} &middot; {len(runs)} runs, {total_eps} episodes (rescanned on every page load)</div>
<div class="card">
  <form method="get" action="/view">
    <label for="run">Run</label>
    <select id="run" name="run">{run_options}</select>

    <label for="a">Episode A</label>
    <select id="a" name="a"></select>

    <label for="b">Episode B (optional, for side-by-side comparison)</label>
    <select id="b" name="b"></select>

    <button class="submit" type="submit">View</button>
    <div class="hint">FK skeleton + camera frames render on the server on first view of an episode (a couple seconds); repeat views of the same episode are served from an in-memory cache for this server process's lifetime.</div>
  </form>
</div>
<script>
window.__runIndex = {run_index_json};
{INDEX_JS}
</script>
</body>
</html>"""


def render_view_html(cache: EpisodeCache, run: str, filenames: list[str]) -> str:
    episodes = []
    skeleton_results = []
    for fn in filenames:
        ep, skel_images, skel_error = cache.get(run, fn)
        episodes.append(ep)
        skeleton_results.append((skel_images, skel_error))
    return viewer.build_html(episodes, jpeg_quality=cache.jpeg_quality, skeleton_results=skeleton_results)


# --------------------------------------------------------------------------
# HTTP plumbing (stdlib only)
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "HilSerlEpisodeViewer/1.0"
    root: str = ""
    cache: EpisodeCache = None  # type: ignore[assignment]

    def _send_html(self, body: str, status: int = 200) -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 (stdlib-mandated name)
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                runs = scan_runs(self.root)
                self._send_html(render_index_html(runs, self.root))
            elif parsed.path == "/view":
                run = (qs.get("run") or [""])[0]
                a = (qs.get("a") or [""])[0]
                b = (qs.get("b") or [""])[0]
                runs = scan_runs(self.root)
                if run not in runs or a not in runs.get(run, []):
                    self._send_html(
                        f"<p>error: unknown run/episode (run={html.escape(run)!r}, a={html.escape(a)!r}). "
                        f"<a href='/'>Back</a></p>",
                        status=404,
                    )
                    return
                filenames = [a] + ([b] if b and b in runs.get(run, []) else [])
                self._send_html(render_view_html(self.cache, run, filenames))
            else:
                self._send_html("<p>404 not found. <a href='/'>Back</a></p>", status=404)
        except Exception as exc:  # keep the server alive, surface the error in-browser
            traceback.print_exc()
            self._send_html(f"<pre>error: {html.escape(str(exc))}</pre>", status=500)

    def log_message(self, fmt: str, *args) -> None:  # quieter default access log
        sys.stderr.write(f"[http] {self.address_string()} {fmt % args}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="outputs/hilserl_bimanual_runs", help="Root dir to scan for */actor_trajectories/*.npz (default: outputs/hilserl_bimanual_runs).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1, local-only).")
    parser.add_argument("--port", type=int, default=8787, help="Bind port (default: 8787).")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="JPEG quality for embedded camera frames (default 80).")
    parser.add_argument("--skeleton", action=argparse.BooleanOptionalAction, default=True, help="Render the FK skeleton pane (default: on; see hilserl_episode_viewer.py).")
    parser.add_argument("--urdf-path", default=viewer.DEFAULT_URDF_PATH, help="URDF for FK skeleton pane.")
    parser.add_argument("--cache-size", type=int, default=6, help="Max episodes kept in the in-memory LRU cache at once (default 6; process-lifetime only, never written to disk).")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"error: --root does not exist or is not a directory: {root}", file=sys.stderr)
        return 1

    runs = scan_runs(root)
    total_eps = sum(len(fs) for fs in runs.values())
    print(f"Root: {root}", file=sys.stderr)
    print(f"Found {len(runs)} runs, {total_eps} episodes total.", file=sys.stderr)
    for r, fs in list(runs.items())[:5]:
        print(f"  {r}: {len(fs)} episodes", file=sys.stderr)
    if len(runs) > 5:
        print(f"  ... and {len(runs) - 5} more runs", file=sys.stderr)
    print(
        f"Cache: in-memory only, up to {args.cache_size} episodes, process-lifetime "
        f"(no disk cache directory is created by this server).",
        file=sys.stderr,
    )
    print(f"Real control frequency used for 1x/real-time playback: {viewer.CONTROL_FREQUENCY_HZ:g} Hz ({viewer.REAL_FRAME_INTERVAL_MS:g} ms/frame).", file=sys.stderr)

    cache = EpisodeCache(
        root=root,
        maxsize=args.cache_size,
        skeleton=args.skeleton,
        urdf_path=args.urdf_path,
        jpeg_quality=args.jpeg_quality,
    )

    Handler.root = root
    Handler.cache = cache
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving on http://{args.host}:{args.port}/  (Ctrl-C to stop)", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
