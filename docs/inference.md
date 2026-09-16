[← Back to README](../README.md)

# Run Inference

All inference scenarios go through `scripts/run_inference.sh`.

**Start by copying `.env.example` to `.env` and editing it for your setup:**

```bash
cp .env.example .env
# then edit .env
```

### `.env` Variables

`.env` is the primary configuration file — set it once and reuse across runs. Key variables:

| Variable | Required | Description |
|----------|----------|-------------|
| `MODEL_PATH` | Yes (inference) | Host path to checkpoint dir. Must be absolute or start with `./` — bare relative paths are treated as Docker named volumes. |
| `ROS_DOMAIN_ID` | Yes | ROS2 domain ID — must match the Anvil Devbox. Leave empty for localhost-only. |
| `CYCLONEDDS_URI` | Yes | Path to CycloneDDS XML config (e.g. `configs/cyclonedds/two_pc_gpu.xml`). |
| `LEROBOT_EXTRAS` | VLA only | Comma-separated policy extras built into the Docker image — e.g. `smolvla`, `pi,smolvla`, `groot`. **Rebuild the image after changing:** `docker compose build`. ACT and Diffusion leave this empty. |
| `HF_CACHE` | VLA only | Host path to HuggingFace model cache (default: `~/.cache/huggingface`). Required for Pi0, Pi0.5, SmolVLA (PaliGemma tokenizer) and GR00T (`nvidia/GR00T-N1.5-3B` + Eagle tokenizer assets), all loaded at runtime. |

> **GR00T note.** `LEROBOT_EXTRAS=groot` triggers a dedicated image layer that
> pins **torch 2.9.0+cu130** and installs a prebuilt flash-attn wheel, because
> GR00T's Eagle-2.5-VL backbone hard-requires flash-attn and the wheel is only
> published for torch 2.9. This pin applies *only* to images built with `groot`
> in `LEROBOT_EXTRAS` — ACT/Diffusion/Pi0/Pi0.5/SmolVLA images are unaffected.
> See [training.md](training.md#groot-flash-attn-requirement) for details.

> **GR00T runs on the synchronous inference path (like ACT/Diffusion), not
> the background-thread (RTC) path.** This was tried and reverted. Moving
> GR00T onto the decoupled thread (a "sham" RTC contract, since it has no
> real gradient-guided smoothing) did fix a real problem — measured directly
> from a real robot run, the synchronous path's control loop ran at a median
> ~7.4Hz with stalls up to 3.2s, because GR00T's real per-inference
> action-chunk length (~16 steps, governed by the checkpoint's own
> `action_horizon`, not `GrootConfig.n_action_steps`) gives far less
> execution runway (~0.53s) than its own inference latency (~3-4s) needs.
> But real-robot testing of the decoupled-thread version surfaced a worse
> problem: severe jerky, discontinuous motion at every chunk boundary. Root
> cause: the background thread re-triggers a new inference as soon as a
> chunk is merged (GR00T's ~16-action chunk is always under the trigger
> threshold), using an observation captured from *before* that chunk even
> started executing. The robot then moves through the chunk while inference
> runs in parallel, so by the time the next chunk arrives ~3-4s later, it's
> driving from a robot state the model never actually saw — producing large,
> independent absolute-position jumps at every chunk boundary, rate-limited
> only by the position-delta safety clamp (fast enough to look violent). The
> synchronous path avoids this because the robot is frozen for the entire
> inference duration, so the observed state can never go stale relative to
> the robot's actual position — the tradeoff is the 7.4Hz stalling above.
> Actually fixing this would need real RTC-style guidance (blending toward
> continuity across chunks, not just a queue trick) or a much faster GR00T
> forward pass (fewer denoising steps, quantization, compiled backbone) —
> both out of scope for now. `patch_groot_for_rtc_dispatch()` in
> `model_loader.py` still exists (unused, not called) with the full history
> of both attempts in its docstring, in case this is revisited. GR00T also
> currently requires absolute actions — see `_setup_vla_inference`'s guard in
> `inference_node.py` for why delta actions aren't supported on the RTC path.

| `CONFIG_FILE` | Yes | Path to inference config YAML (default: `configs/lerobot_control/inference_default.yaml`). |
| `ACTION_TYPE` | No | Action type passed to the **inference monitor node** (`inference_monitor_node`) only. The main inference node always reads this from `anvil_config.json` in the checkpoint via `resolve_action_type()` — this env var does **not** override it. Options: `absolute` · `delta_obs_t` · `delta_sequential`. |
| `ECHO_TOPIC_ONLY` | No | `true` = skip model loading, subscribe topics and log FPS only. For verifying DDS connectivity without a GPU or checkpoint. Equivalent to `--echo-topic-only`. |
| `MONITOR_ENABLE` | No | `true` = enable the inference monitor node (records per-step CSV + PNG report). Equivalent to `--monitor-enable`, but without the auto-plot on exit and output dir pre-creation that the flag provides. |
| `DEBUG` | No | `true` = enable extra metrics: action smoothness, queue depth stats, Action FPS. |
| `FOXGLOVE_PORT` | No | Websocket port for the `--foxglove` camera-viz service (default `8765`). |
| `INFERENCE_GPU_ID` | No | GPU index the inference container uses (default `0`). Override if that GPU is busy on a shared machine. |
| `INFERENCE_TORCH_THREADS` | No | Caps OMP/MKL/NUMEXPR thread pools inside the container (default `4`) — see the `cpuset`/thread-capping comments in `docker-compose.yml` for why this matters on a shared multi-core machine. |

For full descriptions and defaults, see [`.env.example`](../.env.example).

### Script Flags

The script flags are a lightweight way to override behaviour at the command line without editing `.env`:

```bash
./scripts/run_inference.sh [--fake-hardware] [--monitor-enable] [--echo-topic-only] [--debug] [COMPOSE_ARGS...]
```

| Flag | What it does |
|------|-------------|
| `--fake-hardware` | Switches to `docker-compose.fake-hardware.yml` — simulates a 2-PC setup locally over a bridge network (CycloneDDS, no real robot). |
| `--monitor-enable` | Adds `--profile monitor` to the compose command. In production (non-fake-hardware) mode also exports `MONITOR_ENABLE=true`, pre-creates the output dir as the current user, and auto-plots the CSV on exit. |
| `--foxglove` | Adds `--profile viz` to the compose command — starts a `foxglove_bridge` websocket relay so camera feeds and joint/command topics can be viewed live during inference. See [Live Camera Viz](#live-camera-viz-foxglove) below. |
| `--echo-topic-only` | Exports `ECHO_TOPIC_ONLY=true` — subscribes to topics and logs FPS without loading a model or GPU. Equivalent to setting `ECHO_TOPIC_ONLY=true` in `.env`. |
| `--debug` | Exports `DEBUG=true` — enables extra metrics: action smoothness, queue depth stats, Action FPS. Equivalent to setting `DEBUG=true` in `.env`. |

## Test with Fake Hardware First (Recommended)

```bash
# 1. Verify DDS connectivity + camera FPS (no model, no GPU needed)
./scripts/run_inference.sh --fake-hardware --monitor-enable up --build

# 2. Validate full pipeline with your model (GPU required)
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh --fake-hardware --profile inference up --build
```

> **Fake-hardware note:** `--echo-topic-only` / `ECHO_TOPIC_ONLY` and `MONITOR_ENABLE` env vars are
> **not** read by `docker-compose.fake-hardware.yml`. The monitor service hardcodes
> `echo_topic_only:=true` regardless; the inference service does not expose `MONITOR_ENABLE`.
> These variables only take effect with the production `docker-compose.yml`.

If `Control Loop` hits 30 Hz, the setup is ready for real hardware.

## Production (Real Robot)

```bash
# Standard inference
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh up --build

# With inference monitor
MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last \
./scripts/run_inference.sh --monitor-enable up --build

# Verify DDS connectivity without a checkpoint
./scripts/run_inference.sh --echo-topic-only up --build
```

> **`MODEL_PATH` must be absolute or start with `./`.** Bare relative paths are treated as named Docker volumes.
> ```bash
> MODEL_PATH=$(pwd)/model_zoo/my-task/checkpoints/last   # recommended
> MODEL_PATH=./model_zoo/my-task/checkpoints/last        # also valid
> ```

## Inference Config (`configs/lerobot_control/inference_default.yaml`)

Before running, review this file:

**Model**
```yaml
model:
  task_description: null
  # VLA-only (SmolVLA / Pi0 / Pi0.5): task prompt the model was trained on.
  # null = auto-read from anvil_config.json in the checkpoint (recommended).
```

**Per-model inference tuning** — override checkpoint defaults without retraining:
```yaml
inference_tuning:

  act:
    n_action_steps: null
    # Steps to execute per chunk before re-running inference.
    # null = use training value. Jittery? → raise. Hesitates? → lower.
    temporal_ensemble_coeff: null
    # Re-infers every step with exponentially weighted overlapping predictions.
    # Use 0.01 (paper default). Forces n_action_steps=1.

  diffusion:
    n_action_steps: null
    # Steps to execute per chunk. null = use training value.
    num_inference_steps: 10
    # Denoising iterations at inference time.
    # null = num_train_timesteps (100 steps, ~300ms on GPU).
    # 10   = ~30ms on GPU — recommended for real-time deployment.

  smolvla:
    # Model-specific fields here are forwarded generically to the loaded
    # policy config if it has a matching attribute — same mechanism covers
    # Pi0/Pi0.5 under their own `pi0:`/`pi05:` keys if needed.
    num_steps: null
    # Euler denoising iterations per inference call (flow matching).
    # null = training-time value (10). Pure inference-time hyperparameter —
    # safe to lower without retraining, same tradeoff as Diffusion's
    # num_inference_steps above. Real RTC guidance runs a backward pass per
    # step, so this is markedly more expensive than plain forward denoising —
    # lowering it (e.g. 5) cuts per-call latency roughly proportionally if
    # real latency is cutting into your chunk_size's execution budget.

  rtc:
    # VLA models only (SmolVLA / Pi0 / Pi0.5)
    inference_delay: 10
    # Fallback step-count before LatencyTracker auto-calibrates.
    # Rule of thumb: ceil(first_inference_ms × control_freq / 1000)
    queue_trigger_threshold: 300
    # Re-trigger inference when ActionQueue depth ≤ this. Needs to leave
    # enough head start (threshold / control_freq seconds) to cover your
    # model's real per-call inference latency, independent of chunk_size —
    # too low a threshold starves the queue and shows up as stuttering
    # motion even when chunk_size itself is generously sized. 300 here
    # reflects a much larger/slower model (pi0.5, chunk_size=400); a small
    # fast model like SmolVLA doesn't need it this high, but a larger shared
    # value is harmless (just means more continuous background inference).
    execution_horizon: 12
    # Steps consumed per chunk before the next inference fires.
    max_guidance_weight: 10.0
    prefix_attention_schedule: EXP
```

**Safety limits:**
```yaml
# safety:
#   max_position_delta: 0.1
#   # Hard limit on joint position change per control step (radians).
#   min_position_delta: 0.05
#   # Minimum cumulative change before publishing a new command.
#   # Holds the last command until threshold is crossed — useful for
#   # overcoming motor dead zones / friction. Default: disabled (null).
```

## Live Camera Viz (Foxglove)

To watch camera feeds and joint/command topics live while inference is
running — useful for visually diagnosing what a policy is looking at when it
misbehaves — add `--foxglove`:

```bash
COMPOSE_MENU=false ./scripts/run_inference.sh --monitor-enable --foxglove up --build
```

This starts a `foxglove_bridge` container that relays whatever's on the DDS
domain (all four camera topics, `/joint_states`, and the command topics) to a
websocket, independent of the inference container — it's a second subscriber
on the same DDS bus, not something the inference pipeline routes through, so
it works whether or not inference happens to be running, and doesn't block or
steal frames from it.

Connect from **Foxglove Studio** (app.foxglove.dev, or the desktop app) to
`ws://<this-host-ip>:8765` (override the port with `FOXGLOVE_PORT` in
`.env`). Add an "Image" panel per camera topic to watch all four feeds side
by side.

> **Bandwidth caution:** `foxglove_bridge` relays full-resolution camera
> streams, which adds real CPU/network load on top of whatever the inference
> container's own image workers are already consuming. On this project,
> running `--foxglove` during a real-robot test measurably worsened an
> already-chronic camera FPS bottleneck (from ~14-20Hz down to ~4-10Hz),
> which is enough to visibly slow down a synchronous-path policy like ACT
> (whose control loop runs at the same rate as incoming camera frames).
> Prefer using `--foxglove` for one-off visual debugging sessions rather than
> leaving it on during performance-sensitive real-robot tests.

The `--fake-hardware` compose file also supports `--foxglove` (using mock
camera data from `mock_controller_node`) if you want to verify the viz
pipeline itself without a real robot — in that mode the port is explicitly
mapped (`ports:` in `docker-compose.fake-hardware.yml`) since the fake-hardware
stack runs on a bridge network rather than the host network production uses.

## DDS Middleware Selection

Both Fast DDS and CycloneDDS are supported. **CycloneDDS is the default** (faster in our tests).

> ⚠ **Both sides must use the same RMW** — mixing Fast DDS and CycloneDDS = zero discovery (no error, just silence).

| Deployment | `RMW_IMPLEMENTATION` | `CYCLONEDDS_URI` | anvil-loader `.env.config` |
|-----------|----------------------|------------------|---------------------------|
| **Single-PC · CycloneDDS** *(default)* | `rmw_cyclonedds_cpp` | `file://.../single_pc.xml` | `ENABLE_CYCLONEDDS=true`<br>`CYCLONEDDS_PEER_IP=127.0.0.1` |
| Single-PC · Fast DDS | `rmw_fastrtps_cpp` | *(ignored)* | `ENABLE_CYCLONEDDS=false` |
| Two-PC · CycloneDDS | `rmw_cyclonedds_cpp` | `file://.../two_pc_gpu.xml` | `ENABLE_CYCLONEDDS=true`<br>`CYCLONEDDS_PEER_IP=<gpu_pc_ip>` |

All CycloneDDS configs live in `configs/cyclonedds/`. The defaults in `docker-compose.yml` and `.env.example` target single-PC CycloneDDS — override in `.env` to switch modes.

## Deployment Topologies

### Single-PC — inference and workcell on the same machine

```
  Same machine
┌────────────────────────────────────────────────────────────┐
│  anvil-loader (ros2_control)       anvil-embodied-ai       │
│  joint_states (500 Hz)  ◄─────────  inference_node (30 Hz) │
│  cameras (4× 30 Hz)      CycloneDDS  action commands       │
│                           (host net)                       │
└────────────────────────────────────────────────────────────┘
```

Both sides use CycloneDDS on the host network — multicast handles peer discovery automatically. Set in anvil-loader's `.env.config`:
```
ENABLE_CYCLONEDDS=true
CYCLONEDDS_PEER_IP=127.0.0.1
```

### Two-PC — GPU PC separate from the robot PC

```
  Anvil Devbox (anvil-loader)             CycloneDDS              GPU PC (anvil-embodied-ai)
┌─────────────────────────────┐    ┌────────────────────┐    ┌─────────────────────────────┐
│  ros2_control               │    │                    │    │  lerobot_control            │
│  joint_states (500 Hz)      │◄───┤  Gigabit Switch    ├───►│  inference_node (30 Hz)     │
│  cameras (4× 30 Hz)         │    │                    │    │  action commands            │
└─────────────────────────────┘    └────────────────────┘    └─────────────────────────────┘
```

Set `CYCLONEDDS_URI=file:///workspace/configs/cyclonedds/two_pc_gpu.xml` and configure peer IPs in both `two_pc_gpu.xml` and anvil-loader's `.env.config`. See the [full documentation](https://docs.anvil.bot/) for network setup.

---

[← Back to README](../README.md)
