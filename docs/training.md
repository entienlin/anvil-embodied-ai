[← Back to README](../README.md)

# Model Training

All training runs through the `anvil-trainer` CLI — a thin wrapper around LeRobot's `lerobot-train` that adds Anvil-specific transforms, data splits, and checkpoint management.

## Contents

- [The anvil-trainer CLI](#the-anvil-trainer-cli)
- [Supported Policies](#supported-policies)
- [Common Parameters](#common-parameters)
  - [LeRobot Defaults](#lerobot-defaults-auto-set-by-anvil-trainer)
  - [Action Type](#action-type)
  - [Normalization Mapping](#normalization-mapping)
  - [Weights & Biases](#weights--biases)
  - [Data Augmentation](#data-augmentation)
  - [Data Filter](#data-filter)
- [Policy Models](#policy-models)
  - [ACT](#act)
  - [Diffusion](#diffusion)
  - [SmolVLA](#smolvla)
  - [Pi0.5](#pi05)
- [Outputs](#outputs)
  - [Structure](#structure)
  - [Loss Reading](#loss-reading)
  - [Fine-tune](#fine-tune)
  - [Resume](#resume)

---

## The anvil-trainer CLI

`anvil-trainer` wraps LeRobot's `lerobot-train`. It strips out Anvil-specific flags before passing the rest through to LeRobot's own CLI parser.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --job_name=my-run
```

---

## Supported Policies

| Policy | `--policy.type` | Notes |
|--------|----------------|-------|
| ACT | `act` | Action Chunking Transformer — fast, reliable baseline |
| Diffusion | `diffusion` | Diffusion Policy — smooth, handles multimodal distributions |
| SmolVLA | `smolvla` | Language-conditioned VLA; requires `--extra smolvla` |
| Pi0 | `pi0` | Flow-matching VLA; PaliGemma-3B backbone; requires `--extra pi` |
| Pi0.5 | `pi05` | Larger Pi0 variant (~4B params); higher VRAM; requires `--extra pi` |
| GR00T | `groot` | NVIDIA GR00T N1.5 (~3B); Eagle-2.5-VL backbone; requires `--extra groot` **and** flash-attn (see below) |
| OpenVLA-OFT | `openvla` | Out-of-tree plugin (`lerobot_policy_openvla`); ~7.5B params; runs in a **separate** inference image — see below |

Checkpoints are saved to `model_zoo/<dataset>/<job_name>/`. Run `uv run anvil-trainer --help` for the full flag reference.

### GR00T: flash-attn requirement

GR00T is the one policy with an install step beyond its extra. Its Eagle-2.5-VL
backbone hard-requires flash-attn — `modeling_groot.py` prints a "fallback
attention" message but never actually falls back, and the Eagle model asserts
`flash_attention_2`. `--extra groot` deliberately omits flash-attn because it is
sdist-only on PyPI and building it needs a CUDA toolkit with **nvcc >= 12.8** to
emit `sm_120` kernels for Blackwell GPUs (an older nvcc builds without error but
silently omits them, failing at load time with *"no kernel image is available
for execution on the device"*).

Install the official prebuilt wheel instead, which pins the torch version:

```bash
uv sync --all-packages --extra groot
uv pip install --index-url https://download.pytorch.org/whl/cu130 \
    torch==2.9.0 torchvision==0.24.0 torchcodec==0.9.0
uv pip install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

The inference image does this automatically when `LEROBOT_EXTRAS` contains
`groot`; see `docker/inference/Dockerfile`. GR00T also downloads its base model
(`nvidia/GR00T-N1.5-3B`) and tokenizer assets
(`lerobot/eagle2hg-processor-groot-n1p5`) from the Hub on first load, so
`HF_CACHE` must be set when running in Docker.

### OpenVLA-OFT: separate environment, not an `--extra`

> **Status: verified skeleton, not yet deployable.** Real weights, real
> forward pass, real dispatch, real Docker build all verified on this
> machine — but no rig fine-tune exists (heads are random-init), no training
> path is implemented, and RTC support is interface-only (no real cross-chunk
> guidance). Full checklist and path to a real deployment:
> [`packages/lerobot_policy_openvla/README.md`](../packages/lerobot_policy_openvla/README.md#status-as-of-2026-08-28).

OpenVLA-OFT (`moojink/openvla-oft`, MIT) doesn't exist in lerobot at all —
it's an out-of-tree package, `packages/lerobot_policy_openvla/`, discovered
via lerobot's `register_third_party_plugins()`. Unlike GR00T, it can't share
an environment with the rest of the stack: its parallel-decoding action head
needs bidirectional attention over action-chunk tokens, which only exists in
openvla-oft's own transformers fork (`moojink/transformers-openvla-oft`) —
an old, full transformers snapshot (~4.40, ~8000 commits behind upstream),
not a small patch, and it hard-conflicts at import time with the
`transformers==5.3.0` every other policy needs. Verified directly: real
`openvla/openvla-7b` (7.54B params) loaded via this fork, running
bidirectional attention on an RTX PRO 6000 Blackwell (sm_120), on
**torch==2.9.0+cu130** — a full torch generation newer than the fork's own
`torch==2.2.0` pin, which can't target sm_120 at all (added in PyTorch
2.7+/CUDA 12.8, same class of issue as GR00T's flash-attn wheel).

Build and run it with its own image:

```bash
docker build -f docker/inference-openvla/Dockerfile -t lerobot-inference-openvla .
docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /path/to/checkpoint:/workspace/model:ro \
    lerobot-inference-openvla
```

Sized for the OpenArm rig by default: 16-dim action/proprio (2×7 joints + 2
grippers), vs. openvla-oft's own ALOHA default of 14 (2×6 + 2) — a real,
supported resize (`action_dim`/`proprio_dim` in `OpenVLAConfig`), not a
hardcoded architecture assumption. No rig-specific fine-tune exists yet, so
`action_head`/`proprio_projector` weights are random-initialized until one is
trained via openvla-oft's own `vla-scripts/finetune.py`
(see `ALOHA.md` in that repo — `--use_l1_regression True --use_proprio True`).
`use_film` (FiLM language conditioning) defaults to `False`: real FiLM
requires a fine-tuned checkpoint's own saved vision-backbone weights, which
don't exist for the base checkpoint.

**RTC honesty note.** `OpenVLAPolicy` implements `init_rtc_processor()` /
`predict_action_chunk()` so Phase 0's dispatch layer routes it onto the
decoupled background-inference-thread path (avoiding blocking the 30Hz sync
loop on ~500ms VLM latency) — a real, valuable win. It does **not** perform
lerobot's actual RTC cross-chunk gradient-guided smoothing: that's a
flow-matching/diffusion technique, and OFT's default L1-regression action
head is a single deterministic MLP pass with no denoising loop to guide.

**Training is not implemented.** `OpenVLAPolicy.forward()` raises
`NotImplementedError` rather than silently computing a detached, zero-gradient
loss — `vla.predict_action()` (the call `predict_action_chunk` uses) does its
unnormalization in numpy internally, so building a loss on top of it can't
backprop. A real implementation needs openvla-oft's own differentiable
training codepath (`vla-scripts/finetune.py`), not `predict_action()`.

---

## Common Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset.root=PATH` | _(required)_ | Path to converted LeRobot dataset |
| `--policy.type=TYPE` | _(required)_ | Policy type (see table above) |
| `--job_name=NAME` | `<policy>_<timestamp>` | Checkpoint directory name |
| `--steps=N` | `100000` | Total training steps |
| `--batch_size=N` | `8` | Reduce if GPU OOM |
| `--log_freq=N` | `200` | Log train loss every N steps; val loss every `log_freq × 5` steps |
| `--split-ratio=T,V,S` | `8,1,1` | Train/val/test episode split. Two values = no test set. |
| `--max-episodes=N` | all | Subsample N episodes before splitting (reproducible with training seed) |
| `--backbone=NAME` | `resnet18` | Vision backbone for ACT/Diffusion: `resnet18` · `resnet34` · `resnet50`. Ignored for VLA policies (Pi0, Pi0.5, SmolVLA). Under the hood this injects `--policy.vision_backbone`, `--policy.pretrained_backbone_weights` (ImageNet), and for Diffusion also `--policy.use_group_norm=false`. |
| `--save_freq=N` | `10000` | Save a checkpoint every N steps. Lower (e.g. `5000`) for unstable runs; higher (e.g. `25000`) if disk is tight — each checkpoint can be several GB |
| `--resume=PATH` | — | Resume from job root or specific checkpoint |

### LeRobot Defaults (auto-set by anvil-trainer)

These are LeRobot's own flags that `anvil-trainer` sets automatically so you don't have to repeat them. Any of them can be overridden by passing the flag explicitly.

| Flag | Auto value | Why |
|---|---|---|
| `--dataset.repo_id` | `local` | Anvil datasets are always local |
| `--policy.push_to_hub` | `false` | Prevents accidental HF Hub uploads |
| `--eval_freq` | `0` | Disables gym eval (no sim env for MCAP datasets) |
| `--wandb.project` | `<dataset folder name>` | Groups all runs for the same task together |
| `--output_dir` | `model_zoo/<dataset>/<job_name>` | Nested under dataset name |
| `--policy.vision_backbone` + `--policy.pretrained_backbone_weights` | `resnet18` + ImageNet weights | Injected from `--backbone` (ACT/Diffusion only) |
| `--policy.use_group_norm` | `false` | Injected for Diffusion when using a pretrained backbone |

---

### Action Type

Controls how actions are encoded. The chosen mode is persisted to `anvil_config.json` in each checkpoint — inference applies the inverse automatically, no manual YAML change needed.

| `--action-type` | Formula | When to use |
|---|---|---|
| `absolute` (default) | Raw joint positions | Simplest; works well for ACT and Diffusion |
| `delta_obs_t` | `Δ[k] = action[t+k] − obs_state[t]` | Tasks with repeated returns to similar poses; all steps share the same obs reference |
| `delta_sequential` | `Δ[0] = action[0] − obs_state[t]`; `Δ[k] = action[k] − action[k−1]` | Encodes velocity; smoother trajectories since consecutive deltas are small |

```bash
# delta_obs_t shorthand (legacy)
uv run anvil-trainer ... --use-delta-actions

# delta_obs_t (explicit)
uv run anvil-trainer ... --action-type=delta_obs_t

# delta_sequential
uv run anvil-trainer ... --action-type=delta_sequential
```

Additional delta flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--delta-exclude-joints=J1,J2` | — | Keep these joints absolute (e.g. `finger_joint1` for grippers) |
| `--delta-stats-n-steps=N` | `1` | Look-ahead steps for delta normalizer stats. Increase to cover multi-step displacement range |

---

### Normalization Mapping

`--policy.normalization_mapping='{"ACTION":"...","STATE":"...","VISUAL":"..."}'`

| Value | Description |
|-------|-------------|
| `MEAN_STD` | Normalize by μ/σ |
| `MIN_MAX` | Normalize to [−1, 1] by observed min/max |
| `IDENTITY` | Passthrough — always use for `VISUAL` |

**Guidance by policy:**
- **Diffusion** → `ACTION: MIN_MAX`. Diffusion clips denoised actions to ±1 at every step (`clip_sample=True`); `MEAN_STD` silently truncates extreme actions.
- **ACT / SmolVLA / Pi0 / Pi0.5** → `ACTION: MEAN_STD`

> **Pi0.5 note:** Pi0.5's default normalization is `QUANTILE10`, which requires `q01`/`q99` fields in `stats.json`. Datasets converted with `mcap-convert` do not include these. Use `MEAN_STD` instead (recommended), or see [Pi0.5](#pi05) for the quantile augmentation option.

---

### Weights & Biases

```bash
uv run wandb login   # one-time setup

uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --wandb.enable=true
```

W&B project is auto-set to the dataset folder name; run name to `<policy>_<timestamp>`. Override with `--wandb.project=NAME` or `--job_name=NAME`.

| Flag | Description |
|------|-------------|
| `--wandb.enable=true` | Enable W&B logging |
| `--wandb.project=NAME` | Project name (auto-set to dataset folder name) |

Key metrics to watch:

| Metric | What it tells you |
|---|---|
| `train/loss` | Overall training loss — should decrease steadily |
| `train/grad_norm` | Gradient norm — spikes indicate instability; try lowering LR |
| `eval/val_loss` | Validation loss — computed every `log_freq × 5` steps |
| `eval/test_loss` | Test loss — computed at every checkpoint (`save_freq`) |

---

### Data Augmentation

Two built-in augmentation layers, both disabled by default. Can be combined with any policy.

**Layer 1 — Color Augmentation (all policies)**

Randomly applies up to `max_num_transforms` color transforms per image at training time:

| Transform | Range |
|-----------|-------|
| Brightness | [0.8, 1.2] |
| Contrast | [0.8, 1.2] |
| Saturation | [0.5, 1.5] |
| Hue | [−0.05, 0.05] |
| Sharpness | [0.5, 1.5] |
| Affine | ±5° rotation, 5% translation |

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3
```

**Layer 2 — Random Crop (Diffusion only)**

Diffusion's `DiffusionRgbEncoder` applies `RandomCrop` during training and `CenterCrop` during inference — the switch is automatic, no inference-time config needed.

> **`--policy.resize_shape` is required to activate crop.** LeRobot only derives `crop_shape` from `crop_ratio` when `resize_shape` is also set — `crop_is_random`/`crop_ratio` alone are silently ignored, with no warning or error. `resize_shape` takes `(H, W)`; match it to your dataset's stored resolution (e.g. `[270, 480]` for a 480×270 16:9 dataset).

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=diffusion \
  --policy.resize_shape='[270,480]' \
  --policy.crop_is_random=true \
  --policy.crop_ratio=0.9
```

`crop_ratio=0.9` crops to 90% of the original image size. Combine both layers for best generalization:

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=diffusion \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=3 \
  --policy.resize_shape='[270,480]' \
  --policy.crop_is_random=true \
  --policy.crop_ratio=0.9
```

---

### Data Filter

**`--exclude-observs=SUFFIX,...`** — Drop observation keys by suffix after `observation.`. Also reads `LEROBOT_EXCLUDE_OBSERVS` env var.

The suffix is everything after `observation.` in the full dataset key:

| Suffix (what you pass) | Full dataset key |
|------------------------|-----------------|
| `images.wrist_r` | `observation.images.wrist_r` |
| `images.chest` | `observation.images.chest` |
| `images.waist` | `observation.images.waist` |
| `velocity` | `observation.velocity` _(optional)_ |
| `effort` | `observation.effort` _(optional)_ |

`velocity` and `effort` are only present in datasets converted with joint feedback enabled.

```bash
uv run anvil-trainer ... --exclude-observs=images.wrist_r                    # drop one camera
uv run anvil-trainer ... --exclude-observs=images.wrist_r,images.chest       # drop multiple cameras
uv run anvil-trainer ... --exclude-observs=velocity,effort                   # drop optional joint feedback keys
uv run anvil-trainer ... --exclude-observs=images.chest,velocity,effort      # mixed
```


LeRobot also always writes a `last/` symlink at the end of training pointing to the final checkpoint.

---

## Policy Models

### ACT

Action Chunking Transformer. Best starting point for new tasks — fast to train and reliable.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=act \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --wandb.enable=false
```

**chunk_size**

Defaults to `100`. Controls how many future actions the model predicts per forward pass — directly affects model architecture and training loss.

- **Fast, fine-grained tasks** (small precise movements): `chunk_size=50` — shorter prediction horizon.
- **Slow, sweeping tasks**: higher values (100+) reduce jitter.

```bash
--policy.chunk_size=50
```

> `n_action_steps` (how many of those actions are executed before re-querying) is an inference setting, not a training parameter — the training loss does not use it. It is baked into `config.json` with a default value at training time, but tuned at inference via `inference_tuning.n_action_steps` in the inference YAML without retraining.

**kl_weight**

Controls VAE regularization. Default `10.0` works well. Increase (20–50) if actions are jerky; decrease if the model underfits.

**Steps and batch size**

100k steps / batch 16 is a solid default. For small datasets (< 50 episodes), 50k steps is often enough.

**Data quality**

ACT is sensitive to demonstration quality. A small set of clean, consistent demos outperforms a large set of sloppy ones. Discard failed or hesitant episodes before training.

---

### Diffusion

Diffusion Policy models the action distribution as a denoising process. Use it when multiple valid trajectories exist (e.g. approaching an object from several angles) — it produces smooth, natural motions without explicit chunk tuning. Trade-off: inference is slower due to the denoising loop.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=diffusion \
  --policy.normalization_mapping='{"ACTION":"MIN_MAX","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --policy.horizon=24 \
  --policy.down_dims='[256,512,1024]' \
  --backbone=resnet18 \
  --wandb.enable=false
```

`--backbone=resnet18` auto-injects `--policy.vision_backbone`, `--policy.pretrained_backbone_weights`, and `--policy.use_group_norm=false` — no need to pass them individually. To switch backbone: `--backbone=resnet34` or `--backbone=resnet50`.

**Hyperparameters — for datasets under ~500 episodes:**

| Flag | Default | Recommended | Why |
|------|---------|-------------|-----|
| `--policy.horizon` | `16` | `24` | Longer horizon gives UNet more temporal context |
| `--policy.down_dims` | `[512,1024,2048]` | `[256,512,1024]` | Smaller UNet reduces overfitting on small datasets |
| `--backbone` | `resnet18` | `resnet18` | Auto-disables GroupNorm for pretrained ImageNet weights |

**Image resolution — avoid letterbox waste**

If your cameras natively output 1920×1080 (16:9) but the dataset was converted with the default `image_resolution: [640, 480]` (4:3), the converter's aspect-preserving resize pads roughly 25% of every frame with black bars — wasted activation memory and diluted visual signal, for zero information gained.

Use the matching `_16x9` converter config instead — e.g. `configs/mcap_converter/openarm_bimanual_quest_16x9.yaml` — which sets `image_resolution: [480, 270]`, an exact ÷4 downscale of 1920×1080 with zero padding:

```bash
uv run mcap-convert -i data/raw_sessions/my-session \
  --config configs/mcap_converter/openarm_bimanual_quest_16x9.yaml \
  -o data/datasets
```

This requires reconverting the dataset — the black bars are baked into the stored video pixels, so `--policy.resize_shape` at train time can only shrink them further, it can't recover the wasted 25%.

**Steps and batch size**

100k steps is a solid default. Diffusion benefits more from larger batch sizes than ACT — this reduces score-matching variance and stabilizes training.

Two levers raise the practical batch-size ceiling on a 24 GB GPU well past the old 16–32 rule of thumb:

| Lever | How | Effect |
|---|---|---|
| 16:9 dataset conversion | Use a `_16x9` converter config (above) | Removes ~25% letterbox waste, smaller per-image tensor |
| bf16 mixed precision | `export ACCELERATE_MIXED_PRECISION=bf16` before training | Roughly halves activation memory. Zero code change: `anvil-trainer` calls LeRobot's `lerobot_train()` directly, and `Accelerate()` reads this env var automatically |

```bash
export ACCELERATE_MIXED_PRECISION=bf16
uv run anvil-trainer \
  --dataset.root=data/datasets/my-16x9-dataset \
  --policy.type=diffusion \
  --policy.resize_shape='[270,480]' \
  --policy.crop_is_random=true --policy.crop_ratio=0.9 \
  --batch_size=48 \
  --steps=100000
```

Combining both, batch 48 (4 cameras, 2 obs steps, `resnet18`, `down_dims=[512,1024,2048]`, `horizon=16`) ran stably at ~1.5s/step on a single RTX 4090 (24 GB) — 6× the old default of 8. Batch 64 fit in VRAM without OOM but slowed to ~5s/step once the CUDA allocator started thrashing near the memory ceiling — worse throughput despite more headroom on paper. Treat that as a signal to back off, not a target: find your ceiling empirically with a short `--steps=200` dry run before committing to a long run, and watch step time, not just `nvidia-smi`, to catch allocator thrashing.

**Reference: official `diffusion_policy` benchmark settings**

For context on how these batch sizes compare to the published research, here are [Chi et al., 2023](https://arxiv.org/abs/2303.04137)'s own training settings for image-based tasks (PH = "proficient-human" demonstrations, the robomimic standard-quality demo set):

| Task | Demonstrations | Batch Size | Epochs | Source |
|---|---|---|---|---|
| Lift (PH) | 200 | 64 | 3050 | paper Table III; `diffusion_policy/config/train_diffusion_unet_hybrid_workspace.yaml` |
| Can (PH) | 200 | 64 | 3050 | same |
| Square (PH) | 200 | 64 | 3050 | same |
| Transport (PH) | 200 | 64 | 3050 | same |
| Tool Hang (PH) | 200 | 64 | 3050 | same |
| PushT (sim) | 200 per paper Table III; the shipped `diffusion_policy/config/task/pusht_image.yaml` caps training at `max_train_episodes: 90` — both numbers kept rather than merged, since the repo and paper disagree | 64 | 3050 | paper Table III; `config/task/pusht_image.yaml` |
| Real PushT (real robot) | 136 | 64 | 600 | paper text; `diffusion_policy/config/train_diffusion_unet_real_image_workspace.yaml` |

`batch_size=64` is the standard across nearly every published image task — our own batch 48 on a single 4090 is in the same range, not an outlier. Note the official repo trains by **epoch** (full passes over the dataset) rather than by step count like `lerobot`/`anvil-trainer`; converting these to an equivalent step count would require each task's per-episode frame count, which isn't published in the repo or paper, so it's deliberately omitted here rather than estimated.

---

### SmolVLA

Language-conditioned VLA. Always pass `--task-description` and start from the pretrained base.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=smolvla \
  --policy.pretrained_path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --task-description="Grab the gray doll and put it in the bucket" \
  --wandb.enable=false
```

**Pretrained weights**

Always fine-tune from `lerobot/smolvla_base` — training from scratch is not recommended. `--policy.load_vlm_weights=true` is required when loading from a SmolVLA checkpoint; without it only the VLM backbone loads and the action expert starts from random weights.

**Task description**

A clear, specific description improves performance significantly. The description is saved to `anvil_config.json` in the checkpoint and auto-loaded at inference. Mirror it in your inference YAML:

```yaml
model:
  task_description: "Grab the gray doll and put it in the bucket"
```

**Frozen layers**

By default, the vision encoder is frozen (`freeze_vision_encoder=true`) and only the action expert is trained. Only unfreeze if you have a large dataset and the visual domain differs significantly from the pretrained data.

**Steps**

30k–50k steps from a pretrained base is usually sufficient. The default LR scheduler decays over 30k steps, which aligns well with this range.

---

### Pi0.5

Flow-matching VLA (~4B params) built on a PaliGemma-3B backbone. Requires a 24 GB GPU.

**HuggingFace access required**

Pi0.5 downloads `google/paligemma-3b-pt-224` on first use. This model is gated — you need to:

1. Visit the model page and accept the license: [google/paligemma-3b-pt-224](https://huggingface.co/google/paligemma-3b-pt-224)
2. Log in from the CLI (one-time setup):

```bash
uv run huggingface-cli login
# Paste your HF token when prompted (get one at https://huggingface.co/settings/tokens)
```

After login, the model is cached locally and subsequent runs skip the download.

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.train_expert_only=true \
  --policy.normalization_mapping='{"ACTION":"MEAN_STD","STATE":"MEAN_STD","VISUAL":"IDENTITY"}' \
  --batch_size=16 \
  --num_workers=0 \
  --task-description="Grab the gray doll and put it in the bucket" \
  --wandb.enable=false
```

**Pretrained paths:**

| Path | Description |
|---|---|
| `lerobot/pi05_base` | General-purpose base — use this for new tasks |
| `lerobot/pi05_libero` | Pre-trained on the Libero benchmark dataset |

**Required flags on a 24 GB GPU:**

| Flag | Why |
|---|---|
| `--policy.dtype=bfloat16` | Halves VRAM — required to fit 4B model on 24 GB |
| `--policy.gradient_checkpointing=true` | Further reduces VRAM during backprop |
| `--batch_size=16` | Starting point — reduce if GPU OOM |
| `--num_workers=0` | Prevents CPU RAM OOM — forked workers each copy the full model |

**Normalization:**

Pi0.5's default normalization is `QUANTILE10`, which requires `q01`/`q99` stats not produced by `mcap-convert`. Two options:

**Option A — Override (recommended for Anvil datasets)**

Pass `MEAN_STD` for actions and states, which uses the existing mean/std stats. This is the approach shown in the command above.

**Option B — Augment the dataset with quantile stats**

```bash
uv run python -c "
from lerobot.datasets.v30.augment_dataset_quantile_stats import main
main()
" -- --repo-id=local/your-dataset
```

> **Warning:** this modifies the dataset in-place. Back up first: `cp -r data/datasets/my-dataset data/datasets/my-dataset.bak`

After augmentation you can omit `--policy.normalization_mapping` and use the default `QUANTILE10`.

---

## Outputs

### Structure

Checkpoints are written to `model_zoo/<dataset>/<job_name>/`:

```
model_zoo/
└── <dataset>/
    └── <job_name>/
        ├── checkpoints/
        │   ├── last -> 100000/          # symlink to latest checkpoint
        │   ├── 010000/
        │   │   └── pretrained_model/
        │   │       ├── config.json              # LeRobot policy config
        │   │       ├── model.safetensors        # Model weights
        │   │       ├── anvil_config.json        # action_type, task_description, code_commit
        │   │       ├── split_info.json          # train/val/test episode lists
        │   │       ├── policy_preprocessor.json # normalizer + resize config
        │   │       └── policy_postprocessor.json
        │   └── 100000/
        ├── train_config.json            # full training config (for resume)
        └── wandb/
```

---

### Loss Reading

Use `--split-ratio=TRAIN,VAL,TEST` (default `8,1,1`) to hold out episodes for validation and testing.

| Metric | When computed | What it tells you |
|---|---|---|
| `eval/val_loss` | Every `log_freq × 5` steps | Ongoing overfitting signal during training |
| `eval/test_loss` | Every checkpoint (`save_freq`) | More thorough evaluation on a completely held-out set |

**Diagnosing training health:**

- **High error on `train` split** → underfitting — model needs more steps or more capacity
- **Low `train` error but high `val`/`test` error** → overfitting — reduce steps, add augmentation, or collect more diverse data
- **`val_loss` rising while `train_loss` falls** → early overfitting signal — consider using the checkpoint just before the upturn

Use the checkpoint with the lowest `test_loss` for deployment.

---

### Fine-tune

Start a new run from a previously trained checkpoint (step counter resets, new output directory):

```bash
uv run anvil-trainer \
  --dataset.root=data/datasets/my-dataset \
  --policy.path=model_zoo/my-task/checkpoints/last/pretrained_model
```

`--policy.type` is not needed — it is read from the checkpoint's `config.json` automatically.

> **`--policy.path` vs `--resume`:** `--policy.path` starts fresh from a checkpoint's weights (new output dir, step counter at 0). `--resume` continues a stopped run in-place (same output dir, step counter carries over).

---

### Resume

```bash
# Resume from the latest checkpoint
uv run anvil-trainer --resume=model_zoo/pick-and-place

# Resume from a specific step
uv run anvil-trainer --resume=model_zoo/pick-and-place/checkpoints/020000
```

Only pass `--resume` — all other settings are restored from the checkpoint's `train_config.json`. Action type settings are inherited from `anvil_config.json` automatically.

---

[← Back to README](../README.md)
