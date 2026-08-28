# lerobot_policy_openvla

OpenVLA-OFT policy plugin for [lerobot](https://github.com/huggingface/lerobot),
discovered via lerobot's third-party plugin mechanism
(`register_third_party_plugins()`, matched by the `lerobot_policy_` distribution
name prefix — **not** `lerobot-policy-openvla`; see the note in `pyproject.toml`).

Wraps [moojink/openvla-oft](https://github.com/moojink/openvla-oft) (MIT), an
OpenVLA fine-tuning recipe with a parallel-decoding L1-regression action head
and native bimanual ALOHA support. Registers the `openvla` policy type
(`--policy.type=openvla`).

## Environment

This package requires openvla-oft's own transformers fork
(`github.com/moojink/transformers-openvla-oft`) for bidirectional attention
over action-chunk tokens — a real dependency of the parallel-decoding action
head, not a version pin that can be relaxed. That fork cannot coexist with the
`transformers==5.3.0` every other policy in this repo needs (verified: hard
import-time version check failure). Runs in its own environment/image; see
`docker/inference-openvla/Dockerfile`.

Sized for the OpenArm rig: 16-dim action/proprio (2×7 joints + 2 grippers),
vs. openvla-oft's ALOHA default of 14 (2×6 + 2). No rig-specific fine-tune
exists yet — action_head/proprio_projector weights are random-initialized
until one is trained (see `vla-scripts/finetune.py` in the upstream repo).

## Status (as of 2026-08-28)

This is a verified, working **skeleton** — not yet a deployable policy. Read
this section before trusting it near a robot.

**Verified, on a real RTX PRO 6000 Blackwell:**
- Real `openvla/openvla-7b` base weights (7.54B params) load through this
  wrapper's exact classes (not a hand-rolled bypass).
- Real forward pass — bidirectional attention, real GPU kernels, correct
  `(1, chunk_size, 16)` action-chunk shape — through `OpenVLAPolicy.predict_action_chunk`.
- Full dispatch: `register_third_party_plugins()` discovers the package,
  `get_policy_class("openvla")` resolves it, `hasattr(cls, "init_rtc_processor")`
  is `True` so Anvil's Phase-0 dispatch layer routes it onto the decoupled
  background-inference thread.
- `docker build -f docker/inference-openvla/Dockerfile` succeeds from a clean
  `--no-cache` build and reproduces all of the above inside the container.

**NOT done — required before this can run on the real rig:**
1. **No rig fine-tune.** `action_head`/`proprio_projector` are random-init.
   Predicted actions are architecturally correct but numerically meaningless
   until trained on real OpenArm demonstration data.
2. **No real RTC guidance.** `init_rtc_processor()`/`predict_action_chunk()`
   satisfy the interface (so dispatch routes correctly) but do not perform
   lerobot's actual cross-chunk gradient-guided smoothing — OFT's default
   L1-regression head has no denoising loop to guide. See the RTC honesty
   note in `docs/training.md`.
3. **No training path.** `OpenVLAPolicy.forward()` raises `NotImplementedError`
   rather than silently computing a detached, zero-gradient loss.
   `vla.predict_action()` unnormalizes in numpy internally, so a loss built
   on it can't backprop — a real implementation needs openvla-oft's own
   differentiable training codepath (`vla-scripts/finetune.py`), not
   `predict_action()`.
4. **FiLM is off by default and unimplemented here.** Real FiLM needs a
   fine-tuned checkpoint's own saved vision-backbone weights
   (`_apply_film_to_vla` in openvla-oft) — there's nothing to load yet.

**Path to a real deployment, in order:**
1. Collect/convert a bimanual OpenArm dataset into the RLDS format
   `vla-scripts/finetune.py` expects (see `ALOHA.md` in `moojink/openvla-oft`
   for the reference recipe: preprocess → convert to RLDS → register in
   `configs.py`/`transforms.py`/`mixtures.py`).
2. Fine-tune with `ACTION_DIM=16`/`PROPRIO_DIM=16` patched (mirror
   `_patch_prismatic_globals()` in `modeling_openvla.py`), producing
   `action_head.safetensors` + `proprio_projector.safetensors`.
3. Point `OpenVLAPolicy.from_pretrained()` at that checkpoint dir — it
   already looks for those two filenames and loads them if present.
4. Re-run the verification in this README with the real weights: confirm
   `predict_action_chunk` output is no longer just architecturally-shaped
   noise before running it anywhere near the robot.
5. Only then consider implementing real RTC guidance (needs OFT's
   `DiffusionActionHead` variant, which has an actual denoising loop) or a
   differentiable training path, if either turns out to matter in practice.
