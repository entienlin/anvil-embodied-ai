"""Model loading utilities for LeRobot models.

Simplified model loader that uses LeRobot's built-in PolicyProcessorPipeline
for pre/post processing instead of custom implementations.
"""

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_deterministic_mode(seed: int = 42):
    """
    Set PyTorch to deterministic mode for reproducible inference.

    This should be called before loading the model for consistent results.

    Args:
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _register_third_party_policies() -> None:
    """Import installed ``lerobot_policy_*`` plugins so they self-register.

    lerobot auto-discovers out-of-tree policies, but only from its own CLI
    entrypoints (``lerobot_train.main()``, ``lerobot_record``, ...). Nothing in
    ``lerobot.policies.factory`` or ``PreTrainedConfig`` calls it, so a plugin
    would be invisible to ``get_policy_class`` on this path. Idempotent — the
    plugin modules are cached after the first import.

    Best-effort: a missing helper (older lerobot) or a broken plugin must not
    stop a first-party policy from loading.
    """
    try:
        from lerobot.utils.import_utils import register_third_party_plugins

        register_third_party_plugins()
    except Exception:  # noqa: BLE001 - plugin discovery is never load-critical
        pass


def patch_groot_for_transformers5() -> None:
    """Make lerobot 0.5.1's GR00T loadable under its own pinned transformers 5.3.0.

    GR00T vendors NVIDIA's GR00T-N1.5 model code, which was written against
    transformers 4.x. Two things break under 5.x, both during
    ``GR00TN15.from_pretrained``:

    1. ``FlowmatchingActionHead.__init__`` constructs ``Beta(alpha, beta)``.
       transformers 5.x initialises models on the meta device, and Beta's
       argument validation calls ``.item()``, which meta tensors reject
       ("Tensor.item() cannot be called on meta tensors"). Distribution
       argument validation is a debug aid with no effect on numerics, so it is
       disabled globally.
    2. ``GR00TN15.__init__`` never calls ``self.post_init()``. transformers 5.x
       populates ``all_tied_weights_keys`` there, and ``_finalize_model_loading``
       then fails with AttributeError. Calling post_init() restores the
       contract every PreTrainedModel subclass is expected to satisfy.
    3. ``processor_groot.collate()`` calls the (remote-code) Eagle processor as
       ``eagle_processor(text=..., images=..., images_kwargs={...},
       return_tensors="pt", ...)``. ``ProcessorMixin._merge_kwargs`` in this
       transformers version only auto-fills a modality's kwargs from a flat
       top-level kwarg when *no* explicit dict for that modality was passed;
       once ``images_kwargs={...}`` is given explicitly (as it is here), the
       flat ``return_tensors="pt"`` is silently dropped for images specifically
       (text still gets it, since no ``text_kwargs`` dict is passed). So
       ``image_processing_eagle2_5_vl_fast.py`` takes its "no return_tensors"
       branch and returns ``pixel_values`` as a Python list instead of a
       stacked tensor, crashing the placeholder-substitution regex a few lines
       later with ``AttributeError: 'list' object has no attribute 'shape'``.
       Re-adding ``return_tensors`` directly into ``images_kwargs`` while it is
       *also* still flat raises ``ValueError: ... passed two times`` — the
       library's own duplicate-specification guard. The fix is to route it
       through ``common_kwargs`` instead, which ``_merge_kwargs`` broadcasts
       into every modality dict (including already-explicit ones) *before*
       that guard runs — the mechanism this transformers version actually
       intends for this case. Applied generically, on any ``__call__`` that
       mixes a flat ``return_tensors`` with an explicit modality dict, not
       just this one call site.
    4. ``FlowmatchingActionHead.__init__`` (the training-time action head)
       builds ``self.beta_dist = Beta(alpha, beta)`` at construction time, same
       as #1 — but disabling validate_args only stops the constructor from
       *crashing*; it doesn't fix the distribution itself. ``Beta``/``Dirichlet``
       aren't ``nn.Module``s, so their internal tensors are plain object
       attributes, never swept up by ``model.to(device)`` or state_dict loading.
       Built under transformers 5.x's meta-device init context, ``beta_dist``'s
       tensors stay on ``meta`` *permanently* — inference never notices (GR00T's
       predict_action path samples pure noise via ``torch.randn``, never uses
       ``beta_dist``), but training's ``sample_time()`` calls
       ``self.beta_dist.sample(...)`` for real, which meta tensors structurally
       cannot do: ``NotImplementedError: aten::_sample_dirichlet: attempted to
       run this operator with Meta tensors``. Fixed by lazily rebuilding
       ``beta_dist`` the first time ``sample_time`` actually runs — by then the
       meta-device init context has long exited, so the rebuild lands on the
       ambient real device (confirmed: a Beta built outside that context is on
       CPU) and samples correctly; the result is `.to(device, dtype=dtype)`d
       right after anyway (existing code), so a CPU-built distribution is fine.

    Idempotent, and a no-op if lerobot has no GR00T or has fixed this upstream.
    Remove once lerobot ships a GR00T that is transformers-5 clean.
    """
    try:
        import torch.distributions as _dist
        from lerobot.policies.groot import groot_n1, processor_groot
        from lerobot.policies.groot.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )
    except ImportError:
        return

    if getattr(groot_n1.GR00TN15, "_anvil_post_init_patched", False):
        return

    _dist.Distribution.set_default_validate_args(False)

    _orig_init = groot_n1.GR00TN15.__init__

    def _init_with_post_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.post_init()

    groot_n1.GR00TN15.__init__ = _init_with_post_init
    groot_n1.GR00TN15._anvil_post_init_patched = True

    _orig_build_eagle_processor = processor_groot._build_eagle_processor

    def _build_eagle_processor_patched(*args, **kwargs):
        proc = _orig_build_eagle_processor(*args, **kwargs)
        proc_cls = type(proc)

        # __call__ is a dunder: Python resolves it on the type for the `proc(...)`
        # call syntax, so an instance-attribute override would be silently
        # ignored. Patch the class instead (idempotent — same guard as above).
        if not getattr(proc_cls, "_anvil_return_tensors_patched", False):
            _orig_call = proc_cls.__call__

            def _call_with_return_tensors(self, *c_args, **c_kwargs):
                has_explicit_modality_dict = any(
                    k.endswith("_kwargs") and isinstance(v, dict) for k, v in c_kwargs.items()
                )
                if "return_tensors" in c_kwargs and has_explicit_modality_dict:
                    return_tensors = c_kwargs.pop("return_tensors")
                    common = dict(c_kwargs.get("common_kwargs") or {})
                    common.setdefault("return_tensors", return_tensors)
                    c_kwargs["common_kwargs"] = common
                return _orig_call(self, *c_args, **c_kwargs)

            proc_cls.__call__ = _call_with_return_tensors
            proc_cls._anvil_return_tensors_patched = True
        return proc

    processor_groot._build_eagle_processor = _build_eagle_processor_patched

    _orig_sample_time = FlowmatchingActionHead.sample_time

    def _sample_time_with_meta_rebuild(self, batch_size, device, dtype):
        if self.beta_dist.concentration1.is_meta:
            from torch.distributions import Beta

            self.beta_dist = Beta(self.config.noise_beta_alpha, self.config.noise_beta_beta)
        return _orig_sample_time(self, batch_size, device, dtype)

    FlowmatchingActionHead.sample_time = _sample_time_with_meta_rebuild


def patch_groot_for_rtc_dispatch() -> None:
    """Give GrootPolicy a "sham" RTC contract so it runs on Anvil's decoupled
    background-inference-thread path instead of the synchronous, blocking one.

    lerobot's GrootPolicy has no init_rtc_processor, so Phase 0's dispatch
    (hasattr(policy_cls, "init_rtc_processor")) routes it onto the same
    synchronous path as ACT/Diffusion — inference_node.py's obs-update timer
    calls select_action() and blocks until it returns before publishing
    anything or reading a new observation.

    That's a real problem for GR00T specifically: measured directly from a
    real robot run's monitor_output/inference_data.csv, the control loop ran
    at a median ~7.4Hz (mean ~4.4Hz) against a 30Hz target, with stalls up to
    3.2s. Root cause: predict_action_chunk's returned action-chunk length is
    governed by the vendor NVIDIA checkpoint's own action_horizon (16 in the
    cached nvidia/GR00T-N1.5-3B config) — NOT by GrootConfig.n_action_steps
    (default 50, which the wrapper's deque(maxlen=n_action_steps) never
    actually fills past 16 real items). 16 steps is ~0.53s of execution
    runway at 30Hz, well under GR00T's real per-call latency (one Qwen3-1.7B+
    SigLIP2-400M backbone pass, the dominant cost, followed by 4 cheap Euler
    denoising steps — a single atomic, non-interruptible unit with no
    yield/checkpoint inside the loop). So the action queue runs dry and the
    control loop blocks waiting on inference, repeatedly, for the whole run.
    Coarse reaching tolerates that; precise contact-critical behavior (e.g.
    closing a gripper around a target) does not.

    pi0/pi05/smolvla don't have this problem: they implement
    init_rtc_processor and run on inference_node.py's background thread,
    which keeps publishing from the existing action queue while a new chunk
    computes in parallel, so slow inference never blocks control.

    This patch is the same "sham RTC contract" pattern already used for
    OpenVLA-OFT (see lerobot_policy_openvla/modeling_openvla.py, the concrete
    reference this mirrors) — enough to satisfy the hasattr() dispatch check
    and stop blocking the control loop, WITHOUT claiming real cross-chunk
    gradient-guided smoothing. Unlike OpenVLA's L1-regression head (no
    denoising loop, so real RTC guidance is structurally impossible there),
    GR00T's FlowmatchingActionHead does have a real iterative (4-step Euler)
    denoising loop, so genuine guidance may be implementable as a future
    enhancement — but that means patching the loop itself to accept and blend
    toward a guidance signal, a separate and substantially bigger piece of
    work. This patch is deliberately scoped to just "stop blocking the
    control loop," matching OpenVLA's honest scoping.

    IMPORTANT — must be paired with RTCConfig.enabled=False for GR00T (see
    ModelLoader._apply_rtc_config): getting GR00T onto this thread is not
    sufficient by itself. ActionQueue.merge()'s default (enabled=True)
    replace-mode discards `real_delay` control-loop steps from the front of
    every new chunk, assuming real_delay < chunk_length (true for
    pi0/pi05/smolvla). GR00T's real chunk (16 actions, ~0.53s) is far
    shorter than its own inference latency (88-155 steps, ~3-4s), so that
    assumption inverts and the discard empties every chunk unconditionally —
    confirmed on real hardware as a regression from this patch alone: 6
    successful inferences, 0 actions ever published, arms never moved.
    RTCConfig.enabled=False (append-mode, no delay discard) is required
    alongside this patch for GR00T to actually produce publishable actions.

    Idempotent, and a no-op if lerobot has no GR00T. Remove once lerobot's
    own GrootPolicy implements a real RTC contract upstream.
    """
    try:
        from lerobot.policies.groot import modeling_groot
    except ImportError:
        return

    if getattr(modeling_groot.GrootPolicy, "_anvil_rtc_dispatch_patched", False):
        return

    def init_rtc_processor(self) -> None:
        """No real guidance process exists here — see
        patch_groot_for_rtc_dispatch's docstring. This exists purely so
        hasattr(GrootPolicy, "init_rtc_processor") is True, which is what
        Anvil's dispatch layer (Phase 0) actually checks to route this policy
        onto the decoupled background-inference-thread execution path."""
        self.rtc_processor = None

    modeling_groot.GrootPolicy.init_rtc_processor = init_rtc_processor

    _orig_predict_action_chunk = modeling_groot.GrootPolicy.predict_action_chunk

    def predict_action_chunk(self, batch, **kwargs):
        """Accepts and discards the RTC kwargs (inference_delay,
        prev_chunk_left_over, execution_horizon) inference_node.py's
        background thread always passes — GrootPolicy's real
        predict_action_chunk(self, batch) takes none of them, and would raise
        TypeError without this wrapper (caught by the thread's own
        try/except, silently leaving the action queue empty forever)."""
        return _orig_predict_action_chunk(self, batch)

    modeling_groot.GrootPolicy.predict_action_chunk = predict_action_chunk
    modeling_groot.GrootPolicy._anvil_rtc_dispatch_patched = True


def reset_model_state(model):
    """
    Reset model's internal state for reproducible inference.

    For ACT models, this clears the action queue so each inference
    starts fresh with a new action chunk prediction.

    Args:
        model: The policy model (ACTPolicy, DiffusionPolicy, etc.)
    """
    if hasattr(model, "reset"):
        model.reset()


class ModelLoader:
    """
    Load trained LeRobot models from checkpoints.

    Uses LeRobot's built-in PolicyProcessorPipeline for preprocessing
    and postprocessing, eliminating the need for custom implementations.

    Supports any policy registered with lerobot's PreTrainedConfig
    (resolved via lerobot.policies.factory.get_policy_class), including
    third-party plugins — e.g. ACT, Diffusion, SmolVLA, Pi0, Pi0.5, GR00T.
    Policies implementing the RTC contract (init_rtc_processor) get
    RTCConfig injected automatically; others are loaded as-is.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        model_type: str | None = None,
        logger=None,
        deterministic: bool = False,
        seed: int = 42,
        config_overrides: dict = None,
        rtc_config_yaml: dict = None,
    ):
        """
        Initialize model loader.

        Args:
            model_path: Path to model checkpoint directory
            device: Device for inference ("cuda" or "cpu")
            model_type: Any policy name registered with lerobot's
                        PreTrainedConfig.register_subclass (e.g. "act",
                        "diffusion", "smolvla", "pi0", "pi05", "groot", or a
                        third-party plugin). None = auto-detect from config.json.
            logger: Optional ROS2 logger
            deterministic: If True, enable deterministic mode
            seed: Random seed for deterministic mode
            config_overrides: Dict of config values to override at load time
                e.g. {"temporal_ensemble_coeff": 0.01, "n_action_steps": 1}
            rtc_config_yaml: Dict from the ``rtc:`` YAML section. When set and
                the loaded model implements the RTC contract, RTCConfig is
                injected after loading and ``model.init_rtc_processor()`` is
                called.
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model_type = model_type
        self.logger = logger
        self.deterministic = deterministic
        self.seed = seed
        self.config_overrides = config_overrides or {}
        self.rtc_config_yaml = rtc_config_yaml or {}
        self._model = None
        self._pre_processor = None
        self._post_processor = None
        self._orig_n_action_steps: int | None = None

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Auto-detect pretrained_model subdirectory.
        # Accepts paths at checkpoint step level (e.g. checkpoints/last or checkpoints/100000)
        # by checking for a pretrained_model subfolder when config.json is not in the given path.
        if not (self.model_path / "config.json").exists():
            pretrained_model_path = self.model_path / "pretrained_model"
            if pretrained_model_path.exists() and (pretrained_model_path / "config.json").exists():
                self.model_path = pretrained_model_path

        # Auto-detect model type from checkpoint if not provided
        if self.model_type is None:
            self.model_type = self._detect_model_type()

    def _log(self, level: str, msg: str):
        """Log message using ROS2 logger or print."""
        if self.logger:
            try:
                getattr(self.logger, level)(msg)
            except ValueError:
                # ROS2 logger can fail with "severity cannot be changed between calls"
                print(f"[{level.upper()}] {msg}")
        else:
            print(f"[{level.upper()}] {msg}")

    def _detect_model_type(self) -> str | None:
        """Read model type from config.json in the checkpoint directory."""
        config_path = self.model_path / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text()).get("type")
        return None

    @property
    def checkpoint_n_action_steps(self) -> int | None:
        """Original n_action_steps from checkpoint before any overrides were applied."""
        return self._orig_n_action_steps

    @property
    def chunk_size(self) -> int | None:
        """Get model chunk size from config."""
        if (
            self._model
            and hasattr(self._model, "config")
            and hasattr(self._model.config, "chunk_size")
        ):
            return self._model.config.chunk_size
        return None

    @property
    def n_action_steps(self) -> int | None:
        """Get model n_action_steps from config."""
        if (
            self._model
            and hasattr(self._model, "config")
            and hasattr(self._model.config, "n_action_steps")
        ):
            return self._model.config.n_action_steps
        return None

    def load(self):
        """
        Load model from checkpoint.

        Returns:
            Loaded model in eval mode
        """
        self._log("info", f"Loading {self.model_type} model from: {self.model_path}")

        if self.deterministic:
            set_deterministic_mode(self.seed)
            self._log("info", f"Deterministic mode enabled with seed={self.seed}")

        try:
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.policies.factory import get_policy_class

            _register_third_party_policies()
            if self.model_type == "groot":
                patch_groot_for_transformers5()

            try:
                policy_cls = get_policy_class(self.model_type)
            except NotImplementedError as e:
                raise ValueError(f"Unsupported model type: {self.model_type}") from e

            policy_cfg = PreTrainedConfig.from_pretrained(str(self.model_path))
            if hasattr(policy_cfg, "compile_model"):
                policy_cfg.compile_model = False
            model = policy_cls.from_pretrained(str(self.model_path), config=policy_cfg)

            model = model.to(self.device)
            model.eval()
            self._model = model
            self._log("info", f"Model loaded successfully on {self.device}")

            # Snapshot checkpoint n_action_steps before any overrides
            if hasattr(model, "config"):
                self._orig_n_action_steps = getattr(model.config, "n_action_steps", None)

            # Apply config overrides after loading
            self._apply_config_overrides(model)

            # Inject RTCConfig for RTC-capable models (pi0 / pi05 / smolvla / ...)
            self._apply_rtc_config(model)

            return model

        except ImportError as e:
            self._log("error", f"lerobot package not installed: {e}")
            raise
        except Exception as e:
            self._log("error", f"Failed to load model: {e}")
            raise

    def _apply_config_overrides(self, model) -> None:
        """Apply config overrides after model is loaded.

        For temporal_ensemble_coeff, we also need to create the ensembler
        since it's only created in __init__ if the coeff was set at load time.
        """
        if not self.config_overrides or not hasattr(model, "config"):
            return

        config = model.config
        overrides_applied = []

        for key, value in self.config_overrides.items():
            if value is not None and hasattr(config, key):
                old_val = getattr(config, key)
                setattr(config, key, value)
                overrides_applied.append(f"{key}: {old_val} -> {value}")

        if overrides_applied:
            self._log("info", "Applied config overrides:")
            for override in overrides_applied:
                self._log("info", f"  - {override}")

        # Special handling: create temporal_ensembler if enabling at runtime
        if "temporal_ensemble_coeff" in self.config_overrides:
            coeff = self.config_overrides["temporal_ensemble_coeff"]
            if coeff is not None and not hasattr(model, "temporal_ensembler"):
                self._create_temporal_ensembler(model, coeff)

        # Special handling: propagate num_inference_steps to DiffusionModel internal cache
        # (DiffusionModel caches this at __init__ before overrides are applied)
        if "num_inference_steps" in self.config_overrides and hasattr(model, "diffusion"):
            model.diffusion.num_inference_steps = self.config_overrides["num_inference_steps"]

    def _apply_rtc_config(self, model) -> None:
        """Inject RTCConfig into RTC-capable models and call init_rtc_processor().

        Applies to any policy whose class implements the RTC contract
        (pi0 / pi05 / smolvla with real gradient-guided RTC, plus GR00T via
        the "sham" contract from patch_groot_for_rtc_dispatch() purely to get
        onto the decoupled background-thread path). Policies without it
        (e.g. ACT, Diffusion) are skipped silently. Must be called *after*
        _apply_config_overrides so that any n_action_steps override is
        already in place before RTC initialisation.

        `enabled` is forced False for GR00T specifically: RTCConfig.enabled=True
        routes ActionQueue.merge() through _replace_actions_queue(), which
        discards `real_delay` (control-loop steps elapsed during inference)
        from the front of every new chunk — correct for pi0/pi05/smolvla,
        whose inference is fast relative to a long chunk. GR00T inverts that:
        its real per-call chunk length is fixed at ~16 actions
        (action_horizon, baked into the vendor DiT checkpoint) but its real
        inference latency (~3-4s) is 88-155 steps at 30Hz — far longer than
        the chunk itself. So the delay clamp always reduces to
        len(chunk)=16, slicing every chunk down to empty and leaving the
        queue permanently at 0 (confirmed on real hardware: 6 successful
        inferences, 0 actions ever published, arms never moved). GR00T's
        RTCConfig.enabled=False instead routes through _append_actions_queue(),
        which has no delay-based discarding, so chunks actually reach the
        queue. This is safe to scope to GR00T alone: `enabled` is read only
        by ActionQueue.merge() and by each of pi0/pi0_fast/pi05/smolvla's own
        internal guidance-gating property — GR00T's predict_action_chunk
        never reads rtc_config at all, and each policy gets its own RTCConfig
        instance, so this cannot affect pi0/pi05/smolvla's real RTC guidance.
        """
        if not hasattr(model, "init_rtc_processor"):
            return
        if not self.rtc_config_yaml:
            return

        try:
            from lerobot.policies.rtc.configuration_rtc import RTCConfig
            from lerobot.configs.types import RTCAttentionSchedule

            schedule_str = self.rtc_config_yaml.get("prefix_attention_schedule", "EXP")
            schedule = RTCAttentionSchedule[schedule_str]

            rtc_enabled = self.model_type != "groot"
            model.config.rtc_config = RTCConfig(
                enabled=rtc_enabled,
                execution_horizon=self.rtc_config_yaml.get("execution_horizon", 10),
                max_guidance_weight=self.rtc_config_yaml.get("max_guidance_weight", 10.0),
                prefix_attention_schedule=schedule,
            )
            model.init_rtc_processor()
            mode = "replace (real RTC)" if rtc_enabled else "append (no guidance, GR00T sham RTC)"
            self._log(
                "info",
                f"RTC dispatch enabled for {self.model_type}, queue mode={mode} "
                f"(execution_horizon={model.config.rtc_config.execution_horizon}, "
                f"max_guidance_weight={model.config.rtc_config.max_guidance_weight}, "
                f"schedule={schedule_str})",
            )
        except Exception as e:
            self._log("error", f"Failed to initialise RTC: {e}")
            raise

    def _create_temporal_ensembler(self, model, coeff: float) -> None:
        """Create temporal ensembler for ACT model."""
        try:
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

            chunk_size = model.config.chunk_size
            model.temporal_ensembler = ACTTemporalEnsembler(coeff, chunk_size)
            self._log(
                "info", f"Created temporal ensembler (coeff={coeff}, chunk_size={chunk_size})"
            )
        except ImportError as e:
            self._log("error", f"Failed to import ACTTemporalEnsembler: {e}")
        except Exception as e:
            self._log("error", f"Failed to create temporal ensembler: {e}")

    def load_with_processors(self) -> tuple[Any, Any, Any]:
        """
        Load model with LeRobot's built-in processor pipelines.

        Returns:
            Tuple of (model, preprocessor_pipeline, postprocessor_pipeline)
        """
        model = self.load()

        pre_processor = None
        post_processor = None

        try:
            import importlib

            from lerobot.processor import PolicyProcessorPipeline

            # Policies register custom ProcessorStep implementations in their own
            # processor_<model>.py modules (e.g. pi05's "prepare_state_tokenizer"
            # step). These must be imported before calling from_pretrained so that
            # ProcessorStepRegistry recognises the step names. Without this import
            # the registry lookup fails and processor loading silently falls back
            # to None. Not every policy ships a processor_<model>.py module, so a
            # missing one is expected and silently skipped.
            try:
                importlib.import_module(f"lerobot.policies.{self.model_type}.processor_{self.model_type}")
            except ModuleNotFoundError:
                pass

            # Check if processor files exist
            pre_config = self.model_path / "policy_preprocessor.json"
            post_config = self.model_path / "policy_postprocessor.json"

            if pre_config.exists():
                pre_processor = PolicyProcessorPipeline.from_pretrained(
                    str(self.model_path), config_filename="policy_preprocessor.json"
                )
                # Move to device if the pipeline supports it
                if hasattr(pre_processor, "to") and callable(pre_processor.to):
                    try:
                        pre_processor.to(self.device)
                    except (TypeError, AttributeError):
                        pass  # Pipeline doesn't support device movement
                self._pre_processor = pre_processor
                self._log("info", "Loaded preprocessor pipeline from checkpoint")

            if post_config.exists():
                post_processor = PolicyProcessorPipeline.from_pretrained(
                    str(self.model_path), config_filename="policy_postprocessor.json"
                )
                # Move to device if the pipeline supports it
                if hasattr(post_processor, "to") and callable(post_processor.to):
                    try:
                        post_processor.to(self.device)
                    except (TypeError, AttributeError):
                        pass  # Pipeline doesn't support device movement
                self._post_processor = post_processor
                self._log("info", "Loaded postprocessor pipeline from checkpoint")

        except FileNotFoundError as e:
            self._log("warn", f"Processor pipelines not found: {e}")
        except Exception as e:
            self._log("error", f"Failed to load processor pipelines: {e}")

        # For pi05: if no preprocessor found in the checkpoint (fine-tuned models often
        # skip saving policy_preprocessor.json), rebuild it from the policy's own factory.
        # dataset_stats=None → normalization uses empty stats (passthrough), but tokenization
        # is correctly configured with the PaliGemma tokenizer.
        if self.model_type == "pi05" and pre_processor is None:
            try:
                from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

                pre_processor, fallback_post = make_pi05_pre_post_processors(
                    model.config, dataset_stats=None
                )
                if post_processor is None:
                    post_processor = fallback_post
                self._log("info", "Built pi05 processor from policy factory (no policy_preprocessor.json found)")
            except Exception as e:
                self._log("warn", f"Could not build pi05 processor from factory: {e}")

        if pre_processor is None and post_processor is None:
            self._log(
                "warn",
                "No processor pipelines found in checkpoint — observations and actions will NOT be "
                "normalized. If the model was trained with a non-IDENTITY normalization_mapping "
                "(e.g. MEAN_STD), inference results will be incorrect. "
                "Re-train or ensure policy_preprocessor.json / policy_postprocessor.json exist in "
                f"{self.model_path}",
            )

        return model, pre_processor, post_processor

    @staticmethod
    def validate_model_path(model_path: str) -> bool:
        """
        Validate model checkpoint path.

        Args:
            model_path: Path to check

        Returns:
            True if path appears to be valid model checkpoint
        """
        path = Path(model_path)
        if not path.exists():
            return False

        has_config = (path / "config.json").exists()
        has_model = (path / "pretrained_model").exists() or (path / "model.safetensors").exists()
        return has_config or has_model
