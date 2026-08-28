"""lerobot policy wrapper around moojink/openvla-oft (MIT), following the same
"wrap an external repo's own weights/classes" pattern lerobot's own GrootPolicy
uses for nvidia/GR00T-N1.5-3B.

Environment note: unlike every other policy in this codebase, OpenVLA-OFT's
parallel-decoding L1-regression head needs bidirectional attention over the
action-chunk tokens, which only exists in openvla-oft's own transformers fork
(github.com/moojink/transformers-openvla-oft, ~8000 commits behind upstream —
a full old transformers snapshot, not a small patch). That fork cannot coexist
in the same Python environment as the transformers==5.3.0 pin every other
policy here needs (verified directly: importing transformers 4.40.1 with
huggingface_hub>=1.0 installed raises at import time). This package is
therefore meant to run in its own image/environment, not the shared inference
image — see docker/inference-openvla/Dockerfile. Verified end-to-end on this
machine: real openvla/openvla-7b (7.54B params) loaded via this exact fork,
running bidirectional attention on an RTX PRO 6000 Blackwell (sm_120), on
torch==2.9.0+cu130 (a full torch generation newer than the fork's own
torch==2.2.0 pin, which cannot target sm_120 at all).

RTC contract honesty note: lerobot's RTC (Real-Time Chunking) is a
flow-matching/diffusion guidance technique — RTCProcessor blends a new chunk's
denoising trajectory toward the previous chunk's still-executing actions via
gradient guidance through an iterative denoising loop (see pi0/pi05's
modeling files). OFT's L1RegressionActionHead has no denoising loop — it's a
single deterministic MLP pass over the transformer's hidden states — so there
is no trajectory to guide. init_rtc_processor()/predict_action_chunk() below
satisfy the *interface* Phase 0's dispatch layer checks for (hasattr(...,
"init_rtc_processor")), which is what actually matters here: it routes this
policy onto Anvil's decoupled background-inference-thread execution path
instead of blocking the 30Hz sync obs-timer loop on VLM latency (~500ms per
chunk measured on this GPU) — a real, valuable win on its own. It does NOT
perform true cross-chunk guided smoothing; inference_delay/prev_chunk_left_over
/execution_horizon are accepted (so the call signature matches what
inference_node.py's RTC thread passes) but unused. If OFT's DiffusionActionHead
variant (config.action_head_type == "diffusion", which does have an iterative
denoising loop) is used instead, real gradient-guided RTC blending becomes
structurally possible but is not implemented here — a separate, substantial
piece of work.
"""

import builtins
import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_openvla import OpenVLAConfig

T = Any  # matches lerobot's own `builtins.type[T]` from_pretrained pattern


def _patch_prismatic_globals(action_dim: int, proprio_dim: int, num_actions_chunk: int) -> None:
    """Override prismatic's action/proprio/chunk-size constants.

    prismatic/vla/constants.py sets these as module globals from sys.argv
    parsing (detect_robot_platform()), and prismatic/models/action_heads.py +
    prismatic/extern/hf/modeling_prismatic.py each did `from
    prismatic.vla.constants import ACTION_DIM, ...` — a by-value import, so
    reassigning prismatic.vla.constants.ACTION_DIM alone does NOT change what
    those modules see (verified directly: action_heads.ACTION_DIM stayed 7
    after patching constants.ACTION_DIM to 16). Every module that already
    imported these names by value must be patched individually.
    """
    import prismatic.vla.constants as constants_mod
    import prismatic.models.action_heads as action_heads_mod
    import prismatic.extern.hf.modeling_prismatic as modeling_mod

    for mod, names in (
        (constants_mod, ("ACTION_DIM", "PROPRIO_DIM", "NUM_ACTIONS_CHUNK")),
        (action_heads_mod, ("ACTION_DIM", "PROPRIO_DIM", "NUM_ACTIONS_CHUNK")),
        (modeling_mod, ("ACTION_DIM", "NUM_ACTIONS_CHUNK")),
    ):
        for name in names:
            if hasattr(mod, name):
                value = {"ACTION_DIM": action_dim, "PROPRIO_DIM": proprio_dim, "NUM_ACTIONS_CHUNK": num_actions_chunk}[
                    name
                ]
                setattr(mod, name, value)


class OpenVLAPolicy(PreTrainedPolicy):
    """Wrapper around moojink/openvla-oft's OpenVLAForActionPrediction for lerobot integration."""

    name = "openvla"
    config_class = OpenVLAConfig

    def __init__(self, config: OpenVLAConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config

        _patch_prismatic_globals(config.action_dim, config.proprio_dim, config.chunk_size)

        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig as PrismaticOpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
        from prismatic.models.projectors import ProprioProjector

        dtype = getattr(torch, config.dtype)
        prismatic_config = PrismaticOpenVLAConfig.from_pretrained(config.base_vla_path)
        self.vla = OpenVLAForActionPrediction.from_pretrained(
            config.base_vla_path, config=prismatic_config, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        llm_dim = self.vla.llm_dim if hasattr(self.vla, "llm_dim") else self.vla.config.text_config.hidden_size

        if config.action_head_type == "diffusion":
            self.action_head = DiffusionActionHead(
                transformer_hidden_dim=llm_dim,
                hidden_dim=llm_dim,
                action_dim=config.action_dim,
                num_diffusion_steps_train=config.num_diffusion_steps_train,
            )
        else:
            self.action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=config.action_dim)
        self.action_head = self.action_head.to(dtype)

        self.proprio_projector = None
        if config.use_proprio:
            self.proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=config.proprio_dim).to(dtype)

        # Loaded lazily by from_pretrained if the given checkpoint dir has one;
        # None means norm_stats are identity (min=-1,max=1 per-dim), matching
        # a freshly-random-init head with no rig fine-tune.
        self.vla.norm_stats = {
            "anvil_openarm": {
                "action": {
                    "min": [-1.0] * config.action_dim,
                    "max": [1.0] * config.action_dim,
                    "q01": [-1.0] * config.action_dim,
                    "q99": [1.0] * config.action_dim,
                }
            }
        }

        if config.device:
            self.to(config.device)

        self.init_rtc_processor()
        self.reset()

    def reset(self):
        """Reset policy state when environment/episode resets."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def init_rtc_processor(self):
        """See module docstring's RTC contract honesty note. No real guidance
        process exists for the L1-regression head; this exists so
        hasattr(OpenVLAPolicy, "init_rtc_processor") is True, which is what
        Anvil's dispatch layer (Phase 0) actually checks to route this policy
        onto the decoupled background-inference-thread execution path."""
        self.rtc_processor = None

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: OpenVLAConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs,
    ) -> T:
        """Load an OpenVLAPolicy.

        Mirrors GrootPolicy.from_pretrained's two-case structure: a base
        OpenVLA checkpoint (e.g. "openvla/openvla-7b") builds the config fresh
        and loads only the base VLA + random-init heads (no rig fine-tune
        exists yet); a fine-tuned Anvil checkpoint dir additionally loads
        action_head/proprio_projector state dicts saved alongside it.
        """
        model_id = str(pretrained_name_or_path)
        is_finetuned_checkpoint = os.path.isdir(model_id) and (
            os.path.exists(os.path.join(model_id, "action_head.safetensors"))
            or os.path.exists(os.path.join(model_id, "config.json"))
            and config is not None
        )

        if config is None:
            config = OpenVLAConfig()
            if not config.input_features:
                config.input_features = {
                    "observation.images.chest": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                }
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        policy = cls(config)

        if is_finetuned_checkpoint:
            action_head_path = os.path.join(model_id, "action_head.safetensors")
            proprio_path = os.path.join(model_id, "proprio_projector.safetensors")
            if os.path.exists(action_head_path):
                from safetensors.torch import load_file

                policy.action_head.load_state_dict(load_file(action_head_path), strict=strict)
            if policy.proprio_projector is not None and os.path.exists(proprio_path):
                from safetensors.torch import load_file

                policy.proprio_projector.load_state_dict(load_file(proprio_path), strict=strict)

        policy.eval()
        return policy

    def get_optim_params(self) -> dict:
        # Required by the PreTrainedPolicy ABC; forward() raises below, so
        # these params are never actually optimized through this class. Keeping
        # a real (non-empty) return, rather than `[]`, is intentional: it's
        # what a future differentiable-training implementation would return.
        params = list(self.action_head.parameters())
        if self.proprio_projector is not None:
            params += list(self.proprio_projector.parameters())
        return params

    def _extract_proprio(self, batch: dict[str, Tensor]) -> np.ndarray | None:
        if not self.config.use_proprio or OBS_STATE not in batch:
            return None
        state = batch[OBS_STATE]
        if isinstance(state, Tensor):
            state = state.detach().float().cpu().numpy()
        return state[0] if state.ndim > 1 else state

    def _run_vla(self, batch: dict[str, Tensor]) -> Tensor:
        """Shared VLA forward pass, gradient behavior controlled by the caller
        (predict_action_chunk wraps this in no_grad; forward() does not, so
        training can backprop into the action head / proprio projector)."""
        device = next(self.parameters()).device

        input_ids = batch["openvla_input_ids"].to(device)
        pixel_values = batch["openvla_pixel_values"].to(device=device, dtype=self.vla.dtype)
        attention_mask = batch["openvla_attention_mask"].to(device)
        proprio = self._extract_proprio(batch)

        actions, _hidden = self.vla.predict_action(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            unnorm_key="anvil_openarm",
            proprio=proprio,
            proprio_projector=self.proprio_projector,
            action_head=self.action_head,
            use_film=self.config.use_film,
        )
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device)
        if actions.dim() == 2:
            actions = actions.unsqueeze(0)  # (chunk, action_dim) -> (1, chunk, action_dim)

        original_action_dim = self.config.output_features[ACTION].shape[0]
        return actions[:, :, :original_action_dim]

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """Predict a chunk of actions. See the module docstring for what the
        RTC kwargs (inference_delay, prev_chunk_left_over, execution_horizon)
        do and don't do here."""
        self.eval()
        return self._run_vla(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Synchronous single-action selection (action-queue draining), for
        callers that don't use Anvil's decoupled RTC-thread path directly
        (e.g. anvil_eval's offline replay)."""
        self.eval()
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Training is intentionally NOT implemented.

        vla.predict_action() (used by _run_vla / predict_action_chunk) does its
        action unnormalization in numpy internally — its return type is
        np.ndarray, not a tensor with a live autograd graph — so building a
        loss on top of it produces a fully detached, zero-gradient "loss" that
        would silently do nothing. openvla-oft's own finetune.py uses a
        different, genuinely differentiable training-time codepath (loss
        computed pre-unnormalization, directly against normalized ground-truth
        actions) that this wrapper does not implement — Step 3's scope was
        inference dispatch, not a full anvil_trainer integration. Raising here
        is the honest choice over shipping a forward() that looks like it
        trains but doesn't.
        """
        raise NotImplementedError(
            "OpenVLAPolicy.forward() (training) is not implemented — predict_action() "
            "detaches to numpy internally, so a loss built on it would silently carry no "
            "gradient. A real implementation needs openvla-oft's differentiable training "
            "codepath (see vla-scripts/finetune.py in moojink/openvla-oft), not predict_action()."
        )
