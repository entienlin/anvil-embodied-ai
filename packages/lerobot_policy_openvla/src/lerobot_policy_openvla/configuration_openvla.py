"""Config for the OpenVLA-OFT policy plugin.

Wraps moojink/openvla-oft (https://github.com/moojink/openvla-oft, MIT), which
provides its own bimanual ALOHA training/deploy recipe on top of the OpenVLA-7B
base model. See ALOHA.md in that repo for the reference recipe this config
mirrors: `finetune.py --use_l1_regression True --use_film True --use_proprio
True --num_images_in_input 3`.

ALOHA's native ACTION_DIM/PROPRIO_DIM there is 14 (2x6 joints + 2 grippers);
the OpenArm rig is 16 (2x7 joints + 2 grippers). action_dim/proprio_dim below
default to the rig's shape rather than ALOHA's — both are just constructor
args to L1RegressionActionHead/ProprioProjector in the upstream repo, not a
hardcoded architecture assumption, so this is a legitimate resize.
"""

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE

DEFAULT_IMAGE_SIZE = 224


@PreTrainedConfig.register_subclass("openvla")
@dataclass
class OpenVLAConfig(PreTrainedConfig):
    # Base VLA checkpoint (HF repo id or local path). openvla/openvla-7b is the
    # pretrained backbone every openvla-oft fine-tune starts from; a rig-specific
    # fine-tune would point this at that fine-tune's own repo/path instead, in
    # which case action_head/proprio_projector weights below load from the same
    # checkpoint dir rather than being randomly initialised.
    base_vla_path: str = "openvla/openvla-7b"

    # --- Action / proprio shape ---------------------------------------------
    # Padded to these dims; shorter real features are zero-padded, matching the
    # max_state_dim/max_action_dim pattern pi0/pi05/groot already use in this
    # codebase. Defaults sized for the OpenArm rig (2x7 joints + 2 grippers).
    action_dim: int = 16
    proprio_dim: int = 16

    # NUM_ACTIONS_CHUNK in upstream constants.py. ALOHA default is 25 (25Hz
    # control loop in the paper); the rig runs the sync/RTC action queue at
    # 30Hz, so 30 keeps ~1s chunks. n_action_steps == chunk_size: OFT predicts
    # the whole chunk in one forward pass (no autoregressive queue draining
    # logic needed the way ACT's temporal ensembling has).
    chunk_size: int = 30
    n_action_steps: int = 30

    # --- Action head ----------------------------------------------------------
    # "l1_regression" (default; parallel single-pass MLP head, see
    # prismatic/models/action_heads.py:L1RegressionActionHead) or "diffusion"
    # (iterative denoising, prismatic/models/action_heads.py:DiffusionActionHead).
    # l1_regression is what ALOHA.md's reference command uses and is the
    # simpler/faster of the two — see the RTC caveat on predict_action_chunk in
    # modeling_openvla.py for why the action head choice matters for RTC.
    action_head_type: str = "l1_regression"
    num_diffusion_steps_train: int = 50  # only used if action_head_type == "diffusion"

    # --- Conditioning -----------------------------------------------------
    # FiLM language conditioning. ALOHA.md's reference command uses --use_film
    # True, but real FiLM requires swapping in a FiLMedPrismaticVisionBackbone
    # and loading ITS OWN saved weights from a fine-tuned checkpoint (see
    # openvla-oft's _apply_film_to_vla / find_checkpoint_file(..., "vision_backbone")
    # — no base/pretrained FiLMed vision backbone exists for openvla/openvla-7b).
    # Default False: with no rig fine-tune yet, True crashes (verified —
    # _process_vision_features unconditionally passes a 2nd positional arg to
    # the plain, un-FiLMed vision_backbone.forward(), which only takes one).
    # This wrapper does not implement the swap/LoRA wiring _apply_film_to_vla
    # does; flip to True only once loading a fine-tuned checkpoint that has it.
    use_film: bool = False
    use_proprio: bool = True
    num_images_in_input: int = 3  # third-person + 2 wrist cams in the reference recipe; rig has 4

    # Vision preprocessing (openvla-oft's resize_image_for_policy/center_crop_image)
    image_resolution: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    center_crop: bool = True

    # Prompt template the base VLA was trained on (see openvla-oft README/finetune.py)
    prompt_template: str = "In: What action should the robot take to {task}?\nOut:"

    # dtype the base VLA + heads run in. bfloat16 matches openvla-oft's own
    # deploy.py; float32 compute happens in the action head's MLPResNet
    # regardless (small enough to not matter).
    dtype: str = "bfloat16"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,  # ALOHA_CONSTANTS uses BOUNDS (min/max), not quantiles
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Optimizer/scheduler settings mirror openvla-oft's finetune.py defaults
    # (LoRA fine-tuning, lr warmup then constant — no decay schedule is used
    # there, so scheduler_decay_steps is set very large as an effective no-op).
    optimizer_lr: float = 5e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 0
    scheduler_decay_steps: int = 1_000_000_000
    scheduler_decay_lr: float = 5e-4

    device: str | None = None
    compile_model: bool = False

    def __post_init__(self):
        super().__post_init__()

        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )
        if self.action_head_type not in ("l1_regression", "diffusion"):
            raise ValueError(f"Invalid action_head_type: {self.action_head_type}")

    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(self.proprio_dim,))
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
