"""Pre/post-processing for the OpenVLA-OFT policy.

Turns lerobot's batch format (observation.images.<cam>, observation.state,
task) into what OpenVLAForActionPrediction.predict_action() needs (input_ids,
pixel_values, attention_mask, proprio), and unnormalizes actions back out
using lerobot's own generic Normalizer/UnnormalizerProcessorStep (mirroring
pi05's processor_pi05.py pattern), driven by config.normalization_mapping /
dataset_stats.

Unlike pi0/pi05/GR00T's processors, which call a single AutoProcessor built
from HF trust_remote_code, OpenVLA-OFT's own image preprocessing
(resize_image_for_policy / center_crop_image in openvla-oft's
experiments/robot/openvla_utils.py) is plain PIL/numpy, not part of the HF
processor's __call__. We replicate that logic here rather than depend on
openvla-oft's experiments/ package (a script-oriented module tree, not meant
to be imported as a library).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from PIL import Image

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
from lerobot.types import EnvTransition, TransitionKey
from lerobot.utils.constants import (
    OBS_IMAGES,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from .configuration_openvla import OpenVLAConfig


def _center_crop_and_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Matches openvla-oft's center_crop_image + resize_image_for_policy at inference time."""
    h, w = img.shape[:2]
    crop = min(h, w)
    top, left = (h - crop) // 2, (w - crop) // 2
    img = img[top : top + crop, left : left + crop]
    pil = Image.fromarray(img).resize((size, size), Image.BICUBIC)
    return np.asarray(pil)


@ProcessorStepRegistry.register(name="openvla_prepare_inputs_v1")
@dataclass
class OpenVLAPrepareInputsStep(ProcessorStep):
    """Builds the VLA's tokenized prompt + pixel_values from a raw batch.

    The actual tokenizer/HF processor is loaded lazily (first call) from
    base_vla_path, since it's needed only for text/image encoding, not model
    weights. Emits openvla_input_ids / openvla_pixel_values / openvla_attention_mask
    keys, consumed directly by OpenVLAPolicy.predict_action_chunk. Runs *after*
    NormalizerProcessorStep in the pipeline, so observation.state (if present)
    is already normalized by the time OpenVLAPolicy reads it as proprio.
    """

    base_vla_path: str = "openvla/openvla-7b"
    prompt_template: str = "In: What action should the robot take to {task}?\nOut:"
    image_size: int = 224
    center_crop: bool = True
    _processor: Any = field(default=None, init=False, repr=False)

    @property
    def processor(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.base_vla_path, trust_remote_code=True)
        return self._processor

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = dict(transition.get(TransitionKey.OBSERVATION) or {})
        complementary = dict(transition.get(TransitionKey.COMPLEMENTARY_DATA) or {})

        task = complementary.get("task")
        task_str = task[0] if isinstance(task, (list, tuple)) and task else (task or "")
        prompt = self.prompt_template.format(task=task_str)

        img_keys = sorted(k for k in obs if k.startswith(OBS_IMAGES))
        if not img_keys:
            raise ValueError(f"OpenVLA requires at least one observation.images.* key; got {list(obs)}")
        # OpenVLA-7B (and the OFT fine-tunes) take a single primary image per
        # forward pass; wrist/secondary cameras are handled by OFT's
        # num_images_in_input multi-image variant, which needs its own vision
        # backbone entry points we don't wire up here. Use the first camera
        # (by sorted key) until multi-image support is added.
        img_tensor = obs[img_keys[0]]
        img_np = img_tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
        if self.center_crop:
            img_np = _center_crop_and_resize(img_np, self.image_size)
        else:
            img_np = np.asarray(Image.fromarray(img_np).resize((self.image_size, self.image_size)))
        pil_img = Image.fromarray(img_np)

        encoded = self.processor(prompt, pil_img)

        obs["openvla_input_ids"] = encoded["input_ids"]
        obs["openvla_pixel_values"] = encoded["pixel_values"]
        obs["openvla_attention_mask"] = torch.ones_like(encoded["input_ids"])

        new_transition = dict(transition)
        new_transition[TransitionKey.OBSERVATION] = obs
        return new_transition

    def transform_features(self, features):
        return features


def make_openvla_pre_post_processors(
    config: OpenVLAConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Mirrors processor_pi05.make_pi05_pre_post_processors' structure.

    Pre: rename -> batch dim -> normalize state (MIN_MAX bounds, per
    config.normalization_mapping) -> tokenize prompt + prep image -> device.
    Post: unnormalize action -> device (cpu).
    """
    input_steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={OBS_STATE: config.input_features[OBS_STATE]} if OBS_STATE in config.input_features else {},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
        ),
        OpenVLAPrepareInputsStep(
            base_vla_path=config.base_vla_path,
            prompt_template=config.prompt_template,
            image_size=config.image_resolution[0],
            center_crop=config.center_crop,
        ),
        DeviceProcessorStep(device=config.device or "cpu"),
    ]
    output_steps: list[ProcessorStep] = [
        UnnormalizerProcessorStep(
            features=config.output_features, norm_map=config.normalization_mapping, stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=input_steps,
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=output_steps,
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
