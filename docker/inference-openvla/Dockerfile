# Separate inference image for the OpenVLA-OFT policy (lerobot_policy_openvla).
#
# WHY THIS IS A SEPARATE IMAGE, NOT LEROBOT_EXTRAS=openvla ON THE MAIN ONE:
# OpenVLA-OFT's parallel-decoding L1-regression action head needs bidirectional
# attention over action-chunk tokens, which only exists in openvla-oft's own
# transformers fork (github.com/moojink/transformers-openvla-oft — an old,
# full transformers snapshot ~8000 commits behind upstream, not a small patch).
# That fork requires huggingface_hub<1.0 and hard-conflicts at IMPORT TIME
# (not just a version-mismatch warning) with the transformers==5.3.0 /
# huggingface_hub>=1.0 stack docker/inference/Dockerfile uses for
# ACT/Diffusion/Pi0/Pi0.5/SmolVLA/GR00T. Verified directly on this machine:
# real openvla/openvla-7b (7.54B params) loaded via this fork, running
# bidirectional attention on an RTX PRO 6000 Blackwell (sm_120), on
# torch==2.9.0+cu130 — a full torch generation newer than the fork's own
# torch==2.2.0 pin, which cannot target sm_120 (added in PyTorch 2.7+/CUDA
# 12.8) at all. So: modern, Blackwell-capable torch + old transformers fork,
# NOT the fork's own stated torch pin.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    ros-jazzy-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

# Blackwell-capable torch first — everything below installs against this,
# never against openvla-oft's own torch==2.2.0 pin.
RUN pip3 install --break-system-packages --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu130 \
    "torch==2.9.0" "torchvision==0.24.0"

# lerobot (dispatch layer Anvil's ROS2 node depends on) + the forked
# transformers, in that order: lerobot first (its own resolver pulls a modern
# huggingface_hub), then downgrade it and swap in the fork — installing in the
# other order lets lerobot's resolver upgrade huggingface_hub back past 1.0.
#
# lerobot's own resolver (with --ignore-installed, no --no-deps) also silently
# replaces the pinned Blackwell torch above with whatever its own torch
# constraint resolves to from the default PyPI index (observed: torch 2.10+cu128
# instead of the 2.9.0+cu130 actually verified against the forked transformers'
# bidirectional attention) — so re-pin torch back explicitly afterward, forcing
# it, matching the exact sequence that was verified end-to-end.
RUN pip3 install --break-system-packages --no-cache-dir --ignore-installed "lerobot==0.5.1" numpy packaging matplotlib "scipy>=1.14.0,<2.0.0" && \
    pip3 install --break-system-packages --no-cache-dir --force-reinstall --no-deps \
        --index-url https://download.pytorch.org/whl/cu130 \
        "torch==2.9.0" "torchvision==0.24.0" && \
    pip3 install --break-system-packages --no-cache-dir "huggingface_hub>=0.34,<1.0" && \
    pip3 install --break-system-packages --no-cache-dir --force-reinstall --no-deps \
        "git+https://github.com/moojink/transformers-openvla-oft.git"

# transformers 4.40.1's own deps, at versions compatible with a 2025-era
# huggingface_hub/tokenizers/safetensors stack (not its original 2024 pins —
# see lerobot_policy_openvla/pyproject.toml for why each is bumped).
RUN pip3 install --break-system-packages --no-cache-dir \
    "tokenizers==0.19.1" "safetensors" "regex" "requests" "pyyaml" "tqdm"

# Compatibility shims: lerobot 0.5.1's lerobot/policies/__init__.py eagerly
# imports EVERY built-in policy, including pi0/pi0_fast, whose pi_gemma.py
# needs transformers internals (masking_utils.create_causal_mask,
# modeling_layers.GradientCheckpointingLayer, models.paligemma.*) added well
# after transformers 4.40.1. Nothing in this image ever constructs a
# PI0/PI0Fast policy — these three shims only need to satisfy the IMPORT so
# `from lerobot.policies.factory import get_policy_class` (which
# ModelLoader.load() calls unconditionally, regardless of which policy is
# actually being loaded) doesn't crash the whole process. Calling any of the
# real PI0/PI0Fast code paths in this image will raise NotImplementedError —
# that's intentional; those policies belong in docker/inference/Dockerfile.
COPY docker/inference-openvla/transformers_shims/masking_utils.py /usr/local/lib/python3.12/dist-packages/transformers/masking_utils.py
COPY docker/inference-openvla/transformers_shims/modeling_layers.py /usr/local/lib/python3.12/dist-packages/transformers/modeling_layers.py
COPY docker/inference-openvla/transformers_shims/paligemma/__init__.py /usr/local/lib/python3.12/dist-packages/transformers/models/paligemma/__init__.py
COPY docker/inference-openvla/transformers_shims/paligemma/modeling_paligemma.py /usr/local/lib/python3.12/dist-packages/transformers/models/paligemma/modeling_paligemma.py

# openvla-oft itself (prismatic package) + its inference-time deps. Its own
# pyproject.toml unconditionally imports the full RLDS/dlimp/tensorflow
# training-data pipeline from prismatic/__init__.py's import chain — even for
# a pure model-class import — so all of it has to be present, not just the
# training-only pieces trimmed away. See lerobot_policy_openvla/pyproject.toml
# for the exact verified version set (RUN here mirrors it) and modeling_openvla.py
# for the _patch_prismatic_globals() workaround this cascades into (module-global
# ACTION_DIM/PROPRIO_DIM patched-by-value, not just prismatic.vla.constants).
# Order matters: tensorflow_graphics pulls an old protobuf as a side effect
# of its own resolver pass; tensorflow_datasets separately needs a newer
# protobuf runtime for its generated proto code (a genuine, not merely
# pip-resolver, conflict — tensorflow itself tolerates either). Bundling all
# of this into one pip invocation makes pip's single-pass resolver see the
# tensorflow<->protobuf pin clash and refuse outright; splitting into ordered
# calls, each free to move protobuf again, is what actually installs cleanly
# (verified directly — this exact sequence is what worked in isolated testing).
RUN pip3 install --break-system-packages --no-cache-dir --no-deps \
        "git+https://github.com/moojink/openvla-oft.git" \
        "git+https://github.com/moojink/dlimp_openvla" && \
    pip3 install --break-system-packages --no-cache-dir \
        "timm==0.9.10" "peft>=0.19,<0.20" "diffusers==0.30.3" \
        "wandb>=0.29" "einops" "accelerate" "draccus==0.8.0" \
        "json-numpy" "jsonlines" "matplotlib" "sentencepiece>=0.2.0" && \
    pip3 install --break-system-packages --no-cache-dir "tensorflow==2.18.1" && \
    pip3 install --break-system-packages --no-cache-dir "tensorflow_datasets>=4.9.2,<4.10" && \
    pip3 install --break-system-packages --no-cache-dir "tensorflow_graphics" && \
    pip3 install --break-system-packages --no-cache-dir "protobuf>=6.31.1"

# lerobot_policy_openvla itself
COPY packages/lerobot_policy_openvla /workspace/lerobot_policy_openvla
RUN pip3 install --break-system-packages --no-cache-dir --no-deps -e /workspace/lerobot_policy_openvla

# Copy ROS2 package (same node code as the main inference image — the
# dispatch layer is generic, per Phase 0; only the environment differs)
COPY ros2/src/lerobot_control /workspace/src/lerobot_control
COPY configs /workspace/configs

RUN . /opt/ros/jazzy/setup.sh && \
    cd /workspace && \
    colcon build --symlink-install

COPY docker/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["ros2", "launch", "lerobot_control", "inference.launch.py"]
