# Use ROS2 Jazzy as base
FROM ros:jazzy-ros-base

ARG LEROBOT_VERSION=0.5.1
ARG LEROBOT_EXTRAS=""

# Set environment
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
# ros-jazzy-foxglove-bridge: lightweight websocket relay so camera/joint topics
# can be viewed live in Foxglove Studio from any machine — no X11/GUI needed on
# this box. Small package, always installed rather than gated behind a build arg.
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    curl \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create workspace
WORKDIR /workspace

# Base lerobot install — cached as long as version doesn't change
# transformers + scipy are required for VLA models (pi0, pi0.5, smolvla)
RUN pip3 install --break-system-packages --no-cache-dir --ignore-installed \
    numpy packaging "lerobot==${LEROBOT_VERSION}" matplotlib \
    "transformers==5.3.0" "scipy>=1.14.0,<2.0.0" && \
    pip3 install --break-system-packages --no-cache-dir "setuptools<80"

# Extras — separate layer, only rebuilt when LEROBOT_EXTRAS changes.
# "groot" is filtered out here and handled by the dedicated layer below: its
# lerobot[groot] bracket pulls flash-attn, which is sdist-only on PyPI and
# cannot build in this image (no CUDA toolkit). See that layer for details.
RUN LEROBOT_PIP_EXTRAS="$(echo "${LEROBOT_EXTRAS}" | tr ',' '\n' | grep -vx 'groot' | paste -sd, -)"; \
    if [ -n "${LEROBOT_PIP_EXTRAS}" ]; then \
        pip3 install --break-system-packages --no-cache-dir \
            "lerobot[${LEROBOT_PIP_EXTRAS}]==${LEROBOT_VERSION}"; \
    fi

# GR00T layer — only built when LEROBOT_EXTRAS contains "groot".
#
# GR00T N1.5 wraps an Eagle-2.5-VL backbone that HARD-REQUIRES flash-attn:
# eagle2_hg_model/modeling_eagle2_5_vl.py forces the SigLIP vision tower to
# flash_attention_2 and asserts it for the Qwen2 LM. modeling_groot.py's
# _handle_flash_attention_compatibility() only *prints* "will use fallback
# attention" — it never actually changes _attn_implementation, so there is no
# real fallback and flash-attn is mandatory rather than optional.
#
# flash-attn ships sdist-only on PyPI, and building it needs a CUDA toolkit
# (absent from ros:jazzy-ros-base) with nvcc >= 12.8 to emit sm_120 kernels for
# Blackwell GPUs — an older nvcc builds "successfully" but silently omits
# sm_120, failing at runtime with "no kernel image is available". So install the
# official prebuilt wheel instead. That wheel is built against torch 2.9, hence
# the torch pin here; lerobot 0.5.1 allows torch>=2.7,<2.11 so this is in range.
#
# timm and ninja from the lerobot[groot] bracket are deliberately omitted: timm
# is declared but never imported anywhere under lerobot/policies/groot/, and
# ninja is only needed for a source build we are not doing.
#
# The base layer above already pulled torch from PyPI (a cu12 build, with
# nvidia-*-cu12 dependency wheels). Those are purged first so the image ends up
# with one CUDA runtime rather than cu12 and cu13 side by side (~4GB of dead
# weight, since the cu13 wheels use different package names and would not
# otherwise displace them).
ARG GROOT_TORCH_VERSION=2.9.0
ARG GROOT_TORCHVISION_VERSION=0.24.0
ARG GROOT_TORCHCODEC_VERSION=0.9.0
ARG GROOT_FLASH_ATTN_WHEEL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu13torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
RUN if echo "${LEROBOT_EXTRAS}" | tr ',' '\n' | grep -qx 'groot'; then \
        pip3 uninstall --break-system-packages -y torch torchvision torchcodec && \
        pip3 freeze | grep -E '^nvidia-[a-z0-9-]+-cu12==' | cut -d= -f1 \
            | xargs -r pip3 uninstall --break-system-packages -y && \
        pip3 install --break-system-packages --no-cache-dir \
            --index-url https://download.pytorch.org/whl/cu130 \
            "torch==${GROOT_TORCH_VERSION}" \
            "torchvision==${GROOT_TORCHVISION_VERSION}" \
            "torchcodec==${GROOT_TORCHCODEC_VERSION}" && \
        pip3 install --break-system-packages --no-cache-dir \
            "dm-tree>=0.1.8,<1.0.0" \
            "peft>=0.18.0,<1.0.0" \
            "safetensors>=0.4.3,<1.0.0" \
            "Pillow>=10.0.0,<13.0.0" \
            einops && \
        pip3 install --break-system-packages --no-cache-dir --no-deps \
            "${GROOT_FLASH_ATTN_WHEEL}"; \
    fi

# Copy ROS2 package (changes often → late layer)
COPY ros2/src/lerobot_control /workspace/src/lerobot_control

# Copy configs (lerobot_control + cyclonedds)
COPY configs /workspace/configs

# Build ROS2 workspace
RUN . /opt/ros/jazzy/setup.sh && \
    cd /workspace && \
    colcon build --symlink-install

# Set entrypoint
COPY docker/inference/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["ros2", "launch", "lerobot_control", "inference.launch.py"]
