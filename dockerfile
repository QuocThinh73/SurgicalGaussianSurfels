# ==========================================================
# Base: CUDA 11.8 + Ubuntu 22.04 (GPU build toolchain)
# ==========================================================
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=appuser
ARG UID=1000
ARG GID=1000
ARG CONDA_PY=3.9
ARG ENV_NAME=SurgicalGaussianSurfels

# ----------------------------------------------------------
# OS packages
# ----------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    build-essential cmake ninja-build pkg-config \
    ffmpeg \
    libgl1 libglib2.0-0 libxrender1 libxext6 libsm6 \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------
# Create non-root user (nice for mounting host files)
# ----------------------------------------------------------
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

# ----------------------------------------------------------
# Miniconda
# ----------------------------------------------------------
ENV CONDA_DIR=/opt/conda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/mc.sh && \
    bash /tmp/mc.sh -b -p ${CONDA_DIR} && rm /tmp/mc.sh
ENV PATH="${CONDA_DIR}/bin:${PATH}"

# Use bash -lc so `conda activate` works in subsequent RUN steps
SHELL ["/bin/bash", "-lc"]

# ----------------------------------------------------------
# Conda env + core CUDA/PyTorch toolchain
# ----------------------------------------------------------
RUN conda create -y -n ${ENV_NAME} python=${CONDA_PY} && \
    echo "conda activate ${ENV_NAME}" >> /home/${USERNAME}/.bashrc

# Torch 2.1.2 + cu118 wheels; match the stack in your notes
RUN conda activate ${ENV_NAME} && \
    pip install --upgrade pip && \
    pip install \
      plyfile==0.8.1 \
      torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install tqdm matplotlib scikit-image mmcv-full pymeshlab ninja h5py \
               pytorch-lightning imageio==2.27.0 opencv-python imageio-ffmpeg \
               scipy dearpygui lpips timm && \
    pip install --force-reinstall -v "numpy==1.25.2"

# Optional: build arch list for faster torch compile ops (Ampere + Ada)
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"

# ----------------------------------------------------------
# Copy repo (assumes Dockerfile is at the repo root)
# ----------------------------------------------------------
WORKDIR /workspace
COPY . /workspace

# ----------------------------------------------------------
# Editable / submodule installs that need CUDA/toolchain
# ----------------------------------------------------------
# diff-surfel-rasterization, simple-knn, fused-ssim
RUN conda activate ${ENV_NAME} && \
    pip install -e submodules/diff-surfel-rasterization && \
    pip install -e submodules/simple-knn && \
    pip install -e submodules/fused-ssim

# PyTorch3D (builds against installed torch)
RUN conda activate ${ENV_NAME} && \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# tiny-cuda-nn (bindings for torch) — from GitHub
RUN conda activate ${ENV_NAME} && \
    pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"

# ----------------------------------------------------------
# Final touches
# ----------------------------------------------------------
# Make the non-root user own the workspace
RUN chown -R ${USERNAME}:${USERNAME} /workspace
USER ${USERNAME}

# Default shell drops into the conda env
ENV CONDA_DEFAULT_ENV=${ENV_NAME}
CMD ["/bin/bash", "-lc", "echo 'Env ready → conda activate ${ENV_NAME}' && bash"]
