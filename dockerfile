FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip \
    git \
    build-essential \
    cmake \
    ninja-build \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
        plyfile==0.8.1 \
        tqdm \
        matplotlib \
        scikit-image \
        mmcv-full \
        pymeshlab \
        ninja \
        h5py \
        pytorch-lightning \
        imageio==2.27.0 \
        opencv-python \
        imageio-ffmpeg \
        scipy \
        dearpygui \
        lpips \
        timm \
        tensorboard \
        open3d && \
    python3 -m pip install \
        torch==2.1.2 \
        torchvision==0.16.2 \
        torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu118 && \
    python3 -m pip install -e submodules/diff-surfel-rasterization --use-pep517 --no-build-isolation && \
    python3 -m pip install -e submodules/simple-knn --use-pep517 --no-build-isolation && \
    python3 -m pip install -e submodules/fused-ssim --use-pep517 --no-build-isolation && \
    python3 -m pip install "git+https://github.com/facebookresearch/pytorch3d.git" --no-build-isolation && \
    python3 -m pip install "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch" && \
    python3 -m pip install --force-reinstall -v "numpy==1.25.2"

CMD ["bash"]