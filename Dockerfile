# VideoFactory - image reproductible (CUDA + ffmpeg + deps).
# Build : docker build -t videofactory .
# Run   : docker run --gpus all -v "$PWD/out:/root/videogen/out" videofactory wk_demo "<url>" demo.mp4
#   --gpus all = passe le GPU + NVENC de l'hote (necessite nvidia-container-toolkit sur l'hote).
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# capabilities driver = expose NVENC (video) au container
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# chemins du code (/home/boss) -> HOME du container (/root) + layout attendu ~/yolo ~/videogen
RUN sed -i 's|/home/boss|/root|g' videogen/build_seg.py videogen/run_one.sh yolo/scripts/yolo_extent.py \
    && ln -s /app/yolo /root/yolo && ln -s /app/videogen /root/videogen

RUN python3.10 -m venv /root/yolo/.venv \
    && /root/yolo/.venv/bin/pip install --no-cache-dir -U pip \
    && /root/yolo/.venv/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r /root/yolo/requirements.txt

RUN python3.10 -m venv /root/videogen/.venv \
    && /root/videogen/.venv/bin/pip install --no-cache-dir -U pip \
    && /root/videogen/.venv/bin/pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu128 -r /root/videogen/requirements.txt

WORKDIR /root/videogen
ENTRYPOINT ["bash", "run_one.sh"]
