# VideoFactory

A pipeline that **replaces the narrator** of a video (screen recording + webcam) with an
**avatar**, while keeping the on-screen content and the source audio. It detects the webcam
per segment (YOLO facecam + fallbacks), swaps the face for the avatar with rounded corners,
and renders with NVENC.

## Structure (monorepo, 2 sub-projects)

The pipeline = 2 parts that run together:

```
videogen/          # orchestration + compositor
  run_one.sh         # ENTRY POINT: download -> detection -> YOLO extent -> build -> render
  build_seg.py       # segmented compositor (overlays the avatar into the pip webcam)
  agent_yt/          # analyze_host2.py (hero/pip/off classification) + pipeline scripts
  agent_yt3/         # pipeline variants / helpers
  vf_qc.py           # coverage QC
  public/avatar.mp4  # <- the replacement avatar (included)
  face_detection_yunet_2023mar.onnx  # <- YuNet detector (included)
  requirements.txt

yolo/              # facecam detection
  scripts/yolo_extent.py   # precise webcam box per segment (YOLO -> YuNet -> corner)
                           # + intro split, motion-density filter, boundary refinement
  scripts/                 # synthetic dataset, training, validation, score_real.py
  runs/facecam5/weights/best_real.pt  # <- PROD MODEL epoch8 (included)
  runs/facecam/weights/best.pt        # <- v3 rollback model (included)
  requirements.txt
```

Models, avatar and onnx are **in the repo** (no external download). Only the venvs and the
temporary `wk_*` workdirs (per-video source + renders) live outside the repo.

## System requirements

- **Linux** + **NVIDIA GPU** (tested on RTX 5060 Ti 16 GB) with CUDA drivers
- **ffmpeg** built with **NVENC** (`h264_nvenc`) — the render uses it
- **Python 3.10**
- The YOLO model runs on CUDA (torch cu12x)

## Installation

Three ways, from easiest to most manual.

### A. Docker (recommended — fresh machine, zero manual install)

Host requirements: **Docker** + **nvidia-container-toolkit** (to pass the GPU into the container).
Nothing else to install (CUDA/ffmpeg/python are inside the image).

```bash
git clone https://github.com/4n0nz/myVidFactory.git && cd myVidFactory
docker build -t videofactory .
docker run --gpus all -v "$PWD/out:/root/videogen/out" videofactory wk_demo "<youtube_url>" demo.mp4
# result -> ./out/demo.mp4
```

`--gpus all` exposes the host GPU **and NVENC** to the container (via `NVIDIA_DRIVER_CAPABILITIES`).

### B. setup.sh (bare-metal — the machine already has an NVIDIA GPU + CUDA drivers)

```bash
git clone https://github.com/4n0nz/myVidFactory.git && cd myVidFactory
bash setup.sh          # apt: ffmpeg/python + venvs + pip install + symlinks + path fixups
cd ~/videogen && bash run_one.sh wk_demo "<youtube_url>" demo.mp4
```

`setup.sh` does **not** install the NVIDIA/CUDA drivers (too distro-dependent) — it only checks
that `nvidia-smi` responds. Install CUDA first if needed.

### C. Manual

```bash
git clone https://github.com/4n0nz/myVidFactory.git vf
ln -s "$PWD/vf/yolo" ~/yolo ; ln -s "$PWD/vf/videogen" ~/videogen
# fix the hardcoded paths if the user is not 'boss':
sed -i "s|/home/boss|$HOME|g" ~/videogen/build_seg.py ~/videogen/run_one.sh ~/yolo/scripts/yolo_extent.py
python3.10 -m venv ~/yolo/.venv     && ~/yolo/.venv/bin/pip     install --extra-index-url https://download.pytorch.org/whl/cu128 -r ~/yolo/requirements.txt
python3.10 -m venv ~/videogen/.venv && ~/videogen/.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r ~/videogen/requirements.txt
```

> **Hardcoded paths**: the code references `~/yolo`, `~/videogen` and a few `/home/boss/videogen/...`.
> Expected layout = a user whose `$HOME` contains `yolo/` and `videogen/`. For another user, adjust
> the `/home/boss/` paths (grep `/home/boss` in `build_seg.py` and `yolo/scripts/yolo_extent.py`).

## Running the pipeline

```bash
cd ~/videogen
bash run_one.sh <workdir> <youtube_url> <output.mp4>

# example
bash run_one.sh wk_demo "https://www.youtube.com/watch?v=XXXX" demo.mp4
```

- `<output.mp4>`: the **`.mp4` extension is mandatory** (otherwise the ffmpeg muxer fails at the end).
- Steps: download (yt-dlp) -> `analyze_host2.py` (hero/pip/off) -> `yolo_extent.py` (precise webcam
  box, epoch8 model by default) -> `build_seg.py` (avatar overlay) -> NVENC render.
- Output lands in `~/videogen/out/<output.mp4>`.
- Optional 4th arg `PIP_RECT="x,y,w,h"` to force a fixed webcam position (otherwise auto by detection).
- The avatar is fixed = `public/avatar.mp4`. To change the narrator, replace that file.

## Model

`yolo_extent.py` uses **`runs/facecam5/weights/best_real.pt`** (epoch8) by default.
Rollback to v3:

```bash
YOLO_WEIGHTS=~/yolo/runs/facecam/weights/best.pt bash run_one.sh ...
```

## Notes

- Mobile / multi-position webcam: handled per segment (detection + boundary refinement on hard cut).
- Open issue: a **large borderless webcam** (full screen side) is still under-covered (box = face only).
- Audio: the source track is copied as-is (tag `und`).
