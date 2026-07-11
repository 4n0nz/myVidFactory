#!/bin/bash
# Build dataset synthetique -> train YOLOv8n (1 classe facecam) sur le GPU DevBox.
set -e
cd ~/yolo
PY=~/yolo/.venv/bin/python
YOLO=~/yolo/.venv/bin/yolo
echo "=== 1) build dataset ==="
$PY scripts/build_facecam_dataset.py
echo "=== 2) train (GPU) ==="
$YOLO detect train model=yolov8n.pt data=$HOME/yolo/data/facecam/data.yaml \
  epochs=80 imgsz=960 batch=16 device=0 project=$HOME/yolo/runs name=facecam \
  exist_ok=True verbose=True patience=20
echo "TRAIN_DONE weights=$HOME/yolo/runs/facecam/weights/best.pt"
