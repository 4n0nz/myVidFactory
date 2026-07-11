#!/bin/bash
# Re-train YOLO facecam v2 : dataset 3 sources (barbu masortie+lastvideo + Ofr) -> generalise cross-personne.
set -e
cd ~/yolo
PY=~/yolo/.venv/bin/python
YOLO=~/yolo/.venv/bin/yolo
echo "=== build dataset v2 ==="
$PY scripts/build_facecam_dataset.py
echo "=== train v2 (run=facecam2) ==="
$YOLO detect train model=yolov8n.pt data=$HOME/yolo/data/facecam/data.yaml \
  epochs=80 imgsz=960 batch=16 device=0 project=$HOME/yolo/runs name=facecam2 \
  exist_ok=True verbose=True patience=20
echo "=== validate v2 (3 videos incl. Ofr) ==="
$PY scripts/validate_v2.py
echo "RETRAIN_V2_DONE"
