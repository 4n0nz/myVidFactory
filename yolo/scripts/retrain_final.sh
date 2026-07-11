#!/bin/bash
# Train final : dataset avec TOUTES les personnes harvestees dispo (incl. c6 = PiP spherique).
set -e
cd ~/yolo
PY=~/yolo/.venv/bin/python
YOLO=~/yolo/.venv/bin/yolo
echo "=== BUILD dataset (toutes personnes, incl. c6 spherique) ==="
$PY scripts/build_dataset_v3.py
echo "=== TRAIN final (facecam3) ==="
$YOLO detect train model=yolov8n.pt data=$HOME/yolo/data/facecam/data.yaml \
  epochs=80 imgsz=960 batch=16 device=0 project=$HOME/yolo/runs name=facecam3 \
  exist_ok=True verbose=True patience=25
echo "=== VALIDATE (5 nouveaux + c6 spherique + originaux) ==="
$PY scripts/validate_v3.py
echo RETRAIN_V3_DONE
