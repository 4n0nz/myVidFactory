#!/bin/bash
# Pipeline complet v3 : harvest 5 nouveaux createurs (+3 existants) -> dataset 8 personnes -> train -> valide.
set -e
cd ~/yolo
PY=~/yolo/.venv/bin/python
YOLO=~/yolo/.venv/bin/yolo
echo "=== HARVEST (download + crops) ==="
bash ~/yolo/scripts/harvest_all.sh
echo "=== BUILD dataset v3 ==="
$PY scripts/build_dataset_v3.py
echo "=== TRAIN v3 (facecam3) ==="
$YOLO detect train model=yolov8n.pt data=$HOME/yolo/data/facecam/data.yaml \
  epochs=80 imgsz=960 batch=16 device=0 project=$HOME/yolo/runs name=facecam3 \
  exist_ok=True verbose=True patience=25
echo "=== VALIDATE v3 ==="
$PY scripts/validate_v3.py
echo RETRAIN_V3_DONE
