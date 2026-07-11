#!/usr/bin/env python3
# Remplace l'extent heuristique par la box YOLO facecam dans un host_map.json.
# Pour chaque segment PIP : YOLO sur ~5 frames -> mediane de la box -> ecrit seg['bbox'].
# Garde la classification (hero/pip/off) telle quelle. Si YOLO ne trouve rien -> garde l'ancienne box.
# Usage : yolo_extent.py <host_map.json> <source.mp4> [conf]
import sys, json, os, statistics
import cv2
from ultralytics import YOLO

hmap_path = sys.argv[1]
source = sys.argv[2]
CONF = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
WEIGHTS = os.path.expanduser("~/yolo/runs/facecam/weights/best.pt")

m = YOLO(WEIGHTS)
cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(3)); H = int(cap.get(4))
segs = json.load(open(hmap_path))

def best_box(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, f = cap.read()
    if not ok: return None
    r = m.predict(f, imgsz=960, conf=CONF, verbose=False)[0]
    if len(r.boxes) == 0: return None
    # la box la plus confiante
    b = max(r.boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
    return [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]

med = lambda L: statistics.median(L)
fixed = 0; missed = 0
for s in segs:
    if s["host"] != "pip":
        continue
    a, b = s["start"], s["end"]
    ts = [a + (b - a) * f for f in (0.2, 0.35, 0.5, 0.65, 0.8)]
    boxes = [bb for bb in (best_box(t) for t in ts) if bb]
    if len(boxes) >= 2:
        s["bbox"] = [round(med([bx[k] for bx in boxes]), 4) for k in range(4)]
        fixed += 1
    else:
        missed += 1  # garde l'ancienne box

cap.release()
json.dump(segs, open(hmap_path, "w"), indent=2)
print("YOLO extent: %d pip segs corriges, %d sans detection (ancienne box gardee)" % (fixed, missed))
for s in segs:
    if s["host"] == "pip":
        print("  PIP %.1f-%.1f bbox=%s" % (s["start"], s["end"], s["bbox"]))
