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
        bx = med([b_[0] for b_ in boxes]); by = med([b_[1] for b_ in boxes])
        bw = med([b_[2] for b_ in boxes]); bh = med([b_[3] for b_ in boxes])
        fixed += 1
    elif s.get("bbox"):
        # YOLO muet sur ce seg -> on garde l'ancienne box MAIS pad+snap s'appliquent
        # quand meme (une box heritee borne aussi trop serre -> fuite du narrateur).
        bx, by, bw, bh = s["bbox"]
        missed += 1
    else:
        missed += 1
        continue
    if True:
        # overscan : pad 6% de la box par cote (la box YOLO borne serre le cadre
        # detecte ; la webcam reelle deborde parfois -> fuite du narrateur original).
        # Clampe a l'ecran ; dims restent normalisees 0-1.
        PAD = 0.06
        px = bw * PAD; py = bh * PAD
        x0 = max(0.0, bx - px); y0 = max(0.0, by - py)
        x1 = min(1.0, bx + bw + px); y1 = min(1.0, by + bh + py)
        # snap-to-edge : bord de box a <8% d'un bord ecran -> etend jusqu'au bord
        # (webcam de coin ; la bande box-bord = la ou le narrateur original fuit)
        SNAP = 0.08
        if x0 < SNAP: x0 = 0.0
        if y0 < SNAP: y0 = 0.0
        if x1 > 1.0 - SNAP: x1 = 1.0
        if y1 > 1.0 - SNAP: y1 = 1.0
        s["bbox"] = [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)]

cap.release()
json.dump(segs, open(hmap_path, "w"), indent=2)
print("YOLO extent: %d pip segs corriges, %d sans detection (ancienne box gardee)" % (fixed, missed))
for s in segs:
    if s["host"] == "pip":
        print("  PIP %.1f-%.1f bbox=%s" % (s["start"], s["end"], s["bbox"]))
