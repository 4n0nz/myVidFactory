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
        # YOLO muet sur ce seg -> pas de mesure fiable, on garde l'ancienne box heuristique.
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
        fx0, fy0, fw, fh = x0, y0, x1 - x0, y1 - y0
        # GARDE anti-sur-couverture (fallback prudent), sur la box FINALE : une box pip
        # implausiblement GRANDE (h>0.72 / large / aire>0.38) = YOLO muet (box heuristique
        # full-height) OU detection foireuse sur createur inconnu (cf 6GtF h=0.85). Un tel
        # avatar ECRASE tout le contenu. On la rabote a une webcam de COIN prudente, ancree
        # au coin qu'elle vise. Mieux un avatar un peu petit dans le bon coin que plein ecran.
        # Seuil haut (0.72) pour epargner une vraie grosse webcam d'intro (ex. Ethx ~0.70).
        if fh > 0.72 or fw > 0.55 or fw * fh > 0.38:
            hcx, hcy = fx0 + fw / 2, fy0 + fh / 2
            fw, fh = 0.26, 0.42   # taille webcam de coin plausible (assez haute pour couvrir tete+epaules)
            fx0 = 0.0 if hcx < 0.5 else 1.0 - fw
            fy0 = 1.0 - fh if hcy > 0.35 else 0.0
        s["bbox"] = [round(fx0, 4), round(fy0, 4), round(fw, 4), round(fh, 4)]

cap.release()

# --- cleanup anti-decoy : pip bref qui saute loin de la webcam stable ---
def _ctr(b): return (b[0] + b[2] / 2, b[1] + b[3] / 2)
pips = [s for s in segs if s["host"] == "pip" and s.get("bbox")]
anchors = [s for s in pips if (s["end"] - s["start"]) >= 3.0]
snapped = 0
if anchors:
    dom = max(anchors, key=lambda s: s["end"] - s["start"])  # webcam la plus persistante
    acx, acy = _ctr(dom["bbox"])
    darea = dom["bbox"][2] * dom["bbox"][3]
    for s in pips:
        if s is dom:
            continue
        cx, cy = _ctr(s["bbox"])
        brief = (s["end"] - s["start"]) < 3.0
        far = abs(cx - acx) > 0.2 or abs(cy - acy) > 0.2
        same_corner = abs(cx - acx) <= 0.15 and abs(cy - acy) <= 0.15
        area = s["bbox"][2] * s["bbox"][3]
        mis_size = same_corner and (area > darea * 1.6 or area < darea * 0.6)
        if (brief and far) or mis_size:
            s["bbox"] = list(dom["bbox"])  # colle a la vraie webcam (position + taille)
            snapped += 1
else:
    # aucune webcam stable -> les pips brefs hors-coin deviennent off (pas d'avatar sur un decoy)
    for s in pips:
        if (s["end"] - s["start"]) < 3.0:
            cx, cy = _ctr(s["bbox"])
            if 0.34 <= cx <= 0.66:
                s["host"] = "off"; s["bbox"] = None; snapped += 1
if snapped:
    print("cleanup anti-decoy : %d pip bref(s) recale(s)" % snapped)

json.dump(segs, open(hmap_path, "w"), indent=2)
print("YOLO extent: %d pip segs corriges, %d sans detection (ancienne box gardee)" % (fixed, missed))
for s in segs:
    if s["host"] == "pip":
        print("  PIP %.1f-%.1f bbox=%s" % (s["start"], s["end"], s["bbox"]))
