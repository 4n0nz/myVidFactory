#!/usr/bin/env python3
# Remplace l'extent heuristique par la box YOLO facecam dans un host_map.json.
# Pour chaque segment PIP : YOLO sur ~5 frames -> mediane de la box -> ecrit seg['bbox'].
# Garde la classification (hero/pip/off) telle quelle. Si YOLO ne trouve rien -> garde l'ancienne box.
# Usage : yolo_extent.py <host_map.json> <source.mp4> [conf]
import sys, json, os, statistics
import cv2
import numpy as np
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

def _frame(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read(); return f if ok else None

def best_box(t):
    f = _frame(t)
    if f is None: return None
    r = m.predict(f, imgsz=960, conf=CONF, verbose=False)[0]
    if len(r.boxes) == 0: return None
    b = max(r.boxes, key=lambda b: float(b.conf[0]))  # la plus confiante
    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
    return [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]

# --- YuNet = detecteur de SECOURS quand YOLO rate un facecam (createur inconnu).
# YuNet multi-echelle + live-motion trouve souvent la vraie webcam (c'est lui qui
# harveste les crops du dataset). Bounding SERRE facon harvest_crops (anti-bridge vers
# le contenu) -> vraie taille/position, pas une box devinee. Sert de fallback #1 avant
# la box heuristique. Solution BULK : marche sur tout createur que YuNet detecte.
YUNET = '/home/boss/videogen/face_detection_yunet_2023mar.onnx'
_yfd = cv2.FaceDetectorYN.create(YUNET, "", (W, H), score_threshold=0.5, nms_threshold=0.3, top_k=5000)
_yfd.setInputSize((W, H))
_MK = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
_MOT_DT = 0.15; _MOT_TH = 12; _LIVE = 0.12

def yunet_box(t):
    """Box webcam via YuNet + live-motion (facecam vivant de coin, pas une thumbnail statique).
    Retourne [x,y,w,h] normalise ou None. Meme logique que harvest_crops (bounding serre)."""
    f0 = _frame(t)
    if f0 is None: return None
    f1 = _frame(t + _MOT_DT); f1 = f1 if f1 is not None else f0
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    mot = (cv2.absdiff(g0, g1) > _MOT_TH).astype(np.uint8) * 255
    motc = cv2.dilate(cv2.morphologyEx(mot, cv2.MORPH_CLOSE, _MK, iterations=2), _MK, iterations=1)
    _, lab, stats, _ = cv2.connectedComponentsWithStats(motc)
    _, faces = _yfd.detect(f0)
    faces = faces if faces is not None else []
    best = None
    for fc in faces:
        fx, fy, fw, fh = int(fc[0]), int(fc[1]), int(fc[2]), int(fc[3])
        if fw < 8 or fh < 8 or fw > 0.42 * W: continue          # inset seulement (pas plein ecran)
        cx = min(W - 1, max(0, fx + fw // 2)); cy = min(H - 1, max(0, fy + fh // 2))
        rx0, ry0 = max(0, fx - fw // 2), max(0, fy - fh // 2)
        rx1, ry1 = min(W, fx + fw + fw // 2), min(H, fy + fh + 2 * fh)
        if motc[ry0:ry1, rx0:rx1].mean() / 255.0 < _LIVE: continue   # region morte -> thumbnail, pas webcam
        lid = int(lab[cy, cx])
        if lid > 0:
            mx, my, mw, mh = int(stats[lid, 0]), int(stats[lid, 1]), int(stats[lid, 2]), int(stats[lid, 3])
        else:
            mx, my, mw, mh = fx, fy, fw, fh
        cx0, cy0 = max(0, fx - fw), max(0, fy - fh); cx1, cy1 = min(W, fx + 2 * fw), min(H, fy + 3 * fh)
        ax, ay = max(mx, cx0), max(my, cy0); ax2, ay2 = min(mx + mw, cx1), min(my + mh, cy1)
        aw, ah = ax2 - ax, ay2 - ay
        if aw < 40 or ah < 40: continue
        area = aw * ah
        if best is None or area > best[4]:
            best = (ax, ay, aw, ah, area)
    if best:
        ax, ay, aw, ah, _ = best
        return [ax / W, ay / H, aw / W, ah / H]
    return None

def _motion_map(t):
    """Carte binaire du mouvement a l'instant t (diff avec +0.15s). None si illisible."""
    f0 = _frame(t)
    if f0 is None: return None
    f1 = _frame(t + _MOT_DT); f1 = f1 if f1 is not None else f0
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    return (cv2.absdiff(g0, g1) > _MOT_TH).astype(np.uint8)

def _strip_moves(mm, x0, y0, x1, y1):
    """Fraction de pixels en mouvement dans la bande (coords PIXEL)."""
    x0, x1 = max(0, int(x0)), min(W, int(x1)); y0, y1 = max(0, int(y0)), min(H, int(y1))
    if x1 - x0 < 2 or y1 - y0 < 2: return 0.0
    return float(mm[y0:y1, x0:x1].mean())

med = lambda L: statistics.median(L)
fixed = 0; yunet_hit = 0; missed = 0
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
    else:
        # YOLO muet -> FALLBACK #1 : YuNet (2e detecteur, trouve les webcams que YOLO rate).
        yb = [bb for bb in (yunet_box(t) for t in ts) if bb]
        if len(yb) >= 2:
            bx = med([b_[0] for b_ in yb]); by = med([b_[1] for b_ in yb])
            bw = med([b_[2] for b_ in yb]); bh = med([b_[3] for b_ in yb])
            yunet_hit += 1
        elif s.get("bbox"):
            # ni YOLO ni YuNet -> FALLBACK #2 : ancienne box heuristique (la garde
            # anti-sur-couverture plus bas la rabote a un coin prudent si elle est enorme).
            bx, by, bw, bh = s["bbox"]
            missed += 1
        else:
            missed += 1
            continue
    if True:
        # extension PAR-BORD, conditionnee a la proximite de la box BRUTE au bord.
        # Un bord de box qui TOUCHE deja le bord ecran (<NEAR) = webcam de coin collee ->
        # on snap au bord (sinon le vrai narrateur fuit, cf 1DOLq). Un bord EN RETRAIT
        # (webcam flottante avec marge, cf 1x32) = on NE pousse PAS vers le bord, juste un
        # overscan doux (PAD) pour couvrir le cadre/coins arrondis. Evite le debord dans la
        # marge sombre des webcams inset tout en gardant la couverture des webcams au bord.
        # SNAP MOTION-GATED (content-aware, bulk) : au bord d'une box proche du bord ecran,
        # on regarde la BANDE entre la box et le bord. Bande qui BOUGE = la webcam continue
        # (le narrateur fuit) -> on etend jusqu'au bord. Bande STATIQUE = marge/wallpaper ->
        # on garde le retrait (juste micro-overscan). Un seuil de position seul ne distingue
        # pas "vraie marge 2.5%" (1x32) de "webcam au bord sous-estimee" (Jjwv) ; le mouvement si.
        PAD = 0.015; EDGE_ZONE = 0.10; MOT_FILL = 0.05
        rx1, ry1 = bx + bw, by + bh
        mm = _motion_map((a + b) / 2)
        bxp, byp, rxp, ryp = bx * W, by * H, rx1 * W, ry1 * H
        def _ext(near, moves):
            return near and (mm is not None) and moves > MOT_FILL
        x0 = 0.0 if _ext(bx < EDGE_ZONE, mm is not None and _strip_moves(mm, 0, byp, bxp, ryp)) else max(0.0, bx - bw * PAD)
        y0 = 0.0 if _ext(by < EDGE_ZONE, mm is not None and _strip_moves(mm, bxp, 0, rxp, byp)) else max(0.0, by - bh * PAD)
        x1 = 1.0 if _ext(rx1 > 1 - EDGE_ZONE, mm is not None and _strip_moves(mm, rxp, byp, W, ryp)) else min(1.0, rx1 + bw * PAD)
        y1 = 1.0 if _ext(ry1 > 1 - EDGE_ZONE, mm is not None and _strip_moves(mm, bxp, ryp, rxp, H)) else min(1.0, ry1 + bh * PAD)
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
print("YOLO extent: %d par YOLO, %d par YuNet (fallback), %d sans detection (box heuristique)" % (fixed, yunet_hit, missed))
for s in segs:
    if s["host"] == "pip":
        print("  PIP %.1f-%.1f bbox=%s" % (s["start"], s["end"], s["bbox"]))
