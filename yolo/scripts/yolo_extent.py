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
# Defaut = facecam5 epoch8 (= best_real.pt) : le checkpoint qui bat v3 sur les VRAIES frames
# (24/24, 6gtf/1x32/vkmx/masortie natifs). PAS best.pt de facecam5 (= overfit sur val SYNTHETIQUE).
# Rollback vers v3 : YOLO_WEIGHTS=~/yolo/runs/facecam/weights/best.pt
WEIGHTS = os.environ.get("YOLO_WEIGHTS", os.path.expanduser("~/yolo/runs/facecam5/weights/best_real.pt"))

m = YOLO(WEIGHTS)
cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(3)); H = int(cap.get(4))
segs = json.load(open(hmap_path))

def _frame(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, f = cap.read(); return f if ok else None

_BB_DT = 0.15; _BB_MTH = 12   # filtre fantome : diff de frames pour la densite de mouvement

def _box_motion(f0, f1, bx):
    """Fraction de pixels en mouvement dans la box bx (normalisee). 1.0 si pas de 2e frame (=ne filtre pas)."""
    if f1 is None: return 1.0
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    d = cv2.absdiff(g0, g1) > _BB_MTH
    x1, y1 = max(0, int(bx[0] * W)), max(0, int(bx[1] * H))
    x2, y2 = min(W, int((bx[0] + bx[2]) * W)), min(H, int((bx[1] + bx[3]) * H))
    if x2 - x1 < 2 or y2 - y1 < 2: return 0.0
    return float(d[y1:y2, x1:x2].mean())

def best_box(t):
    f = _frame(t)
    if f is None: return None
    r = m.predict(f, imgsz=960, conf=CONF, verbose=False)[0]
    if len(r.boxes) == 0: return None
    cands = []
    for b in r.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        cands.append((float(b.conf[0]), [x1 / W, y1 / H, (x2 - x1) / W, (y2 - y1) / H]))
    if len(cands) == 1:                         # 1 seule detection = pas d'ambiguite (INCHANGE vs avant)
        return cands[0][1]
    # >1 detection : YOLO donne parfois une box conf-haute TROP GRANDE qui englobe la vraie webcam
    # vivante + du vide immobile au-dessus (cf Ethx 560-606s epoch8 : box h=0.91 conf 0.84 vs vraie
    # webcam h=0.57 conf 0.71). Prendre la max-conf = box gonflee. Discriminant bulk = DENSITE de
    # mouvement : une webcam serree = personne vivante = mouvement DENSE ; une box gonflee = mouvement
    # dilue par le vide. On choisit la box la mieux remplie de mouvement (parmi celles conf plausible).
    f1 = _frame(t + _BB_DT)
    if f1 is None:
        return max(cands, key=lambda c: c[0])[1]   # pas de 2e frame -> ancien comportement (max conf)
    # densite = signal RELATIF (comparer les box entre elles), pas un seuil absolu : sur un segment
    # peu anime la densite est faible partout (0.006 vs 0.009) mais la box la mieux remplie reste
    # la vraie webcam. On prend donc la densite MAX parmi les conf-plausibles, sans plancher absolu
    # (un plancher reprendrait la max-conf = la box gonflee, cf bug Ethx).
    scored = [(_box_motion(f, f1, bx), cf, bx) for cf, bx in cands]
    ok = [s for s in scored if s[1] >= 0.40] or scored   # garde-fou conf (ecarte une box junk faible conf)
    return max(ok, key=lambda s: s[0])[2]

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

def raw_box(wa, wb, fallback_bbox):
    """Box brute pour la fenetre [wa,wb] : YOLO (mediane 5 samples) -> YuNet -> box heuristique.
    Retourne (bx,by,bw,bh, method) ou None. method in yolo|yunet|heur. Chaine INCHANGEE vs avant."""
    ts = [wa + (wb - wa) * f for f in (0.2, 0.35, 0.5, 0.65, 0.8)]
    boxes = [bb for bb in (best_box(t) for t in ts) if bb]
    if len(boxes) >= 2:
        return (med([q[0] for q in boxes]), med([q[1] for q in boxes]),
                med([q[2] for q in boxes]), med([q[3] for q in boxes]), "yolo")
    # YOLO muet -> FALLBACK #1 : YuNet (2e detecteur, trouve les webcams que YOLO rate).
    yb = [bb for bb in (yunet_box(t) for t in ts) if bb]
    if len(yb) >= 2:
        return (med([q[0] for q in yb]), med([q[1] for q in yb]),
                med([q[2] for q in yb]), med([q[3] for q in yb]), "yunet")
    # ni YOLO ni YuNet -> FALLBACK #2 : ancienne box heuristique (la garde anti-sur-couverture
    # dans finalize() la rabote a un coin prudent si elle est enorme).
    if fallback_bbox:
        bx, by, bw, bh = fallback_bbox
        return (bx, by, bw, bh, "heur")
    return None

def _detect_split(big, wa, wb):
    """SPLIT : panneau lateral pleine hauteur = une camera entiere (layout ecran divise),
    pas un pip flottant. Criteres stricts (anti faux-positif type ofr, panneau flottant) :
    colle a un bord lateral + touche haut ET bas + frontiere verticale STATIQUE nette.
    Retourne [x,y,w,h] du panneau entier, ou None."""
    bx0, by0, bw0, bh0 = big
    left = bx0 < 0.03
    right = bx0 + bw0 > 0.97
    if not (left or right): return None
    if by0 > 0.05 or by0 + bh0 < 0.95: return None
    guess = (bx0 + bw0) if left else bx0
    lo, hi = max(0.10, guess - 0.10), min(0.90, guess + 0.10)
    acc = None; n = 0
    for f_ in (0.2, 0.35, 0.5, 0.65, 0.8):
        fr = _frame(wa + (wb - wa) * f_)
        if fr is None: continue
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        sx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
        acc = sx if acc is None else acc + sx; n += 1
    if n < 3: return None
    acc = acc / n
    x_lo, x_hi = int(lo * W), int(hi * W)
    if x_hi - x_lo < 4: return None
    col = int(np.argmax(acc[x_lo:x_hi])) + x_lo
    if float(acc[col]) < 3.0 * float(np.median(acc)): return None   # pas de frontiere nette
    b = col / W
    if left:  return [0.0, 0.0, round(b, 4), 1.0]
    else:     return [round(b, 4), 0.0, round(1.0 - b, 4), 1.0]

def finalize(bx, by, bw, bh, wa, wb):
    """Extension bord motion-gated + garde anti-sur-couverture. Retourne [x,y,w,h]. INCHANGE vs avant.
    Voir commentaires historiques : snap motion-gated (1DOLq/1x32/Jjwv) + rabot anti-ecrasement (6GtF/Ethx)."""
    PAD = 0.015; EDGE_ZONE = 0.10; MOT_FILL = 0.05
    rx1, ry1 = bx + bw, by + bh
    mm = _motion_map((wa + wb) / 2)
    bxp, byp, rxp, ryp = bx * W, by * H, rx1 * W, ry1 * H
    def _ext(near, moves):
        return near and (mm is not None) and moves > MOT_FILL
    x0 = 0.0 if _ext(bx < EDGE_ZONE, mm is not None and _strip_moves(mm, 0, byp, bxp, ryp)) else max(0.0, bx - bw * PAD)
    y0 = 0.0 if _ext(by < EDGE_ZONE, mm is not None and _strip_moves(mm, bxp, 0, rxp, byp)) else max(0.0, by - bh * PAD)
    x1 = 1.0 if _ext(rx1 > 1 - EDGE_ZONE, mm is not None and _strip_moves(mm, rxp, byp, W, ryp)) else min(1.0, rx1 + bw * PAD)
    y1 = 1.0 if _ext(ry1 > 1 - EDGE_ZONE, mm is not None and _strip_moves(mm, bxp, ryp, rxp, H)) else min(1.0, ry1 + bh * PAD)
    fx0, fy0, fw, fh = x0, y0, x1 - x0, y1 - y0
    if fh > 0.72 or fw > 0.55 or fw * fh > 0.38:
        hcx, hcy = fx0 + fw / 2, fy0 + fh / 2
        big = (fx0, fy0, fw, fh)
        fw, fh = 0.26, 0.42
        # rabot ancre-VISAGE : si YuNet trouve un visage DANS la grande box, la box prudente
        # se centre dessus (visage au tiers haut) au lieu du coin geometrique — sinon un
        # narrateur en panneau lateral pleine hauteur recoit l'avatar sur le TORSE (coin bas).
        fpos = None
        for _t in (wa + (wb - wa) * _f for _f in (0.3, 0.5, 0.7)):
            _fr = _frame(_t)
            if _fr is None: continue
            _, _fcs = _yfd.detect(_fr)
            if _fcs is None: continue
            cand = []
            for _fc in _fcs:
                _fx, _fy, _fw2, _fh2 = _fc[0] / W, _fc[1] / H, _fc[2] / W, _fc[3] / H
                _cx, _cy = _fx + _fw2 / 2, _fy + _fh2 / 2
                if big[0] <= _cx <= big[0] + big[2] and big[1] <= _cy <= big[1] + big[3]:
                    cand.append((_fw2 * _fh2, _cx, _cy))
            if cand:
                cand.sort(reverse=True)
                import math as _m
                fpos = (float(cand[0][1]), float(cand[0][2]), float(_m.sqrt(cand[0][0])))
                break
        _sp = _detect_split(big, wa, wb)
        if _sp and fpos and _sp[0] - 0.02 <= fpos[0] <= _sp[0] + _sp[2] + 0.02:
            # vrai SPLIT : le panneau entier est la camera -> l'avatar couvre tout le panneau
            fx0, fy0, fw, fh = _sp
        elif fpos:
            # box dimensionnee sur la taille du visage (grand narrateur = grande box),
            # ancree pour couvrir cheveux/front (visage au ~42% de la hauteur de box).
            _fcx, _fcy, _fs = fpos
            fh = min(0.62, max(0.42, 3.2 * _fs))
            fw = min(0.50, max(0.26, fh * 0.62))
            fx0 = min(1.0 - fw, max(0.0, _fcx - fw / 2))
            fy0 = min(1.0 - fh, max(0.0, _fcy - 1.35 * _fs))
        else:
            fx0 = 0.0 if hcx < 0.5 else 1.0 - fw
            fy0 = 1.0 - fh if hcy > 0.35 else 0.0
    return [round(fx0, 4), round(fy0, 4), round(fw, 4), round(fh, 4)]

def intro_split_time(a, b):
    """CHANTIER #2 : webcam d'INTRO plus grosse qui retrecit en cours de segment (cf Ethx grosse->petite).
    Une seule box mediane par segment ecrase ce cas. CHIRURGICAL (bulk-safe) : on ne split QUE si le
    debut du segment est NETTEMENT plus grand (>=1.6x l'aire) que le regime stable de la 2e moitie.
    Pas de saut = None -> comportement IDENTIQUE a avant (vKMx/1x32/... intouches, aucune sous-seg aveugle).
    Retourne l'instant de bascule (absolu) ou None."""
    if a > 90.0:                            # un INTRO est par definition au DEBUT de la video ;
        return None                         # un segment qui commence tard (ex. vKMx 440s) n'en est pas un
    if (b - a) < 6.0:                       # trop court pour un vrai intro qui retrecit
        return None
    fracs = [0.03, 0.08, 0.14, 0.22, 0.32, 0.45, 0.60, 0.75, 0.90]
    samp = []
    for f in fracs:
        bb = best_box(a + (b - a) * f)      # YOLO seul = signal propre (box complete)
        if bb:
            samp.append((f, bb))
    late = [bx for f, bx in samp if f >= 0.35]
    early = [(f, bx) for f, bx in samp if f < 0.20]
    if len(late) < 2 or not early:
        return None
    ar = lambda bx: bx[2] * bx[3]
    steady = [med([bx[i] for bx in late]) for i in range(4)]   # box mediane du regime stable
    s_area = ar(steady)
    # coin ancre du regime stable (webcam collee a un coin d'ecran) : un VRAI resize garde ce
    # coin fixe, seule la taille change. Un saut de coin/position = autre chose (decoy, changement
    # de plan) -> PAS un resize -> on ne split pas. C'est ce qui distingue Ethx (meme coin bas-gauche,
    # accepte) de vKMx (la partie "grosse" saute en (0.20,0.64), autre position -> rejete).
    scx, scy = steady[0] + steady[2] / 2, steady[1] + steady[3] / 2
    sx = 0 if scx < 0.5 else 1; sy = 0 if scy < 0.5 else 1
    corner = lambda bx: (bx[0] if sx == 0 else bx[0] + bx[2], bx[1] if sy == 0 else bx[1] + bx[3])
    scn = corner(steady)
    big = []
    for f, bx in early:
        if ar(bx) >= 1.6 * s_area:                        # intro nettement plus grosse
            cn = corner(bx)
            if abs(cn[0] - scn[0]) <= 0.10 and abs(cn[1] - scn[1]) <= 0.10:  # meme coin = meme webcam
                big.append(f)
    if not big:
        return None
    f_big = max(big)
    after = [f for f, bx in samp if f > f_big and ar(bx) <= 1.3 * s_area]  # retour au stable
    f_small = min(after) if after else min(f_big + 0.1, 0.99)
    return a + (b - a) * (f_big + f_small) / 2

fixed = 0; yunet_hit = 0; missed = 0
out_segs = []
for s in segs:
    if s["host"] != "pip":
        out_segs.append(s)
        continue
    a, b = s["start"], s["end"]
    # split intro chirurgical : 2 fenetres seulement si un vrai resize est detecte ET que
    # CHAQUE sous-fenetre detecte quelque chose (sinon on retombe sur le segment entier = zero trou).
    tsplit = intro_split_time(a, b)
    parts = None
    if tsplit is not None:
        rb1 = raw_box(a, tsplit, s.get("bbox"))
        rb2 = raw_box(tsplit, b, s.get("bbox"))
        if rb1 and rb2:
            parts = [(a, tsplit, rb1), (tsplit, b, rb2)]
    if parts is None:
        rb = raw_box(a, b, s.get("bbox"))
        if rb is None:
            missed += 1
            out_segs.append(s)              # aucune detection -> segment inchange (bbox d'origine)
            continue
        parts = [(a, b, rb)]
    for wa, wb, rb in parts:
        bx, by, bw, bh, method = rb
        if method == "yolo": fixed += 1
        elif method == "yunet": yunet_hit += 1
        else: missed += 1
        bbox = finalize(bx, by, bw, bh, wa, wb)
        ns = s if len(parts) == 1 else dict(s)
        ns["start"], ns["end"], ns["bbox"] = wa, wb, bbox
        out_segs.append(ns)
segs = out_segs

# --- raffinement de frontiere : aligner les bornes sur le VRAI hard-cut (scene change) ---
# analyze_host2 echantillonne tous les 0.5s -> une frontiere de segment peut etre en RETARD jusqu'a
# 0.5s sur la vraie transition. Pendant ce retard, l'avatar reste a l'ancienne position pendant que
# la webcam a deja saute -> le vrai visage FUIT (cf ofr : webcam 2 positions, cut a 38.72 mais borne
# a 39.0 = 0.28s de fuite). Fix bulk : a chaque frontiere ou l'apparence de l'avatar change (pip<->
# autre, ou pip qui se DEPLACE), on cherche un hard-cut (pic de diff globale) dans +/-0.5s et on snap
# la borne au frame du cut. Ne touche QUE les frontieres a vrai scene-cut -> videos mono-position et
# sans cut net = intactes. Un slide sans cut (pas de pic) = borne inchangee (fallback sur).
_SC_WIN = 0.5; _SC_STEP = 0.05; _SC_TH = 25.0
def _scene_cut(t0, t1):
    prev = None; best_t = None; best_d = 0.0; tt = max(0.0, t0)
    while tt <= t1:
        f = _frame(tt)
        if f is not None:
            g = cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                d = float(cv2.absdiff(g, prev).mean())
                if d > best_d: best_d, best_t = d, tt
            prev = g
        tt += _SC_STEP
    return best_t if best_d > _SC_TH else None

def _pcen(s):
    b = s.get("bbox"); return (b[0] + b[2] / 2, b[1] + b[3] / 2) if b else None

refined = 0
for i in range(len(segs) - 1):
    s1, s2 = segs[i], segs[i + 1]
    if abs(s1["end"] - s2["start"]) > 1e-6:
        continue                                        # segments non adjacents (trou) -> on saute
    # CIBLE = le saut de POSITION de webcam (cf ofr, pos A<->B) : 2 pip adjacents dont le centre
    # bouge nettement. C'est le seul cas ou l'avatar reste visiblement au mauvais endroit pendant
    # que le vrai visage a saute. On NE touche PAS les frontieres pip<->hero/off (=> videos
    # mono-position et deja validees restent byte-identique).
    c1, c2 = _pcen(s1), _pcen(s2)
    moved = (s1["host"] == "pip" and s2["host"] == "pip" and c1 and c2
             and (abs(c1[0] - c2[0]) > 0.12 or abs(c1[1] - c2[1]) > 0.12))
    if not moved:
        continue
    bnd = s1["end"]
    cut = _scene_cut(bnd - _SC_WIN, bnd + _SC_WIN)
    if cut is not None and abs(cut - bnd) > 0.06:
        s1["end"] = round(cut, 2); s2["start"] = round(cut, 2)
        refined += 1
if refined:
    print("raffinement frontiere : %d borne(s) snappee(s) sur hard-cut" % refined)

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
