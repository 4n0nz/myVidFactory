#!/usr/bin/env python3
# vf_cardscan.py <workdir> [--probes N] [--json out.json]
#   Mesure la CARTE reelle dans la SOURCE, scene pip par scene pip, et la compare a la
#   box appliquee. Ancre sur la SOURCE, jamais sur le prior ni sur le vert.
#
# POURQUOI : qc_fid est ancre sur le consensus depuis 861a5c3 — il verifie que le vert
# epouse le prior, jamais que le prior epouse la carte. Quand le prior vise le conteneur
# (fenetre d'app, colonne de layout) il confirme fidelement la mauvaise boite. Mesure
# 2026-08-01 : o9x8kycf3Wo 422-441 box x=316-778 pour une carte x=110-548 (47% de la
# largeur a nu) ; gnfHlIoh34Q 143-154 box y=0-1080 pour une carte y=143-942 (64% de
# surface en trop, vert colle aux deux bords ecran alors que la carte a des marges).
#
# METHODE — la lecon vKMxg du 30/07 : "une fenetre de recherche centree sur l'hypothese
# qu'on teste ne peut que la confirmer". Donc :
#   1. fenetre de recherche LARGE (box + 50% de sa taille, plancher 15% de l'ecran), pour
#      que la vraie carte soit trouvable meme si la box est franchement a cote ;
#   2. le fond de page est estime sur un ANNEAU autour de la fenetre, jamais dans la box ;
#   3. la carte = plus grande composante connexe qui differe du fond ;
#   4. on ne regarde le vert du rendu NULLE PART. Cet outil ne lit que la source.
import sys, os, json
import cv2, numpy as np

wd = sys.argv[1]
PROBES = int(sys.argv[sys.argv.index("--probes") + 1]) if "--probes" in sys.argv else 5
OUTJSON = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None

src = os.path.join(wd, "source.mp4")
hm = json.load(open(os.path.join(wd, "host_map.json")))
cap = cv2.VideoCapture(src)
W = int(cap.get(3)); H = int(cap.get(4))

def frame(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, f = cap.read()
    return f if ok else None

VG = "/home/boss/videogen"
_yfd = _rec = _narr = None
def _face_seed(fr):
    """Centre du visage du NARRATEUR dans la source, ou None. Sert d'ancre : sur une page
    chargee (canvas Make.com d'o9x8kycf3Wo) le plus gros bloc "different du fond" attrape
    la barre d'outils, pas la carte. Le visage, lui, est toujours DANS la carte — c'est
    l'ancre la plus sure qu'on ait, et elle est deja calculee ailleurs dans le pipeline.
    Mesure o9x8 t=428 : cos narrateur 0.662, tres au-dessus du seuil."""
    global _yfd, _rec, _narr
    if _narr is None:
        p = os.path.join(wd, "narrator_feat.npy")
        if not os.path.exists(p): return None
        _narr = np.load(p)
        _yfd = cv2.FaceDetectorYN.create(VG + "/face_detection_yunet_2023mar.onnx", "",
                                         (W, H), score_threshold=0.6)
        _rec = cv2.FaceRecognizerSF.create(VG + "/face_recognition_sface_2021dec.onnx", "")
    _, fs = _yfd.detect(fr)
    if fs is None: return None
    best = None
    for f in fs:
        try:
            ft = _rec.feature(_rec.alignCrop(fr, f)).flatten().astype(np.float32)
        except Exception:
            continue
        c = float(np.dot(ft, _narr) / (np.linalg.norm(ft) * np.linalg.norm(_narr) + 1e-9))
        if c >= 0.363 and (best is None or c > best[0]):
            best = (c, f[0] + f[2] / 2.0, f[1] + f[3] / 2.0)
    return None if best is None else (best[1], best[2])

def card_rect(fr, box):
    """Bbox de la carte reelle autour de `box`, mesuree dans la SOURCE.
    Rend (x0,y0,x1,y1,conf) ou None. conf = fraction de la fenetre occupee par la
    composante retenue (sert a jeter les mesures douteuses)."""
    bx0, by0 = box[0] * W, box[1] * H
    bx1, by1 = (box[0] + box[2]) * W, (box[1] + box[3]) * H
    mw = max(0.5 * (bx1 - bx0), 0.15 * W)
    mh = max(0.5 * (by1 - by0), 0.15 * H)
    X0 = int(max(0, bx0 - mw)); X1 = int(min(W, bx1 + mw))
    Y0 = int(max(0, by0 - mh)); Y1 = int(min(H, by1 + mh))
    if X1 - X0 < 20 or Y1 - Y0 < 20: return None
    roi = fr[Y0:Y1, X0:X1]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.int16)

    # fond estime sur un anneau de 8 px au bord de la FENETRE (donc hors box quand la
    # fenetre est large) — median robuste aux petits elements decoratifs.
    k = 8
    ring = np.concatenate([lab[:k].reshape(-1, 3), lab[-k:].reshape(-1, 3),
                           lab[:, :k].reshape(-1, 3), lab[:, -k:].reshape(-1, 3)])
    bg = np.median(ring, axis=0)
    d = np.linalg.norm(lab - bg, axis=2)
    thr = max(18.0, float(np.percentile(d, 60)))
    m = (d > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None

    # Composante CONTENANT le visage du narrateur si on le trouve ; sinon la plus grosse.
    # Sans cette ancre, o9x8kycf3Wo t=428 renvoyait (108,120,1066,176) — la barre d'outils
    # du navigateur — au lieu de la carte (110,344,548,866).
    seed = _face_seed(fr)
    c = None
    if seed is not None:
        sx, sy = seed[0] - X0, seed[1] - Y0
        if 0 <= sx < X1 - X0 and 0 <= sy < Y1 - Y0:
            for k in cs:
                if cv2.pointPolygonTest(k, (float(sx), float(sy)), False) >= 0:
                    c = k; break
    if c is None:
        if seed is not None: return None   # visage trouve mais hors composante : douteux
        c = max(cs, key=cv2.contourArea)
    a = cv2.contourArea(c)
    if a < 0.02 * (X1 - X0) * (Y1 - Y0): return None
    x, y, w, h = cv2.boundingRect(c)
    conf = a / float(w * h + 1e-9)          # remplissage du rectangle par la forme
    return X0 + x, Y0 + y, X0 + x + w, Y0 + y + h, conf

rows = []
for i, s in enumerate(hm):
    if s.get("host") != "pip" or not s.get("bbox"): continue
    b = [float(v) for v in s["bbox"]]
    bx0, by0 = b[0] * W, b[1] * H
    bx1, by1 = (b[0] + b[2]) * W, (b[1] + b[3]) * H
    t0, t1 = s["start"], s["end"]
    ts = [t0 + (t1 - t0) * f for f in np.linspace(0.12, 0.88, PROBES)]
    meas = []
    for t in ts:
        fr = frame(t)
        if fr is None: continue
        r = card_rect(fr, b)
        if r and r[4] >= 0.55: meas.append(r[:4])
    if not meas:
        rows.append(dict(i=i, t0=t0, t1=t1, status="CARTE-INTROUVABLE"))
        continue
    a = np.array(meas, dtype=np.float64)
    med = np.median(a, axis=0)
    spread = float(np.max(a[:, 0]) - np.min(a[:, 0]))   # deplacement horizontal
    cx0, cy0, cx1, cy1 = med
    # debordements du VERT (= de la box) au-dela de la carte, en pixels
    over = dict(g=cx0 - bx0, h=cy0 - by0, d=bx1 - cx1, b=by1 - cy1)
    box_a = (bx1 - bx0) * (by1 - by0)
    card_a = (cx1 - cx0) * (cy1 - cy0)
    inter = max(0, min(bx1, cx1) - max(bx0, cx0)) * max(0, min(by1, cy1) - max(by0, cy0))
    rows.append(dict(i=i, t0=t0, t1=t1, status="OK", n=len(meas),
                     card=[round(v, 1) for v in med],
                     box=[round(bx0, 1), round(by0, 1), round(bx1, 1), round(by1, 1)],
                     over={k: round(v, 1) for k, v in over.items()},
                     ratio=round(box_a / (card_a + 1e-9), 3),
                     couvert=round(100.0 * inter / (card_a + 1e-9), 1),
                     spread=round(spread, 1)))
cap.release()

# --- verdict ---------------------------------------------------------------
# Deux defauts distincts, deux seuils separes. Les seuils sont larges expres : cet outil
# SIGNALE, il ne corrige pas. Mieux vaut rater un debordement de 20 px que noyer le
# rapport de faux positifs — la carte a des coins arrondis et la box est un rect.
SOUS = 92.0     # % de la carte couverte par la box en dessous duquel c'est une fuite
SUR  = 1.35     # rapport surface box / surface carte au-dessus duquel c'est du gaspillage
bad = []
for r in rows:
    if r["status"] != "OK":
        bad.append(("CARTE-INTROUVABLE", r)); continue
    if r["couvert"] < SOUS: bad.append(("SOUS-COUVERTURE", r))
    elif r["ratio"] > SUR:  bad.append(("SUR-COUVERTURE", r))

vid = os.path.basename(os.path.abspath(wd)).replace("wk_b_", "")
print("%s : %d scenes pip, %d signalees" % (vid, len(rows), len(bad)))
for kind, r in bad:
    if r["status"] != "OK":
        print("  %-17s scene[%d] %.1f-%.1f" % (kind, r["i"], r["t0"], r["t1"])); continue
    o = r["over"]
    print("  %-17s scene[%d] %7.1f-%7.1f  carte=%s box=%s  couvert=%.1f%% ratio=%.2f  "
          "debord G%+d H%+d D%+d B%+d%s"
          % (kind, r["i"], r["t0"], r["t1"], r["card"], r["box"], r["couvert"], r["ratio"],
             o["g"], o["h"], o["d"], o["b"],
             "  CARTE MOBILE (%.0f px)" % r["spread"] if r["spread"] > 60 else ""))
if OUTJSON:
    json.dump({"video": vid, "rows": rows}, open(OUTJSON, "w"), indent=1)
sys.exit(1 if bad else 0)
