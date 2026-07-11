#!/usr/bin/env python3
# vf_qc.py <host_map.json> [source.mp4] — QC auto du placement avatar.
#
# 2 couches :
#  (1) GEOMETRIE (host_map seul, sans video) : pip au centre / pip bref hors-coin.
#  (2) COUVERTURE (regarde la SOURCE, YuNet) : detecte le cas ou un segment 'hero'
#      (avatar plein ecran) recouvre en fait un screen-share dont le narrateur n'est
#      qu'un petit cam de COIN -> l'avatar ecrase le contenu. C'est le trou que la
#      couche geometrique ne voit pas (un hero n'a pas de bbox = "coherent" tout seul).
#      Oracle = le meme signal live-visage que le detecteur : un vrai hero = gros
#      visage centre ; un faux hero = petit visage decale en coin.
#
# La couche (2) est SAUTEE proprement si la source ou cv2/YuNet manquent (retro-compat).
# exit 0 = clean, 1 = flags. Ecrit <dir>/qc_report.txt.
import json, sys, os

hm_path = sys.argv[1]
hm = json.load(open(hm_path))
hm_dir = os.path.dirname(os.path.abspath(hm_path))
src_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(hm_dir, 'source.mp4')

flags = []

# ---------- couche (1) GEOMETRIE (inchangee) ----------
for s in hm:
    if s.get('host') != 'pip' or not s.get('bbox'):
        continue
    bx, by, bw, bh = s['bbox']
    cx = bx + bw / 2
    dur = s['end'] - s['start']
    corner = (bx > 0.66 or bx + bw < 0.34 or by < 0.12 or by + bh > 0.88)
    if 0.34 <= cx <= 0.66:
        flags.append((s, "pip au CENTRE (cx=%.2f) — webcam attendue en coin, probable video-dans-contenu" % cx))
    elif dur < 3.0 and not corner:
        flags.append((s, "pip bref (%.1fs) hors-coin — probable faux positif transitoire" % dur))

# ---------- couche (2) COUVERTURE (source + YuNet) ----------
YUNET = '/home/boss/videogen/face_detection_yunet_2023mar.onnx'
# Discriminant hero legitime vs faux-hero : TAILLE + POSITION du visage.
# - Un GROS visage (h>=0.30) = plan serre plein ecran = vrai hero, PEU IMPORTE la position
#   (un createur peut se cadrer sur le cote, regle des tiers). Jamais flag.
# - Un cam de COIN recouvert (screen-share) = PETIT visage colle a un bord
#   (cas 1DOLq : h=0.21, cx=0.17). C'est ca qu'on veut attraper.
# Donc on flag seulement si le visage n'est PAS gros ET est decale :
#  - coin FRANC (cx<0.25 ou >0.75) -> flag ; ou decale MODERE (<0.34/>0.66) + tres petit (h<0.16).
HERO_BIG_FACE = 0.30   # au-dessus = plan serre legitime, on ne flag jamais
HERO_MIN_FACE = 0.16
DEEP_L, DEEP_R = 0.25, 0.75
MILD_L, MILD_R = 0.34, 0.66
SAMPLES_PER_SEG = 4
cov_note = "couverture: OK"
try:
    import cv2, numpy as np
    if not os.path.exists(src_path) or not os.path.exists(YUNET):
        raise RuntimeError("source ou modele YuNet absent")
    cap = cv2.VideoCapture(src_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if W == 0 or H == 0:
        raise RuntimeError("source illisible")
    fd = cv2.FaceDetectorYN.create(YUNET, "", (W, H), score_threshold=0.5, nms_threshold=0.3, top_k=5000)

    def dominant_face(t):
        """Visage le plus GROS a l'instant t -> (fh/H, fcx/W) ou None si aucun visage."""
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, fr = cap.read()
        if not ok or fr is None: return None
        _, faces = fd.detect(fr)
        if faces is None or len(faces) == 0: return None
        best = max(faces, key=lambda f: f[2] * f[3])  # w*h
        fx, fy, fw, fh = best[0], best[1], best[2], best[3]
        return (fh / H, (fx + fw / 2) / W)

    n_hero_checked = 0
    for s in hm:
        if s.get('host') != 'hero':
            continue
        a, b = s['start'], s['end']
        dur = b - a
        if dur < 1.0:  # trop court pour juger
            continue
        # echantillonne a l'interieur du segment (evite les 0.3s de bord = HERO_PAD)
        pad = min(0.4, dur * 0.15)
        ts = [a + pad + (dur - 2 * pad) * (k + 0.5) / SAMPLES_PER_SEG for k in range(SAMPLES_PER_SEG)]
        fhs, fcxs = [], []
        for t in ts:
            r = dominant_face(t)
            if r is None: continue
            fhs.append(r[0]); fcxs.append(r[1])
        if not fhs:
            continue  # aucun visage detecte = probable b-roll absorbe -> on ne flag pas (evite le bruit)
        n_hero_checked += 1
        mfh = sorted(fhs)[len(fhs) // 2]      # mediane hauteur visage
        mfcx = sorted(fcxs)[len(fcxs) // 2]   # mediane position x
        if mfh >= HERO_BIG_FACE:
            continue  # gros visage = plan serre plein ecran legitime, jamais un faux hero
        deep = mfcx < DEEP_L or mfcx > DEEP_R
        mild = mfcx < MILD_L or mfcx > MILD_R
        small = mfh < HERO_MIN_FACE
        if deep or (mild and small):
            flags.append((s, "HERO SUSPECT — source = visage de COIN (h=%.2f, cx=%.2f) : l'avatar plein ecran ECRASE probablement un screen-share (devrait etre pip)" % (mfh, mfcx)))
    cap.release()
    cov_note = "couverture: %d segs hero verifies (YuNet)" % n_hero_checked
except Exception as e:
    cov_note = "couverture: SAUTEE (%s)" % e

npip = sum(1 for s in hm if s.get('host') == 'pip')
nhero = sum(1 for s in hm if s.get('host') == 'hero')
lines = ["QC host_map: %d segs (hero=%d pip=%d), flags=%d" % (len(hm), nhero, npip, len(flags)),
         "  %s" % cov_note]
for s, why in flags:
    lines.append("  FLAG %.1f-%.1fs — %s" % (s['start'], s['end'], why))
report = "\n".join(lines)
print(report)
open(os.path.join(hm_dir, 'qc_report.txt'), 'w').write(report + "\n")
sys.exit(1 if flags else 0)
