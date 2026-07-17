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
#  (3) VLM (source + gemma3 local via Ollama) : juge la SCENE comme un humain — 1 frame
#      mediane par segment. pip: la region ou le VLM voit la webcam doit matcher le quadrant
#      de la box (sinon decoy / cam ratee). hero: la frame doit etre un talking-head plein
#      ecran (sinon faux hero). Toute contradiction est CONFIRMEE sur une 2e frame avant
#      de flagger (anti-bruit). Attrape les 3 familles batch20 : decoys longue duree,
#      faux heros, cams jamais detectees. ~2s/frame (gemma3 charge). Env QC_VLM=0 pour couper.
#
# Les couches (2) et (3) sont SAUTEES proprement si source/cv2/YuNet/Ollama manquent.
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

# ---------- couche (3) VLM (source + gemma3 via Ollama) ----------
vlm_note = "vlm: OFF"
if os.environ.get('QC_VLM', '1') != '0':
    try:
        import cv2
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'agent_yt'))
        import vlm_probe

        vcap = cv2.VideoCapture(src_path)
        if not vcap.isOpened():
            raise RuntimeError("source illisible")

        def _vframe(t):
            vcap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, fr = vcap.read()
            return fr if ok else None

        def _quadrant(b):
            cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
            if 0.30 <= cx <= 0.70 and 0.30 <= cy <= 0.70:
                return "center"
            return ("top" if cy < 0.5 else "bottom") + "-" + ("left" if cx < 0.5 else "right")

        already = set(id(s) for s, _ in flags)
        n_vlm = 0
        for s in hm:
            dur = s['end'] - s['start']
            if dur < 2.0 or id(s) in already:
                continue
            mid = s['start'] + dur / 2
            alt = s['start'] + dur * 0.25
            if s.get('host') == 'pip' and s.get('bbox'):
                b = s['bbox']
                if b[2] >= 0.5 or b[3] >= 0.72:
                    continue  # colonne/split : crop plein de contenu, juge pas fiable
                fr = _vframe(mid)
                if fr is None: continue
                # test CROP (fiable, valide sur vkmx page-a-vignettes) : la box contient-elle
                # une vraie webcam ? La recherche ouverte de region = distraite par les decoys.
                w1 = vlm_probe.webcam_crop(fr, b)
                n_vlm += 1
                if w1 is not False:
                    continue
                fr2 = _vframe(alt)
                w2 = vlm_probe.webcam_crop(fr2, b) if fr2 is not None else None
                if w2 is False:
                    hint = vlm_probe.region(fr)
                    where = (" (VLM voit la webcam en %s)" % hint) if hint and hint != "none" else ""
                    flags.append((s, "VLM: la box pip %s ne contient PAS de webcam (2 frames) — decoy ou cam ratee%s" % (_quadrant(b), where)))
            elif s.get('host') == 'hero':
                fr = _vframe(mid)
                if fr is None: continue
                f1 = vlm_probe.fullface(fr)
                n_vlm += 1
                if f1 is not False:
                    continue
                fr2 = _vframe(alt)
                f2 = vlm_probe.fullface(fr2) if fr2 is not None else None
                if f2 is False:
                    r = vlm_probe.region(fr)
                    where = (" (webcam vue en %s)" % r) if r and r != "none" else ""
                    flags.append((s, "VLM: pas un talking-head plein ecran (2 frames)%s — faux hero, l'avatar ecrase du contenu" % where))
        vcap.release()
        vlm_note = "vlm: %d segs juges (%s)" % (n_vlm, os.environ.get('VLM_MODEL', 'gemma3:12b'))
    except Exception as e:
        vlm_note = "vlm: SAUTE (%s)" % e

npip = sum(1 for s in hm if s.get('host') == 'pip')
nhero = sum(1 for s in hm if s.get('host') == 'hero')
lines = ["QC host_map: %d segs (hero=%d pip=%d), flags=%d" % (len(hm), nhero, npip, len(flags)),
         "  %s" % cov_note,
         "  %s" % vlm_note]
for s, why in flags:
    lines.append("  FLAG %.1f-%.1fs — %s" % (s['start'], s['end'], why))
report = "\n".join(lines)
print(report)
open(os.path.join(hm_dir, 'qc_report.txt'), 'w').write(report + "\n")
sys.exit(1 if flags else 0)
