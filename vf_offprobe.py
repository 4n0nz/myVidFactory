#!/usr/bin/env python3
# vf_offprobe.py <workdir> [--json out.json] — miroir de vf_headcover, cote SOURCE.
#   Deux controles que PERSONNE ne fait (plan Efforts/VF-Plan-Robustesse.md, #1) :
#
#   (a) OFF-LIVE : une scene "off" ne peint rien — si le narrateur y est LIVE, son
#       visage passe a nu et AUCUN juge ne le voit : qc_ident cherche dans le RENDU
#       (le narrateur non peint y est... visible, mais qc_ident n'audite que la
#       coherence avec le plan via covered_by_avatar), les invariants ne voient que le
#       pavage (une scene off est LEGALE), et le trou de rognage d'itWI (20,2 s de
#       visage a nu, scene off issue du rognage de fin de pip par les sondes) est passe
#       inv=OK. Ici : sonder la SOURCE sur chaque scene off, visage narrateur
#       (cos>=0.363, le seuil du pipeline) + porte de MOUVEMENT (doctrine : le
#       narrateur en banniere/photo/b-roll est du CONTENU, pas une fuite — piege
#       verifie KKniWb9RKq4, sa tete est dans la banniere Skool affichee a l'ecran).
#       Si present : bissection grossiere pour rapporter les bornes reelles.
#
#   (b) MARGE-TETE (pip) : la tete du narrateur trouvee dans la SOURCE doit tenir DANS
#       la bbox de la scene. Marge negative = la box coupe la tete = fuite garantie au
#       rendu, detectable AVANT de bruler 25 min de rendu. Geometrie pure une fois la
#       tete detectee (1-3 sondes/scene). C'est le morceau "pre-rendu" du plan #2,
#       place ici et pas dans vf_invariants pour garder aux invariants leur identite
#       zero-pixel.
#
#   RAPPORT SEUL : exit 1 si flags, mais aucun fixer ne consomme ce verdict. Champ op=
#   dans la ligne DONE de vf_one.sh. Regle gravee (verification adversariale du plan) :
#   OFF-LIVE = SUSPECT a perpetuite, jamais fatal, jamais de fixer automatique — un
#   narrateur live plein cadre et son b-roll de lui-meme sont indiscernables sans
#   jugement. Le flag pointe, la loop confirme aux frames.
import sys, os, json
import cv2, numpy as np

wd = sys.argv[1]
OUTJSON = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
VG = "/home/boss/videogen"
COS = 0.363
MOTION_DT = 0.5
MOTION_MIN = 0.10

hm = json.load(open(os.path.join(wd, "host_map.json")))
feat_p = os.path.join(wd, "narrator_feat.npy")
if not os.path.exists(feat_p):
    print("offprobe : pas de narrator_feat, rien a verifier"); sys.exit(0)
narr = np.load(feat_p)
cap = cv2.VideoCapture(os.path.join(wd, "source.mp4"))
W = int(cap.get(3)); H = int(cap.get(4))
yfd = cv2.FaceDetectorYN.create(VG + "/face_detection_yunet_2023mar.onnx", "", (W, H),
                                score_threshold=0.6)
rec = cv2.FaceRecognizerSF.create(VG + "/face_recognition_sface_2021dec.onnx", "")

def frame(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, f = cap.read()
    return f if ok else None

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def narr_faces(fr):
    """Boxes YuNet du narrateur (cos>=COS) dans la frame."""
    _, fs = yfd.detect(fr)
    if fs is None: return []
    out = []
    for f in fs:
        try:
            ft = rec.feature(rec.alignCrop(fr, f)).flatten().astype(np.float32)
        except Exception:
            continue
        if cos(ft, narr) >= COS:
            out.append((float(f[0]), float(f[1]), float(f[2]), float(f[3])))
    return out

def moving(t, fb):
    """La zone du visage bouge-t-elle entre t et t+MOTION_DT ? (meme porte que
    qc_ident/_hot et vf_headcover : une image statique de soi = contenu)."""
    a = frame(t); b = frame(t + MOTION_DT)
    if a is None or b is None: return False
    x, y, w, h = [int(v) for v in fb]
    m = int(0.3 * max(w, h))
    x0, y0 = max(0, x - m), max(0, y - m)
    x1, y1 = min(W, x + w + m), min(H, y + h + m)
    pa, pb = a[y0:y1, x0:x1], b[y0:y1, x0:x1]
    if pa.size == 0 or pa.shape != pb.shape: return False
    return float((cv2.absdiff(cv2.cvtColor(pa, cv2.COLOR_BGR2GRAY),
                              cv2.cvtColor(pb, cv2.COLOR_BGR2GRAY)) > 18).mean()) >= MOTION_MIN

def live_at(t):
    fr = frame(t)
    if fr is None: return False
    for fb in narr_faces(fr):
        if moving(t, fb): return True
    return False

flags = []
nprobe = 0
for s in hm:
    t0, t1 = s["start"], s["end"]; dur = t1 - t0

    if s.get("host") == "off" and dur >= 1.0:
        # bornes (le trou de rognage colle aux frontieres) + interieur stratifie
        ts = [t0 + 0.15, t1 - 0.15]
        if dur > 4:  ts.append(t0 + dur * 0.5)
        if dur > 12: ts += [t0 + dur * 0.25, t0 + dur * 0.75]
        ts = sorted(set(round(max(t0, min(t1 - 0.05, t)), 2) for t in ts))
        hits = []
        for t in ts:
            nprobe += 1
            if live_at(t): hits.append(t)
        if hits:
            # bissection grossiere des bornes reelles autour des hits (pas 0.5 s)
            lo, hi = min(hits), max(hits)
            t = lo - 0.5
            while t > t0 and live_at(round(t, 2)): lo = t; t -= 0.5; nprobe += 1
            t = hi + 0.5
            while t < t1 and live_at(round(t, 2)): hi = t; t += 0.5; nprobe += 1
            flags.append(dict(kind="OFF-LIVE", scene=[round(t0, 2), round(t1, 2)],
                              bornes=[round(lo, 2), round(hi, 2)],
                              duree=round(hi - lo + 0.5, 2)))

    elif s.get("host") == "pip" and s.get("bbox") and dur >= 1.0:
        b = s["bbox"]
        bx0, by0 = b[0] * W, b[1] * H
        bx1, by1 = (b[0] + b[2]) * W, (b[1] + b[3]) * H
        worst = None
        for f in (0.3, 0.7) if dur > 8 else (0.5,):
            t = t0 + dur * f
            fr = frame(t)
            if fr is None: continue
            nprobe += 1
            for (x, y, w, h) in narr_faces(fr):
                cx, cy = x + w / 2.0, y + h / 2.0
                if not (bx0 <= cx <= bx1 and by0 <= cy <= by1): continue  # visage hors box = contenu ailleurs, pas notre sujet
                # tete = box dilatee (memes facteurs que vf_headcover, mesures o9x8)
                hx0 = cx - 0.75 * w; hx1 = cx + 0.75 * w
                hy0 = cy - 0.15 * h - 0.9 * h; hy1 = cy - 0.15 * h + 0.9 * h
                marge = min(hx0 - bx0, bx1 - hx1, hy0 - by0, by1 - hy1)
                if worst is None or marge < worst[0]:
                    worst = (marge, round(t, 1))
        if worst and worst[0] < -4:   # 4 px de bruit YuNet toleres
            flags.append(dict(kind="MARGE-TETE", scene=[round(t0, 2), round(t1, 2)],
                              marge_px=round(worst[0]), t=worst[1]))
cap.release()

vid = os.path.basename(os.path.abspath(wd)).replace("wk_b_", "")
print("%s : %d sondes, %d flag(s)" % (vid, nprobe, len(flags)))
for f in flags:
    if f["kind"] == "OFF-LIVE":
        print("  OFF-LIVE    scene %8.2f-%8.2f : narrateur LIVE ~%.2f-%.2f (%.1f s a nu)"
              % (f["scene"][0], f["scene"][1], f["bornes"][0], f["bornes"][1], f["duree"]))
    else:
        print("  MARGE-TETE  scene %8.2f-%8.2f : la box coupe la tete de %d px (t=%.1f)"
              % (f["scene"][0], f["scene"][1], -f["marge_px"], f["t"]))
if OUTJSON:
    json.dump({"video": vid, "flags": flags, "probes": nprobe}, open(OUTJSON, "w"), indent=1)
sys.exit(1 if flags else 0)
