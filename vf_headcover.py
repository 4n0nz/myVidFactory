#!/usr/bin/env python3
# vf_headcover.py <workdir> <rendu.mp4> [--probes N] [--thr F] [--dist]
#   Le narrateur est-il VRAIMENT recouvert la ou le plan dit qu'il doit l'etre ?
#
# IDEE (Boss, 2026-08-01) : garder le narrateur en memoire et verifier qu'il est couvert.
# On l'ancre sur la SOURCE, jamais sur le rendu.
#
# POURQUOI PAS DANS LE RENDU — mesure du 2026-08-01 sur o9x8kycf3Wo t=428 :
#   SOURCE : 1 visage, cos_narrateur = 0.662
#   RENDU  : 0 visage detecte
# Le vert recouvre pile la box serree du detecteur, donc YuNet ne trouve plus rien et
# qc_ident conclut "narrateur introuvable, 0 fuite" — pendant qu'un humain voit une
# demi-tete. Un recouvrement PARTIEL qui tranche a travers le visage aveugle le juge
# precisement parce qu'il cache ce que le juge cherche. Chercher dans le rendu, c'est
# chercher ce qu'on vient d'effacer.
#
# POURQUOI LA TETE ET PAS LE VISAGE : la box YuNet s'arrete aux sourcils et au menton.
# Sur o9x8 t=428 elle vaut x=317-476 et tombe INTEGRALEMENT dans le vert (qui commence a
# 316) — j'en ai conclu a tort que le visage etait couvert. La tete, elle, va jusqu'a
# ~238 : oreille, tempe, branche de lunettes et joue restaient a nu. On dilate donc la
# box aux proportions d'une tete avant de mesurer.
#
# CE QUE CA NE FAIT PAS : la sur-couverture. Que sa tete soit verte ne dit rien sur le
# fait que le vert s'arrete au bord de la carte. Autre defaut, autre mesure — et sans
# signal universel de "carte" (mesure du 2026-08-01 : frontiere franche sur gnfHlIoh34Q,
# AUCUNE sur KKniWb9RKq4 ou la webcam est incrustee a ras dans une capture d'ecran).
#
# PAS DE CIRCULARITE : le host_map sert uniquement a savoir OU le pipeline avait
# l'INTENTION de couvrir (scenes pip et hero). Le verdict, lui, vient des pixels. On ne
# demande jamais au plan de valider la peinture — c'est exactement le bug de
# covered_by_avatar() dans qc_ident.
import sys, os, json
import cv2, numpy as np

wd, rend = sys.argv[1], sys.argv[2]
PROBES = int(sys.argv[sys.argv.index("--probes") + 1]) if "--probes" in sys.argv else 4
THR = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 0.90
DIST = "--dist" in sys.argv
VG = "/home/boss/videogen"
COS = 0.363                     # meme seuil que partout ailleurs dans le pipeline

hm = json.load(open(os.path.join(wd, "host_map.json")))
narr = np.load(os.path.join(wd, "narrator_feat.npy"))
cs = cv2.VideoCapture(os.path.join(wd, "source.mp4"))
cr = cv2.VideoCapture(rend)
W = int(cs.get(3)); H = int(cs.get(4))
yfd = cv2.FaceDetectorYN.create(VG + "/face_detection_yunet_2023mar.onnx", "", (W, H),
                                score_threshold=0.6)
rec = cv2.FaceRecognizerSF.create(VG + "/face_recognition_sface_2021dec.onnx", "")

def grab(cap, t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, f = cap.read()
    return f if ok else None

def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def head_boxes(fr):
    """Boxes TETE du narrateur dans la source. La box YuNet couvre sourcils->menton ;
    une tete avec cheveux et oreilles fait ~1.5x en largeur et ~1.8x en hauteur, centree
    un peu plus haut (les cheveux depassent vers le haut)."""
    _, fs = yfd.detect(fr)
    if fs is None: return []
    out = []
    for f in fs:
        x, y, w, h = float(f[0]), float(f[1]), float(f[2]), float(f[3])
        try:
            ft = rec.feature(rec.alignCrop(fr, f)).flatten().astype(np.float32)
        except Exception:
            continue
        c = cos(ft, narr)
        if c < COS: continue
        cx, cy = x + w / 2.0, y + h / 2.0 - 0.15 * h
        hw, hh = 1.5 * w, 1.8 * h
        out.append((max(0, int(cx - hw / 2)), max(0, int(cy - hh / 2)),
                    min(W, int(cx + hw / 2)), min(H, int(cy + hh / 2)), c))
    return out

MOTION_DT = 0.5
MOTION_MIN = 0.10
def is_live(t, box):
    """Le narrateur est-il LIVE ici, ou n'est-ce que son image dans le contenu ?
    Doctrine Boss : le narrateur present dans un plan (photo, banniere, b-roll) est du
    CONTENU, pas un pip — il ne doit PAS etre recouvert. Sans cette porte, KKniWb9RKq4
    remontait 4 fausses fuites a 0% : sa tete apparait dans la banniere Skool de sa
    propre communaute affichee a l'ecran (t=50, 62, 76.7), pendant que sa VRAIE webcam
    ronde en bas a droite est, elle, correctement recouverte.
    Meme mesure que le gate _hot de qc_ident : fraction de pixels qui bougent fort."""
    a = grab(cs, t); b = grab(cs, t + MOTION_DT)
    if a is None or b is None: return True          # dans le doute, on garde
    x0, y0, x1, y1 = box[:4]
    pa = cv2.cvtColor(a[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    pb = cv2.cvtColor(b[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if pa.shape != pb.shape or pa.size == 0: return True
    return float((cv2.absdiff(pa, pb) > 18).mean()) >= MOTION_MIN

def green_frac(fr, box):
    x0, y0, x1, y1 = box[:4]
    if x1 <= x0 or y1 <= y0: return None
    roi = fr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (45, 120, 120), (85, 255, 255))
    return float((m > 0).mean())

fracs = []; bad = []
for i, s in enumerate(hm):
    if s.get("host") not in ("pip", "hero"): continue     # off = contenu, intention nulle
    t0, t1 = s["start"], s["end"]
    if t1 - t0 < 0.4: continue
    worst = None
    for t in np.linspace(t0 + 0.15 * (t1 - t0), t0 + 0.85 * (t1 - t0), PROBES):
        fs = grab(cs, float(t)); fr = grab(cr, float(t))
        if fs is None or fr is None: continue
        for hb in head_boxes(fs):
            g = green_frac(fr, hb)
            if g is None: continue
            # Porte de mouvement seulement quand ca compte : une tete deja verte ne
            # demande aucune justification, on evite 2 decodages de frame par sonde.
            if g < THR and not is_live(float(t), hb): continue
            fracs.append(g)
            if worst is None or g < worst[0]:
                worst = (g, float(t), hb)
    if worst and worst[0] < THR:
        bad.append((i, s["host"], t0, t1, worst))

vid = os.path.basename(os.path.abspath(wd)).replace("wk_b_", "")
if DIST and fracs:
    a = sorted(fracs)
    def q(p): return a[int(p * (len(a) - 1))]
    print("%s : %d sondes tete — couverture verte  min=%.3f p05=%.3f p25=%.3f p50=%.3f p95=%.3f"
          % (vid, len(a), a[0], q(.05), q(.25), q(.5), q(.95)))

print("%s : %d scene(s) pip/hero avec tete decouverte (seuil %.2f)" % (vid, len(bad), THR))
for i, host, t0, t1, (g, t, hb) in bad:
    print("  scene[%d] %-4s %8.2f-%8.2f  pire sonde t=%.1f : tete verte a %.0f%% "
          "(tete %d,%d -> %d,%d, cos=%.3f)"
          % (i, host, t0, t1, t, 100 * g, hb[0], hb[1], hb[2], hb[3], hb[4]))
cs.release(); cr.release()
sys.exit(1 if bad else 0)
