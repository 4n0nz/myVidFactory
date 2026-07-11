#!/usr/bin/env python3
# Genere un dataset YOLO synthetique pour detecter le FACECAM (webcam PiP) dans une video.
# Idee : coller de vrais facecams (extraits de tes videos, box connue) sur des fonds screen-share
# varies, a des positions/tailles/coins ALEATOIRES -> labels parfaits (on sait ou on a colle).
# YOLO apprend "un facecam = tete-dans-un-cadre incruste" independamment de la position -> generalise.
#
# Usage : python build_facecam_dataset.py
# Config des sources en dur ci-dessous (source.mp4 + box facecam exacte "x,y,w,h" + host_map).
import cv2, os, json, random, glob
import numpy as np

random.seed(1234)
OUT = os.path.expanduser("~/yolo/data/facecam")
N_TRAIN = 2400
N_VAL = 300
CANVAS_W, CANVAS_H = 1280, 720   # taille des images d'entrainement (16:9)

# (source_mp4, box_facecam "x,y,w,h" natif OU "auto" (=box par-segment du host_map), host_map_json)
# "auto" = pour les videos ou la webcam BOUGE : on crop chaque frame avec la bbox de son segment pip.
SOURCES = [
    ("/home/boss/videogen/wk_masortie/source.mp4",  "1348,614,552,464", "/home/boss/videogen/wk_masortie/host_map.json"),
    ("/home/boss/videogen/wk_lastvideo/source.mp4", "1345,615,555,465", "/home/boss/videogen/wk_lastvideo/host_map.json"),
    ("/home/boss/videogen/wk_ofr/source.mp4",       "auto",             "/home/boss/videogen/wk_ofr/host_map.json"),
]

def pip_samples(host_map, W, H):
    """liste (t, (x,y,w,h)px) : milieux d'echantillons des segments pip + leur bbox."""
    out = []
    for s in json.load(open(host_map)):
        if s["host"] != "pip" or not s.get("bbox"):
            continue
        a, b = s["start"], s["end"]; fx, fy, fw, fh = s["bbox"]
        px = (int(fx*W), int(fy*H), int(fw*W), int(fh*H))
        k = max(1, int((b - a) / 3))
        for i in range(k):
            out.append((a + (b - a) * (i + 0.5) / k, px))
    return out

def grab(cap, t, fps):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, f = cap.read()
    return f if ok else None

# 1) extraire facecams (crops) + fonds (contenu sans facecam) des sources
faces, bgs = [], []
for src, box, hmap in SOURCES:
    if not os.path.exists(src) or not os.path.exists(hmap):
        print("skip (absent):", src); continue
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W = int(cap.get(3)); H = int(cap.get(4))
    if box == "auto":
        samples = pip_samples(hmap, W, H)         # box par-segment (webcam mobile)
    else:
        x, y, w, h = [int(v) for v in box.split(",")]
        samples = [(t, (x, y, w, h)) for (t, _) in pip_samples(hmap, W, H)]  # box fixe
        if not samples:  # fallback si pas de pip dans le host_map
            samples = [(i, (x, y, w, h)) for i in range(10, int(cap.get(7)/fps)-5, 12)]
    random.shuffle(samples); samples = samples[:80]
    n0 = len(faces)
    for t, (x, y, w, h) in samples:
        f = grab(cap, t, fps)
        if f is None: continue
        fc = f[max(0,y):y+h, max(0,x):x+w].copy()
        if fc.size and fc.shape[0] > 30 and fc.shape[1] > 30:
            faces.append(fc)
        bg = f[:, :max(1, x-10)].copy()           # fond = region sans le facecam
        if bg.size and bg.shape[1] > 200:
            bgs.append(cv2.resize(bg, (CANVAS_W, CANVAS_H)))
    cap.release()
    print("source %s -> +%d faces (total=%d) bgs=%d" % (os.path.basename(os.path.dirname(src)), len(faces)-n0, len(faces), len(bgs)))

if not faces or not bgs:
    raise SystemExit("pas assez de materiel (faces=%d bgs=%d)" % (len(faces), len(bgs)))

# 2) synthese : coller un facecam sur un fond, position/taille/coin aleatoires + bordure
def synth_one():
    bg = random.choice(bgs).copy()
    fc = random.choice(faces)
    # echelle : facecam = 12% a 34% de la largeur canvas
    tw = int(CANVAS_W * random.uniform(0.12, 0.34))
    ar = fc.shape[0] / fc.shape[1]
    th = int(tw * ar * random.uniform(0.85, 1.15))
    tw = max(40, min(tw, CANVAS_W - 20)); th = max(40, min(th, CANVAS_H - 20))
    fcr = cv2.resize(fc, (tw, th))
    # bordure aleatoire parfois (certains facecams en ont, d'autres non comme Ofr)
    if random.random() < 0.45:
        bcol = random.choice([(255,255,255),(230,230,230),(40,40,40),(0,0,255)])
        bt = random.randint(2, 6)
        fcr = cv2.copyMakeBorder(fcr, bt,bt,bt,bt, cv2.BORDER_CONSTANT, value=bcol)
        th, tw = fcr.shape[:2]
    # position : biais vers les coins (comme les vrais) mais parfois n'importe ou
    if random.random() < 0.8:
        px = random.choice([random.randint(0, 30), CANVAS_W - tw - random.randint(0, 30)])
        py = random.choice([random.randint(0, 30), CANVAS_H - th - random.randint(0, 30)])
    else:
        px = random.randint(0, CANVAS_W - tw); py = random.randint(0, CANVAS_H - th)
    px = max(0, min(px, CANVAS_W - tw)); py = max(0, min(py, CANVAS_H - th))
    bg[py:py+th, px:px+tw] = fcr
    # label YOLO normalise (classe 0 = facecam)
    cx = (px + tw/2) / CANVAS_W; cy = (py + th/2) / CANVAS_H
    nw = tw / CANVAS_W; nh = th / CANVAS_H
    return bg, (cx, cy, nw, nh)

for split, n in (("train", N_TRAIN), ("val", N_VAL)):
    idir = os.path.join(OUT, "images", split); ldir = os.path.join(OUT, "labels", split)
    os.makedirs(idir, exist_ok=True); os.makedirs(ldir, exist_ok=True)
    for i in range(n):
        img, (cx, cy, nw, nh) = synth_one()
        cv2.imwrite(os.path.join(idir, "s%05d.jpg" % i), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        open(os.path.join(ldir, "s%05d.txt" % i), "w").write("0 %.6f %.6f %.6f %.6f\n" % (cx, cy, nw, nh))
    print("ecrit %s: %d images" % (split, n))

# 3) data.yaml
open(os.path.join(OUT, "data.yaml"), "w").write(
    "path: %s\ntrain: images/train\nval: images/val\nnc: 1\nnames: [facecam]\n" % OUT)
print("data.yaml ecrit. faces=%d bgs=%d" % (len(faces), len(bgs)))
print("DATASET_DONE")
