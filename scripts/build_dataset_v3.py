#!/usr/bin/env python3
# Dataset synthetique YOLO v3 : lit TOUS les crops facecam harvestes (~/yolo/data/crops/*/)
# + fonds (~/yolo/data/bgs/*/), colle a positions/tailles/coins/bordures aleatoires.
import cv2, os, random, glob
import numpy as np
random.seed(7)
OUT = os.path.expanduser("~/yolo/data/facecam")
N_TRAIN, N_VAL = 3200, 400
CW, CH = 1280, 720

def load(paths):
    out = []
    for p in paths:
        im = cv2.imread(p)
        if im is not None and im.shape[0] > 30 and im.shape[1] > 30:
            out.append(im)
    return out

faces = load(glob.glob(os.path.expanduser("~/yolo/data/crops/*/*.jpg")))
bgraw = load(glob.glob(os.path.expanduser("~/yolo/data/bgs/*/*.jpg")))
bgs = [cv2.resize(b, (CW, CH)) for b in bgraw]
print("faces=%d bgs=%d (personnes=%d)" % (len(faces), len(bgs),
      len(glob.glob(os.path.expanduser("~/yolo/data/crops/*/")))))
if len(faces) < 20 or len(bgs) < 20:
    raise SystemExit("pas assez de materiel")

def synth_one():
    bg = random.choice(bgs).copy(); fc = random.choice(faces)
    tw = int(CW * random.uniform(0.10, 0.36)); ar = fc.shape[0]/fc.shape[1]
    th = int(tw * ar * random.uniform(0.85, 1.15))
    tw = max(40, min(tw, CW-20)); th = max(40, min(th, CH-20))
    fcr = cv2.resize(fc, (tw, th))
    if random.random() < 0.45:   # bordure parfois (certains facecams en ont)
        bcol = random.choice([(255,255,255),(230,230,230),(30,30,30),(0,0,255),(200,200,200)])
        bt = random.randint(2, 6)
        fcr = cv2.copyMakeBorder(fcr, bt,bt,bt,bt, cv2.BORDER_CONSTANT, value=bcol); th, tw = fcr.shape[:2]
    if random.random() < 0.8:    # biais coins (comme les vrais facecams)
        px = random.choice([random.randint(0,30), CW-tw-random.randint(0,30)])
        py = random.choice([random.randint(0,30), CH-th-random.randint(0,30)])
    else:
        px = random.randint(0, CW-tw); py = random.randint(0, CH-th)
    px = max(0, min(px, CW-tw)); py = max(0, min(py, CH-th))
    bg[py:py+th, px:px+tw] = fcr
    cx = (px+tw/2)/CW; cy = (py+th/2)/CH
    return bg, (cx, cy, tw/CW, th/CH)

for split, n in (("train", N_TRAIN), ("val", N_VAL)):
    idir = os.path.join(OUT, "images", split); ldir = os.path.join(OUT, "labels", split)
    for d in (idir, ldir):
        os.makedirs(d, exist_ok=True)
        for old in glob.glob(os.path.join(d, "*")): os.remove(old)
    for i in range(n):
        img, (cx, cy, nw, nh) = synth_one()
        cv2.imwrite(os.path.join(idir, "s%05d.jpg" % i), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        open(os.path.join(ldir, "s%05d.txt" % i), "w").write("0 %.6f %.6f %.6f %.6f\n" % (cx, cy, nw, nh))
    print("ecrit %s: %d" % (split, n))
open(os.path.join(OUT, "data.yaml"), "w").write(
    "path: %s\ntrain: images/train\nval: images/val\nnc: 1\nnames: [facecam]\n" % OUT)
print("DATASET_V3_DONE")
