#!/usr/bin/env python3
# mesure_derive.py <clip.webm|dossier> ...
# Mesure la derive de chainage : saturation / luminance / nettete sur le SUJET seul
# (alpha > 128) a la 1re, au milieu et a la derniere image. Trois nombres suffisent a
# reperer une graine instable (cf graine 500 qui SUR-sature au lieu de se delaver).
import os, subprocess, sys, glob
import cv2, numpy as np

def mesure(png):
    im = cv2.imread(png, cv2.IMREAD_UNCHANGED)
    m = im[:, :, 3] > 128
    if m.sum() < 100:
        return None
    hsv = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2HSV)
    gris = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2GRAY)
    net = cv2.Laplacian(gris, cv2.CV_64F).var()
    return hsv[:, :, 1][m].mean(), hsv[:, :, 2][m].mean(), net

cibles = []
for a in sys.argv[1:]:
    cibles += sorted(glob.glob(os.path.join(a, "*.webm"))) if os.path.isdir(a) else [a]

for clip in cibles:
    tmp = "/tmp/md_%d" % os.getpid()
    subprocess.run(["rm", "-rf", tmp]); os.makedirs(tmp)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-c:v", "libvpx-vp9", "-i", clip,
                    "-vf", "format=rgba", "%s/f%%04d.png" % tmp], check=True)
    fs = sorted(os.listdir(tmp))
    idx = [0, len(fs) // 2, len(fs) - 1]
    vals = [mesure(os.path.join(tmp, fs[i])) for i in idx]
    if any(v is None for v in vals):
        print("%s : sujet introuvable" % os.path.basename(clip)); continue
    (s0, v0, n0), (s1, v1, n1), (s2, v2, n2) = vals
    ds = 100.0 * (s2 - s0) / max(1.0, s0)
    dv = 100.0 * (v2 - v0) / max(1.0, v0)
    # Seuils cales sur le corpus reel (clips 6 s, 3 maillons) : les clips GARDES perdent
    # 6 a 45 % de saturation selon la PHOTO (IMG_3602 perd ~45 % sur toutes les graines) et ~20 % de luminance. La graine 500 ecartee, elle, MONTE
    # (+43 % sat / +54 % lum). Une graine instable se voit au SIGNE, pas a l amplitude.
    etat = "OK"
    if ds > 15 or dv > 15:
        etat = "SUR-SATURE"          # cas graine 500 : a ecarter
    elif ds < -50 or dv < -35:
        etat = "DELAVE"              # au-dela du corpus valide
    print("%-42s img=%3d  sat %5.1f -> %5.1f -> %5.1f (%+.0f%%)  lum %5.1f -> %5.1f -> %5.1f (%+.0f%%)  net %4.0f -> %4.0f  %s"
          % (os.path.basename(clip), len(fs), s0, s1, s2, ds, v0, v1, v2, dv, n0, n2, etat))
    subprocess.run(["rm", "-rf", tmp])
