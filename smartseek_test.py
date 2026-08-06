#!/usr/bin/env python3
# smartseek_test.py <video> — preuve d'equivalence SmartCap vs seek classique.
# Rejoue le pattern d'acces reel de la chaine (fracs par scene, ordre croissant,
# quelques retours arriere) et compare chaque frame BITWISE.
import sys, cv2, numpy as np
from smartseek import SmartCap

path = sys.argv[1]
ref = cv2.VideoCapture(path)
DUR = ref.get(7) / (ref.get(5) or 30)

# pattern type pin_render/qc_geom : scenes de ~8 s balayees a 0.3/0.5/0.7 + paires
# +0.5 s, plus 3 sauts arriere et 2 sauts lointains
ts = []
t0 = 0.0
while t0 + 8 < DUR:
    for frac in (0.3, 0.5, 0.7):
        ts.append(t0 + 8 * frac)
        ts.append(t0 + 8 * frac + 0.5)
    t0 += 8.0
if len(ts) > 120: ts = ts[:60] + ts[-60:]
if DUR > 30:
    ts.insert(20, 2.0)                # saut arriere
    ts.insert(40, max(0.5, DUR - 5))  # saut lointain avant
    ts.insert(41, 10.0)               # gros retour
sc = SmartCap(path)
bad = 0
for i, t in enumerate(ts):
    ref.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0); okr, fr = ref.read()
    sc.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0); oks, fs = sc.read()
    if okr != oks or (okr and not np.array_equal(fr, fs)):
        bad += 1
        print("MISMATCH t=%.3f okr=%s oks=%s" % (t, okr, oks))
print("%s : %d acces, %d mismatch -> %s" % (path.split('/')[-1], len(ts), bad,
      "EQUIVALENT" if bad == 0 else "DIVERGENT"))
sys.exit(1 if bad else 0)
