#!/usr/bin/env python3
# qc_geom.py <workdir> — QC GEOMETRIQUE : valide que chaque masque pip colle a la carte
# REELLE de la source (ce que le QC identite ne voit pas : slivers de carte sans visage,
# avatar trop grand, forme fausse). Mesures deterministes sur la source :
#   - bords reels (scan fond->carte au milieu des cotes, mediane 3 frames)
#   - couverture : le masque doit contenir la carte reelle (tolerance 1.5% par bord)
#   - anti-fantome : un cote en echec n'est retenu que si la bande de fuite revendiquee
#     est VIVANTE (activite temporelle inter-frames). Une carte reelle bouge ; une bande
#     figee = structure statique du fond (vignettes, gaps) prise pour un bord de carte.
#   - justesse : aire masque <= 1.6x aire carte reelle (sinon avatar geant)
#   - stabilite : segments pip adjacents meme position -> meme bbox
# Rapport JSON qc_geom.json + exit 1 si echecs. Zero VLM.
import sys, os, json, cv2, numpy as np

wd = sys.argv[1]
src = os.path.join(wd, "source.mp4")
hm = json.load(open(os.path.join(wd, "host_map.json")))
cap = cv2.VideoCapture(src)
W = int(cap.get(3)); H = int(cap.get(4))
ACT_MIN = 0.5  # sous ce niveau d'activite, la bande est consideree figee (fond)

def scan(line):
    if len(line) < 10: return 0
    ref = line[:4].mean(axis=0)
    for i in range(4, len(line)):
        if np.linalg.norm(line[i]-ref) > 35: return i
    return 0

def true_rect(box, t0, t1):
    x = int(box[0]*W); y = int(box[1]*H); w = max(8, int(box[2]*W)); h = max(8, int(box[3]*H))
    x = max(0, min(W-w, x)); y = max(0, min(H-h, y))
    res = []
    for frac in (0.3, 0.5, 0.7):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*frac)*1000.0)
        ok, img = cap.read()
        if not ok: continue
        img = img.astype('float32')
        row = img[y+h//2, x:x+w]; col = img[y:y+h, x+w//2]
        res.append((scan(row[:w//2]), scan(row[::-1][:w//2]),
                    scan(col[:h//2]), scan(col[::-1][:h//2])))
    if len(res) < 2: return None
    med = [sorted(r[i] for r in res)[len(res)//2] for i in range(4)]
    L, R, T, B = med
    nw = w-L-R; nh = h-T-B
    if nw < 16 or nh < 16: return None
    return (x+L, y+T, nw, nh)

def strip_act(b, t0, t1):
    # activite temporelle max inter-frames d'une bande ; bande trop petite ou
    # non mesurable -> 1e9 (= consideree vivante, le flag est garde par prudence)
    x0 = max(0, int(b[0])); y0 = max(0, int(b[1]))
    x1 = min(W, int(b[2])); y1 = min(H, int(b[3]))
    if x1-x0 < 3 or y1-y0 < 3: return 1e9
    fs = []
    for frac in (0.3, 0.5, 0.7):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*frac)*1000.0)
        ok, img = cap.read()
        if not ok: continue
        fs.append(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype('float32'))
    if len(fs) < 2: return 1e9
    return max(float(np.abs(a-c).mean()) for a, c in zip(fs, fs[1:]))

fails = []
pips = [s for s in hm if s["host"] == "pip" and s.get("bbox")]
for s in pips:
    bb = s["bbox"]
    mx = bb[0]*W; my = bb[1]*H; mw = bb[2]*W; mh = bb[3]*H
    # carte reelle mesuree dans une fenetre elargie autour du masque (au cas ou il sous-couvre)
    ex = [max(0.0, bb[0]-0.04), max(0.0, bb[1]-0.04),
          min(1.0-max(0.0, bb[0]-0.04), bb[2]+0.08), min(1.0-max(0.0, bb[1]-0.04), bb[3]+0.08)]
    tr = true_rect(ex, s["start"], s["end"])
    if tr is None: continue
    cx, cy, cw, ch = tr
    tol = 0.015*min(W, H)
    iy0 = max(cy, my); iy1 = min(cy+ch, my+mh)
    ix0 = max(cx, mx); ix1 = min(cx+cw, mx+mw)
    sides = []
    if cx < mx-tol: sides.append((cx, iy0, mx, iy1))
    if cy < my-tol: sides.append((ix0, cy, ix1, my))
    if cx+cw > mx+mw+tol: sides.append((mx+mw, iy0, cx+cw, iy1))
    if cy+ch > my+mh+tol: sides.append((ix0, my+mh, ix1, cy+ch))
    if sides and any(strip_act(b, s["start"], s["end"]) > ACT_MIN for b in sides):
        fails.append({"t": round((s["start"]+s["end"])/2, 1), "type": "SOUS-COUVERTURE",
                      "carte": [round(cx/W,3), round(cy/H,3), round(cw/W,3), round(ch/H,3)],
                      "masque": [round(v,3) for v in bb]})
        continue
    if mw*mh > 1.6*cw*ch:
        fails.append({"t": round((s["start"]+s["end"])/2, 1), "type": "TROP-GRAND",
                      "ratio": round((mw*mh)/(cw*ch), 2),
                      "carte": [round(cx/W,3), round(cy/H,3), round(cw/W,3), round(ch/H,3)],
                      "masque": [round(v,3) for v in bb]})

# stabilite : pips adjacents (<3s d'ecart) de centres proches -> meme bbox exigee
for i in range(1, len(pips)):
    a, b = pips[i-1], pips[i]
    if b["start"] - a["end"] > 3.0: continue
    ab, bbx = a["bbox"], b["bbox"]
    acx, acy = ab[0]+ab[2]/2, ab[1]+ab[3]/2
    bcx, bcy = bbx[0]+bbx[2]/2, bbx[1]+bbx[3]/2
    if abs(acx-bcx) < 0.08 and abs(acy-bcy) < 0.08 and ab != bbx:
        fails.append({"t": round(b["start"], 1), "type": "SAUT-DE-BOX",
                      "avant": [round(v,3) for v in ab], "apres": [round(v,3) for v in bbx]})

cap.release()
json.dump(fails, open(os.path.join(wd, "qc_geom.json"), "w"), indent=2)
if fails:
    print("QC-GEOM ECHECS : %d" % len(fails))
    for f in fails[:15]:
        print("  t=%.1fs %s" % (f["t"], f["type"]))
    sys.exit(1)
print("QC-GEOM PROPRE : masques fideles a la source")
