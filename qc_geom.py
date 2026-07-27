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
    # fenetre adaptative : si la carte mesuree COLLE a un bord de fenetre (<2% cadre),
    # la fenetre demarre DANS la carte (eglV t=33 : masque [0.3,0,0.47,1.0] mais carte
    # reelle x=0.075 -> la reference scan() = contenu de carte, pas le fond, mesure
    # tronquee au bord de fenetre -> SOUS-COUVERTURE sterile 3 tours, l'union ne
    # grandit que de la fenetre). On pousse le cote colle de 0.12 et on re-mesure
    # (<=4 fois) jusqu'a decoller ou atteindre l'ecran. Fenetres deja au bord ecran
    # (pips de coin 4D7) : cote non poussable, comportement inchange.
    bx = list(box)
    rect = None
    for _ in range(4):
        x = int(bx[0]*W); y = int(bx[1]*H); w = max(8, int(bx[2]*W)); h = max(8, int(bx[3]*H))
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
        rect = (x+L, y+T, nw, nh)
        gl = 0.12 if (L < 0.02*W and x > 0) else 0.0
        gr = 0.12 if (R < 0.02*W and x+w < W) else 0.0
        gt = 0.12 if (T < 0.02*H and y > 0) else 0.0
        gb = 0.12 if (B < 0.02*H and y+h < H) else 0.0
        if not (gl or gr or gt or gb): return rect
        nx = max(0.0, bx[0]-gl); ny = max(0.0, bx[1]-gt)
        bx = [nx, ny,
              min(1.0-nx, bx[2]+(bx[0]-nx)+gr), min(1.0-ny, bx[3]+(bx[1]-ny)+gb)]
    return rect

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

# FAUX-HERO : les scenes hero n'avaient AUCUN controle (trou de juge — verdict Boss
# 4D7 : 5 scenes hero alors que la source montre contenu + pip normal). Un vrai
# talking-head plein cadre est VIVANT au centre ; une page + pip de coin n'a
# d'activite que dans 1-2 cellules de bord. Grille 4x4 sur diff inter-frames.
# mediane par cellule sur 5 paires rapprochees : le scroll de contenu est PONCTUEL
# (il allume tout l'ecran sur 1-2 paires), le pip est vivant en continu — la mediane
# tue le scroll, sinon le juge rate exactement les scenes que le scroll a fait
# classer hero (4D7 : 3 rates sur 5 avec 3 frames espacees).
for s in [s for s in hm if s["host"] == "hero"]:
    acts = []
    for frac in (0.15, 0.3, 0.5, 0.7, 0.85):
        t = s["start"]+(s["end"]-s["start"])*frac
        pair = []
        for dt in (0.0, 0.3):
            cap.set(cv2.CAP_PROP_POS_MSEC, (t+dt)*1000.0)
            ok, img = cap.read()
            if not ok: break
            pair.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32'))
        if len(pair) < 2: continue
        d = np.abs(pair[0]-pair[1])
        gh, gw = d.shape[0]//4, d.shape[1]//4
        acts.append([[float(d[r*gh:(r+1)*gh, c*gw:(c+1)*gw].mean()) for c in range(4)]
                     for r in range(4)])
    if len(acts) < 3: continue
    med = [[sorted(a[r][c] for a in acts)[len(acts)//2] for c in range(4)] for r in range(4)]
    cells = [(r, c) for r in range(4) for c in range(4) if med[r][c] > 1.0]
    if len(cells) <= 2 and not any(r in (1, 2) and c in (1, 2) for r, c in cells):
        fails.append({"t": round((s["start"]+s["end"])/2, 1), "type": "FAUX-HERO",
                      "actives": len(cells)})

cap.release()
json.dump(fails, open(os.path.join(wd, "qc_geom.json"), "w"), indent=2)
if fails:
    print("QC-GEOM ECHECS : %d" % len(fails))
    for f in fails[:15]:
        print("  t=%.1fs %s" % (f["t"], f["type"]))
    sys.exit(1)
print("QC-GEOM PROPRE : masques fideles a la source")
