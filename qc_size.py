#!/usr/bin/env python3
# qc_size.py [wk_dir ...] — verificateur CORPUS taille/position des avatars (hors batch).
# Reproduit le jugement visuel Boss : pour chaque scene pip, l'avatar rendu (bbox host_map)
# est compare a la PRESENCE narrateur mesuree dans la source (mouvement soutenu, meme
# ancrage que le cap de pin_render). Rapport TSV global, AUCUN re-render, AUCUNE correction.
#   - RATIO  : aire bbox avatar / aire presence (>2.2 = TROP-GRAND probable)
#   - DECAL  : distance des centres en fraction d'ecran (>0.12 = avatar au mauvais endroit)
#   - HERO%  : part de la duree en hero (>70% = suspect, verifier faux hero "libre")
# Sortie : /tmp/qc_size_report.tsv (une ligne par scene flaggee + resume par video).
import sys, os, json, glob, cv2, numpy as np

VG = "/home/boss/videogen"
RATIO_MAX = 2.2
DECAL_MAX = 0.12
HERO_PCT = 0.70

def narrator_region(cap, W, H, box, t0, t1):
    x = int(box[0]*W); y = int(box[1]*H); w = max(8, int(box[2]*W)); h = max(8, int(box[3]*H))
    x = max(0, min(W-w, x)); y = max(0, min(H-h, y))
    acc = np.zeros((H, W), np.uint8); npairs = 0
    for frac in (0.15, 0.3, 0.5, 0.7, 0.85):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*frac)*1000.0); ok1, a = cap.read()
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0+(t1-t0)*frac+0.5)*1000.0); ok2, b = cap.read()
        if not (ok1 and ok2): continue
        d = cv2.absdiff(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY))
        acc += (d > 18).astype(np.uint8); npairs += 1
    if npairs < 4: return None
    sust = acc[y:y+h, x:x+w] >= 3
    ys, xs = np.nonzero(sust)
    if len(xs) < 2000: return None
    return [(x+xs.min())/W, (y+ys.min())/H, (xs.max()-xs.min()+1)/W, (ys.max()-ys.min()+1)/H]

def check(wd, out):
    vid = os.path.basename(wd).replace("wk_b_", "")
    hm_p = os.path.join(wd, "host_map.json")
    src = os.path.join(wd, "source.mp4")
    if not (os.path.exists(hm_p) and os.path.exists(src)):
        out.write("%s\tSKIP\tpas de host_map/source\n" % vid); return
    hm = json.load(open(hm_p))
    cap = cv2.VideoCapture(src); W = int(cap.get(3)); H = int(cap.get(4))
    dur = sum(s["end"]-s["start"] for s in hm) or 1.0
    hero_d = sum(s["end"]-s["start"] for s in hm if s["host"] == "hero")
    nflag = 0
    for s in hm:
        if s["host"] != "pip" or not s.get("bbox") or s["end"]-s["start"] < 2.0: continue
        bb = s["bbox"]
        reg = narrator_region(cap, W, H, bb, s["start"], s["end"])
        if reg is None: continue
        ratio = (bb[2]*bb[3]) / max(1e-6, reg[2]*reg[3])
        dc = ((bb[0]+bb[2]/2 - (reg[0]+reg[2]/2))**2 + (bb[1]+bb[3]/2 - (reg[1]+reg[3]/2))**2) ** 0.5
        # PRESENCE-DEBORDE (verdict Boss ADJj 0:08 : oeuf vert sur la tete, corps visible) :
        # part de la presence narrateur CONTENUE dans la box. Tete couverte + corps dehors
        # = le QC identite est aveugle (pas de visage) mais la presence deborde massivement.
        ix0 = max(bb[0], reg[0]); iy0 = max(bb[1], reg[1])
        ix1 = min(bb[0]+bb[2], reg[0]+reg[2]); iy1 = min(bb[1]+bb[3], reg[1]+reg[3])
        inter = max(0.0, ix1-ix0) * max(0.0, iy1-iy0)
        contain = inter / max(1e-6, reg[2]*reg[3])
        if ratio > RATIO_MAX or dc > DECAL_MAX or contain < 0.60:
            nflag += 1
            out.write("%s\t%.1f-%.1fs\tratio=%.2f\tdecal=%.3f\tcontain=%.2f\tavatar=%s\tpresence=%s\n"
                      % (vid, s["start"], s["end"], ratio, dc, contain,
                         [round(v,3) for v in bb], [round(v,3) for v in reg]))
    hp = hero_d/dur
    verdict = []
    if nflag: verdict.append("%d scenes flaggees" % nflag)
    if hp > HERO_PCT: verdict.append("HERO %.0f%% de la duree" % (hp*100))
    out.write("%s\tRESUME\t%s\n" % (vid, " + ".join(verdict) if verdict else "OK"))
    out.flush(); cap.release()

wds = sys.argv[1:] or sorted(glob.glob(os.path.join(VG, "wk_b_*")))
rp = "/tmp/qc_size_report.tsv"
with open(rp, "w") as out:
    for wd in wds:
        check(wd, out)
print("rapport: %s" % rp)
