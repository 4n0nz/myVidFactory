#!/usr/bin/env python3
# pinpoint3.py <workdir> — tracking IDENTITE du narrateur. ZERO VLM dans la decision.
#
# Pourquoi v3 : pin2..pin5 = chaque decision passait par un VLM (region/fullface/crop-verify)
# et chaque probe VLM flake ~5-10% -> fuites au hasard, rustine sur rustine. Ici :
#   1. Pass 1 (1 sample/s) : YuNet detecte les visages, SFace fait l'embedding de chacun.
#   2. Le NARRATEUR = le plus gros cluster d'identite de toute la video (talking-head).
#   3. Decision par sample, deterministe : visage narrateur (cosine >= seuil standard 0.363)
#      + MOUVEMENT dans la carte (photo/vignette statique du narrateur exclue, cas 876s)
#      -> hero si plein cadre, sinon pip a la carte card_extent.
#   4. Scenes = runs bridge (trous <= 3s combles — sous-couvrir interdit, regle Boss),
#      clustering position -> box canonique enveloppe (pip statique), edge-snap.
# Sortie : host_map_pin.json meme format que pinpoint v2 -> pin_render/build_seg inchanges.
# Scenes marquees "src":"ident" -> pin_render saute son is_hero() VLM (decide ici).
import sys, os, json, cv2, numpy as np

sys.path.insert(0, "/home/boss/videogen/agent_yt")
sys.path.insert(0, "/home/boss/yolo/scripts")
try: import card_extent
except Exception: card_extent = None

wd = sys.argv[1]
src = os.path.join(wd, "source.mp4")
VG = "/home/boss/videogen"
cap = cv2.VideoCapture(src)
W = int(cap.get(3)); H = int(cap.get(4)); FPS = cap.get(5) or 30; DUR = cap.get(7)/FPS
yfd = cv2.FaceDetectorYN.create(VG+"/face_detection_yunet_2023mar.onnx", "", (W, H), score_threshold=0.6)
rec = cv2.FaceRecognizerSF.create(VG+"/face_recognition_sface_2021dec.onnx", "")

STEP = 1.0
COS_SAME = 0.363      # seuil standard SFace meme personne
COS_CLUST = 0.40      # assignation cluster (plus strict que SAME)
MOTION_MIN = 0.35     # diff moyenne grayscale 0.5s ; photo statique ~0.1, humain IMMOBILE ~0.5
                      # (1.2 excluait le narrateur assis tranquille -> trou 496-500s pin6)
FACE_MIN = 0.045      # visage < 4.5% H = trop petit pour etre la cam (vignettes)
EDGE = 0.025
QUADS = {"top-left":(0.25,0.25),"top-right":(0.75,0.25),"bottom-left":(0.25,0.75),
         "bottom-right":(0.75,0.75),"center":(0.5,0.5)}

def _frame(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0); ok, fr = cap.read()
    return fr if ok else None

def _card(fr, f):
    if card_extent is not None and hasattr(card_extent, "_card_one"):
        cb = card_extent._card_one(fr, int(f[0]), int(f[1]), int(f[2]), int(f[3]))
        if cb is not None:
            x, y, w, h = cb
            return [x/W, y/H, w/W, h/H]
    fx, fy, fw, fh = f[0]/W, f[1]/H, f[2]/W, f[3]/H
    return [max(0, fx-fw*0.6), max(0, fy-fh*0.7), min(1, fw*2.2), min(1, fh*3.0)]

def _motion(fa, fb, box):
    if fa is None or fb is None: return 99.0
    x = int(box[0]*W); y = int(box[1]*H)
    w = max(4, int(box[2]*W)); h = max(4, int(box[3]*H))
    x = max(0, min(W-w, x)); y = max(0, min(H-h, y))
    a = cv2.cvtColor(fa[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(fb[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(a, b).mean())

def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

# ---- PASS 1 : visages + embeddings sur toute la video ----
samples = []   # (t, frame_pas_garde) -> liste de (facebox, feat)
t = 0.0
while t < DUR:
    fr = _frame(t)
    if fr is None: t += STEP; continue
    fr2 = _frame(min(t+0.5, DUR-0.05))
    _, faces = yfd.detect(fr)
    entry = {"t": round(t, 2), "faces": []}
    if faces is not None:
        for f in faces:
            if f[3]/H < FACE_MIN: continue
            try:
                crop = rec.alignCrop(fr, f)
                feat = rec.feature(crop).flatten().astype(np.float32)
            except Exception:
                continue
            card = _card(fr, f)
            mo = _motion(fr, fr2, card)
            entry["faces"].append({"f": [float(v) for v in f[:4]], "feat": feat,
                                   "card": [float(v) for v in card], "mo": mo})
    samples.append(entry)
    t += STEP

# ---- PASS 2 : clustering identite -> narrateur = plus gros cluster ----
cents = []   # {"sum": vec, "n": int}
for s in samples:
    for fc in s["faces"]:
        best, bi = -1.0, -1
        for i, c in enumerate(cents):
            cv_ = _cos(fc["feat"], c["sum"]/c["n"])
            if cv_ > best: best, bi = cv_, i
        if best >= COS_CLUST:
            cents[bi]["sum"] += fc["feat"]; cents[bi]["n"] += 1
        else:
            cents.append({"sum": fc["feat"].copy(), "n": 1})
if not cents:
    print("AUCUN visage dans la video — rien a couvrir"); json.dump([], open(os.path.join(wd,"host_map_pin.json"),"w")); sys.exit(0)
narr = max(cents, key=lambda c: c["n"])
narr_feat = narr["sum"]/narr["n"]
np.save(os.path.join(wd, "narrator_feat.npy"), narr_feat)
print("clusters identite: %d | narrateur: %d/%d visages" % (len(cents), narr["n"], sum(c["n"] for c in cents)))

# ---- PASS 3 : decision par sample ----
# hero si carte quasi plein cadre OU visage tres gros ; pip sinon ; None si pas de narrateur vivant
decisions = []   # (t, "hero"|"pip"|None, box)
for s in samples:
    cands = [fc for fc in s["faces"]
             if _cos(fc["feat"], narr_feat) >= COS_SAME and fc["mo"] >= MOTION_MIN]
    if not cands:
        decisions.append((s["t"], None, None)); continue
    # TOUS ses visages comptent : cam + previews video de lui sur la meme frame (fuite 872-886s
    # pin6 ou seul le plus gros etait couvert) -> box = UNION des cartes. Hero si un seul est gros.
    if any((fc["card"][2] > 0.5 and fc["card"][3] > 0.7) or fc["f"][3]/H > 0.25 for fc in cands):
        decisions.append((s["t"], "hero", [0.0, 0.0, 1.0, 1.0]))
    else:
        x0 = min(fc["card"][0] for fc in cands); y0 = min(fc["card"][1] for fc in cands)
        x1 = max(fc["card"][0]+fc["card"][2] for fc in cands)
        y1 = max(fc["card"][1]+fc["card"][3] for fc in cands)
        decisions.append((s["t"], "pip", [x0, y0, x1-x0, y1-y0]))

# ---- PASS 4 : runs -> scenes (bridge trous <= 3 samples : YuNet cligne, sous-couvrir interdit)
def _pos_close(a, b):
    return abs(a[0]+a[2]/2-(b[0]+b[2]/2)) < 0.09 and abs(a[1]+a[3]/2-(b[1]+b[3]/2)) < 0.09
scenes = []; cur = None; miss = 0
for t, kind, card in decisions:
    if kind is None:
        if cur:
            miss += 1
            if miss > 4: scenes.append(cur); cur = None; miss = 0
        continue
    if cur and cur["kind"] == kind and (kind == "hero" or _pos_close(cur["cards"][-1], card)):
        cur["end"] = t + STEP; cur["cards"].append(card); miss = 0
    else:
        # PRIORITE HERO aux frontieres : le hero demarre 1s AVANT son 1er sample et rogne la
        # scene precedente (fuite 769.5s pin6 : la scene pip mordait sur le plein ecran)
        start = max(0.0, t-1.0) if kind == "hero" else max(0.0, t-0.5)
        if cur:
            if kind == "hero" and cur["end"] > start: cur["end"] = start
            scenes.append(cur)
        cur = {"kind": kind, "start": start, "end": t + STEP, "cards": [card]}; miss = 0
if cur: scenes.append(cur)
scenes = [s for s in scenes if s["end"]-s["start"] >= 1.0]

# ---- PASS 5 : box par scene = enveloppe percentile 10-90 des cartes, puis clustering position
def pc(vals, q):
    v = sorted(vals); return v[min(len(v)-1, max(0, int(q*(len(v)-1))))]
out = []
for sc in scenes:
    if sc["kind"] == "hero":
        out.append({"start": round(float(sc["start"]),2), "end": round(float(sc["end"]),2),
                    "region": "hero", "box": [0.0,0.0,1.0,1.0], "edges": ["L","T","R","B"],
                    "n": len(sc["cards"]), "src": "ident"})
        continue
    cs = sc["cards"]
    x0 = pc([c[0] for c in cs], 0.10); y0 = pc([c[1] for c in cs], 0.10)
    x1 = pc([c[0]+c[2] for c in cs], 0.90); y1 = pc([c[1]+c[3] for c in cs], 0.90)
    cx, cy = (x0+x1)/2, (y0+y1)/2
    reg = min(QUADS, key=lambda k: (QUADS[k][0]-cx)**2 + (QUADS[k][1]-cy)**2)
    out.append({"start": round(float(sc["start"]),2), "end": round(float(sc["end"]),2),
                "region": reg, "box": [round(float(x0),4), round(float(y0),4),
                round(float(x1-x0),4), round(float(y1-y0),4)], "edges": [],
                "n": len(cs), "src": "ident"})

# clustering position (regle Boss : pip statique -> UNE box canonique enveloppe par position)
clusters = []
for s in out:
    if s["region"] == "hero": continue
    b = s["box"]; cx, cy = b[0]+b[2]/2, b[1]+b[3]/2
    hit = None
    for c in clusters:
        bb = c["box"]; ccx, ccy = bb[0]+bb[2]/2, bb[1]+bb[3]/2
        if abs(cx-ccx) < 0.09 and abs(cy-ccy) < 0.09: hit = c; break
    if hit is None:
        clusters.append({"box": list(b), "members": [s]})
    else:
        bb = hit["box"]
        x0 = min(bb[0], b[0]); y0 = min(bb[1], b[1])
        x1 = max(bb[0]+bb[2], b[0]+b[2]); y1 = max(bb[1]+bb[3], b[1]+b[3])
        hit["box"] = [x0, y0, x1-x0, y1-y0]; hit["members"].append(s)
for c in clusters:
    x0, y0 = c["box"][0], c["box"][1]; x1, y1 = x0+c["box"][2], y0+c["box"][3]
    edges = []
    if x0 < EDGE: x0 = 0.0; edges.append("L")
    if y0 < EDGE: y0 = 0.0; edges.append("T")
    if x1 > 1-EDGE: x1 = 1.0; edges.append("R")
    if y1 > 1-EDGE: y1 = 1.0; edges.append("B")
    cb = [round(float(x0),4), round(float(y0),4), round(float(x1-x0),4), round(float(y1-y0),4)]
    for s in c["members"]:
        s["box"] = list(cb); s["edges"] = edges

# fusion scenes adjacentes meme kind+box (apres canonisation elles sont identiques)
out.sort(key=lambda s: s["start"])
merged = []
for s in out:
    if merged and s["region"] == merged[-1]["region"] and s["box"] == merged[-1]["box"] \
            and s["start"] - merged[-1]["end"] <= 2.0:
        merged[-1]["end"] = s["end"]; merged[-1]["n"] += s["n"]
    else:
        merged.append(s)
out = merged
# chevauchements (bridge/start-0.5) : PRIORITE HERO (rogne le voisin), sinon clip au precedent
for i in range(1, len(out)):
    if out[i]["start"] < out[i-1]["end"]:
        if out[i]["region"] == "hero" and out[i-1]["region"] != "hero":
            out[i-1]["end"] = out[i]["start"]
        else:
            out[i]["start"] = out[i-1]["end"]
out = [s for s in out if s["end"]-s["start"] >= 0.5]

json.dump(out, open(os.path.join(wd, "host_map_pin.json"), "w"), indent=2)
print("clusters position : %d" % len(clusters))
print("SCENES (%d) : start-end | duree | box | type | bords" % len(out))
for s in out:
    print("  %.1f-%.1fs | %5.1fs | [%.3f,%.3f,%.3f,%.3f] | %-12s | %s"
          % (s["start"], s["end"], s["end"]-s["start"], *s["box"], s["region"], "".join(s["edges"]) or "flottant"))
cap.release()
