#!/usr/bin/env python3
# pinpoint3.py <workdir> — tracking IDENTITE du narrateur. ZERO VLM dans la decision.
#
# v2 (stabilisation globale) : la geometrie derive des CLUSTERS DE POSITION calcules sur
# TOUS les samples de la video, jamais des scenes individuelles. Le bruit card_extent par
# sample fragmentait en micro-scenes d'1s aux boxes delirantes (KKni : 31 pips d'1s,
# ellipse quasi plein ecran). Ici : box canonique = percentiles 10-90 sur l'ensemble des
# cartes d'un cluster (robuste), scenes = runs d'ID de cluster (stables par nature).
#
#   1. Pass 1 (1 sample/s) : YuNet visages + SFace embedding chacun + mouvement carte.
#   2. Narrateur = plus gros cluster d'identite (talking-head).
#   3. Decision par sample : visage(s) narrateur (cos >= 0.363) + mouvement >= 0.35
#      (photo statique ~0.1, humain immobile ~0.5) -> hero si un visage/carte plein cadre,
#      sinon pip = UNION des cartes de TOUS ses visages (cam + previews de lui).
#   4. Clusters de position globaux -> box canonique percentile + edge-snap 8%.
#      Cluster > 50% de l'ecran = narrateur geant -> HERO propre (pas d'ellipse plein ecran).
#   5. Scenes = runs de cluster, trous <= 4s combles (sous-couvrir interdit), hero
#      prioritaire aux frontieres, micro-scenes absorbees.
# Sortie : host_map_pin.json (format pinpoint) -> pin_render/build_seg inchanges.
# Scenes "src":"ident" -> pin_render saute son is_hero() VLM.
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
COS_CLUST = 0.40      # assignation cluster identite
MOTION_MIN = 0.35     # photo statique ~0.1, humain IMMOBILE ~0.5 (1.2 excluait le narrateur calme)
FACE_MIN = 0.045      # visage < 4.5% H = vignette, pas la cam
EDGE = 0.08           # snap-bord : box a <8% d'un bord = cam collee -> etend au bord
QUADS = {"top-left":(0.25,0.25),"top-right":(0.75,0.25),"bottom-left":(0.25,0.75),
         "bottom-right":(0.75,0.75),"center":(0.5,0.5)}

def _frame(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0); ok, fr = cap.read()
    return fr if ok else None

def _card(fr, f):
    """(box, bounded) : bounded=True si card_extent a trouve de VRAIS bords de carte.
    False = fallback ancre-visage -> narrateur probablement LIBRE dans la scene."""
    if card_extent is not None and hasattr(card_extent, "_card_one"):
        cb = card_extent._card_one(fr, int(f[0]), int(f[1]), int(f[2]), int(f[3]))
        if cb is not None:
            x, y, w, h = cb
            return [x/W, y/H, w/W, h/H], True
    fx, fy, fw, fh = f[0]/W, f[1]/H, f[2]/W, f[3]/H
    return [max(0, fx-fw*0.6), max(0, fy-fh*0.7), min(1, fw*2.2), min(1, fh*3.0)], False

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

def pc(vals, q):
    v = sorted(vals); return v[min(len(v)-1, max(0, int(q*(len(v)-1))))]

# ---- PASS 1 : visages + embeddings + mouvement ----
samples = []
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
            card, bounded = _card(fr, f)
            mo = _motion(fr, fr2, card)
            entry["faces"].append({"f": [float(v) for v in f[:4]], "feat": feat,
                                   "card": [float(v) for v in card], "mo": mo,
                                   "bounded": bounded})
    samples.append(entry)
    t += STEP

# ---- PASS 2 : clustering identite -> narrateur = plus gros cluster ----
cents = []
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
    print("AUCUN visage dans la video — rien a couvrir")
    json.dump([], open(os.path.join(wd, "host_map_pin.json"), "w")); sys.exit(0)
narr = max(cents, key=lambda c: c["n"])
narr_feat = narr["sum"]/narr["n"]
np.save(os.path.join(wd, "narrator_feat.npy"), narr_feat)
print("clusters identite: %d | narrateur: %d/%d visages" % (len(cents), narr["n"], sum(c["n"] for c in cents)))

# ---- PASS 3 : decision par sample ----
decisions = []   # (t, "hero"|"pip"|None, box)
for s in samples:
    cands = [fc for fc in s["faces"]
             if _cos(fc["feat"], narr_feat) >= COS_SAME and fc["mo"] >= MOTION_MIN]
    # HERO = n'importe quel talking-head plein ecran qui bouge, narrateur OU PAS (invite/
    # celebrite plein cadre doit etre couvert aussi — N1r Mark Cuban intro, retour Boss)
    # hero = la CARTE couvre ~tout l'ecran (pas "gros visage" : un panneau split-screen
    # 3-bords a un gros visage mais ne doit couvrir QUE le panneau — QU-f, retour Boss).
    # Visage enorme (>0.4H) = plan serre sans carte mesurable -> hero aussi.
    def _heroish(fc):
        # hero : carte quasi plein cadre, OU visage enorme, OU narrateur LIBRE dans la
        # scene (aucun bord de carte trouve) avec un gros visage (TzJC plein cadre sans
        # overlay -> ellipse tete au lieu de hero). Panneau split-screen = bounded -> pip.
        return ((fc["card"][2] > 0.65 and fc["card"][3] > 0.85)
                or fc["f"][3]/H > 0.40
                or (not fc.get("bounded", True) and fc["f"][3]/H > 0.20))
    bigface = any(_heroish(fc) and fc["mo"] >= MOTION_MIN for fc in s["faces"])
    if not cands and not bigface:
        decisions.append((s["t"], None, None)); continue
    if bigface or any(_heroish(fc) for fc in cands):
        decisions.append((s["t"], "hero", [0.0, 0.0, 1.0, 1.0]))
    else:
        x0 = min(fc["card"][0] for fc in cands); y0 = min(fc["card"][1] for fc in cands)
        x1 = max(fc["card"][0]+fc["card"][2] for fc in cands)
        y1 = max(fc["card"][1]+fc["card"][3] for fc in cands)
        decisions.append((s["t"], "pip", [x0, y0, x1-x0, y1-y0]))

# ---- PASS 4 : clusters de POSITION globaux (stabilisation) ----
pclust = []
for t, kind, card in decisions:
    if kind != "pip": continue
    cx, cy = card[0]+card[2]/2, card[1]+card[3]/2
    hit = None
    for c in pclust:
        if abs(cx-c["cx"]) < 0.10 and abs(cy-c["cy"]) < 0.10: hit = c; break
    if hit is None:
        pclust.append({"cx": cx, "cy": cy, "cards": [card]})
    else:
        n = len(hit["cards"])
        hit["cx"] = (hit["cx"]*n+cx)/(n+1); hit["cy"] = (hit["cy"]*n+cy)/(n+1)
        hit["cards"].append(card)
for c in pclust:
    cs = c["cards"]
    x0 = pc([a[0] for a in cs], 0.05); y0 = pc([a[1] for a in cs], 0.05)
    x1 = pc([a[0]+a[2] for a in cs], 0.95); y1 = pc([a[1]+a[3] for a in cs], 0.95)
    edges = []
    if x0 < EDGE: x0 = 0.0; edges.append("L")
    if y0 < EDGE: y0 = 0.0; edges.append("T")
    if x1 > 1-EDGE: x1 = 1.0; edges.append("R")
    if y1 > 1-EDGE: y1 = 1.0; edges.append("B")
    c["box"] = [round(float(x0),4), round(float(y0),4), round(float(x1-x0),4), round(float(y1-y0),4)]
    c["edges"] = edges
    # box quasi plein ecran = narrateur geant -> HERO propre, pas d'ellipse/rect plein ecran
    c["hero"] = (c["box"][2]*c["box"][3] > 0.75)
print("clusters position (samples) : %d" % len(pclust))

def _clid(card):
    cx, cy = card[0]+card[2]/2, card[1]+card[3]/2
    best, bi = 1e9, -1
    for i, c in enumerate(pclust):
        d = (cx-c["cx"])**2 + (cy-c["cy"])**2
        if d < best: best, bi = d, i
    return bi

# ---- PASS 5 : scenes = runs de (kind, cluster) ; trous <= 4 ; hero prioritaire ----
scenes = []; cur = None; miss = 0
for t, kind, card in decisions:
    if kind == "pip" and pclust and pclust[_clid(card)]["hero"]:
        kind = "hero"
    key = ("hero", None) if kind == "hero" else (("pip", _clid(card)) if kind == "pip" else None)
    if key is None:
        if cur:
            miss += 1
            if miss > 4: scenes.append(cur); cur = None; miss = 0
        continue
    if cur and cur["key"] == key:
        cur["end"] = t + STEP; miss = 0
    else:
        start = max(0.0, t-1.0) if key[0] == "hero" else max(0.0, t-0.5)
        if cur:
            if key[0] == "hero" and cur["end"] > start: cur["end"] = start
            scenes.append(cur)
        cur = {"key": key, "start": start, "end": t + STEP}; miss = 0
if cur: scenes.append(cur)
scenes = [s for s in scenes if s["end"]-s["start"] >= 1.0]

out = []
for sc in scenes:
    if sc["key"][0] == "hero":
        out.append({"start": round(float(sc["start"]),2), "end": round(float(sc["end"]),2),
                    "region": "hero", "box": [0.0,0.0,1.0,1.0], "edges": ["L","T","R","B"],
                    "n": 0, "src": "ident"})
    else:
        c = pclust[sc["key"][1]]
        b = c["box"]; bcx, bcy = b[0]+b[2]/2, b[1]+b[3]/2
        reg = min(QUADS, key=lambda k: (QUADS[k][0]-bcx)**2 + (QUADS[k][1]-bcy)**2)
        out.append({"start": round(float(sc["start"]),2), "end": round(float(sc["end"]),2),
                    "region": reg, "box": list(b), "edges": list(c["edges"]),
                    "n": len(c["cards"]), "src": "ident"})
out.sort(key=lambda s: s["start"])

# fusion scenes adjacentes meme region+box
merged = []
for s in out:
    if merged and s["region"] == merged[-1]["region"] and s["box"] == merged[-1]["box"] \
            and s["start"] - merged[-1]["end"] <= 2.0:
        merged[-1]["end"] = s["end"]
    else:
        merged.append(s)
out = merged

# chevauchements : PRIORITE HERO (rogne le voisin), sinon clip au precedent
for i in range(1, len(out)):
    if out[i]["start"] < out[i-1]["end"]:
        if out[i]["region"] == "hero" and out[i-1]["region"] != "hero":
            out[i-1]["end"] = out[i]["start"]
        else:
            out[i]["start"] = out[i-1]["end"]
out = [s for s in out if s["end"]-s["start"] >= 0.5]

# micro-scene pip (<2s) collee a un hero = transition de zoom -> devient hero (sur-couvre)
for i in range(1, len(out)-1):
    b = out[i]
    if b["end"]-b["start"] < 2.0 and b["region"] != "hero" \
            and (out[i-1]["region"] == "hero" or out[i+1]["region"] == "hero"):
        b["region"] = "hero"; b["box"] = [0.0,0.0,1.0,1.0]; b["edges"] = ["L","T","R","B"]

# micro-scenes (<2.5s) coincees entre deux scenes de MEME box -> absorbees puis re-fusionnees
for i in range(1, len(out)-1):
    b = out[i]
    if (b["region"] != "hero" and out[i-1]["region"] != "hero"
            and out[i-1]["box"] == out[i+1]["box"] and b["box"] != out[i-1]["box"]
            and b["end"] - b["start"] < 2.5):
        b["box"] = list(out[i-1]["box"]); b["region"] = out[i-1]["region"]
        b["edges"] = list(out[i-1]["edges"])
merged2 = []
for s in out:
    if merged2 and s["region"] == merged2[-1]["region"] and s["box"] == merged2[-1]["box"] \
            and s["start"] - merged2[-1]["end"] <= 2.0:
        merged2[-1]["end"] = s["end"]
    else:
        merged2.append(s)
out = merged2

# bords temporels : une scene qui demarre dans les 5 premieres secondes s'etend a 0
# (fade-in = YuNet rate les 1res frames -> narrateur a decouvert des la seconde 0, SdMp) ;
# idem fin de video
if out and out[0]["start"] <= 5.0: out[0]["start"] = 0.0
if out and out[-1]["end"] >= DUR-5.0: out[-1]["end"] = round(float(DUR), 2)

json.dump(out, open(os.path.join(wd, "host_map_pin.json"), "w"), indent=2)
print("SCENES (%d) : start-end | duree | box | type | bords" % len(out))
for s in out:
    print("  %.1f-%.1fs | %5.1fs | [%.3f,%.3f,%.3f,%.3f] | %-12s | %s"
          % (s["start"], s["end"], s["end"]-s["start"], *s["box"], s["region"], "".join(s["edges"]) or "flottant"))
cap.release()
