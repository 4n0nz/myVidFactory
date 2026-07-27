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
# Decider qu'une CARTE du narrateur existe demande bien plus que le seuil de fuite :
# le narrateur apparait aussi DANS le contenu b-roll (photo de groupe, plan d'atelier)
# et y matche faiblement. Mesures eglV 1395s : pip live cos 0.80-0.93 a chaque sonde
# (6/8.3/43.5/60/200/500/800/1000/1200/1300/1380) ; visage du narrateur DANS une photo
# b-roll cos 0.55-0.69 (29.5-38.5). Couvrir ces derniers = vert par-dessus du contenu
# qui doit rester INTACT (verdict Boss 27/07 07h50). Seuil a mi-chemin des deux nuages.
COS_PIP = 0.75        # identite MEDIANE exigee d'un cluster pour etre une vraie carte
COS_CLUST = 0.40      # assignation cluster identite
SIZE_SAME = 0.55      # rapport de taille min pour que deux cartes soient le MEME layout
# Un LAYOUT, c'est une position ET une taille. Regrouper sur le seul centre laisse un
# sample parasite fonder un cluster qu'un vrai layout adopte ensuite : eglV t=33 (visage
# du narrateur dans une photo de groupe, carte 0.108x0.296, cos 0.687) a cree le cluster
# de centre (0.47,0.53), que les 21 samples du narrateur LIVE plein cadre (carte
# 0.35x1.00, cos 0.90) ont rejoint. Le parasite heritait alors du cos MEDIAN 0.901 du
# cluster, echappait au filtre FANTOME et devenait une scene pip -> promue hero plein
# cadre -> vert sur la photo de groupe, qui doit rester INTACTE (verdict Boss 27/07
# 07h50). Tolerance large (+/-45%) : le bruit de taille d'un vrai pip est de quelques %,
# l'ecart parasite/layout est d'un facteur 3.
MOTION_MIN = 0.35     # photo statique ~0.1, humain IMMOBILE ~0.5 (1.2 excluait le narrateur calme)
FACE_MIN = 0.045      # visage < 4.5% H = vignette, pas la cam
EDGE = 0.03           # snap-bord reduit : les pips gardent une marge design 2-5%, 8% collait tout aux bords (verdict Boss)
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
pip_cos = {}     # t -> meilleur cos narrateur du sample (juge d'identite du cluster)
_prev_big = []   # centres des visages a carte plein cadre du sample precedent
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
        # LIBRE exige aussi un visage au CENTRE horizontal : un vrai narrateur plein
        # cadre est cadre central (TzJC cx=0.51), un gros pip de coin dont card_extent
        # rate les bords ne l'est pas (O58 cx=0.88, faceH=0.22 -> faux hero 80-286s,
        # avatar plein ecran sur la page). Mesure : pips de coin cx~0.89.
        cx = (fc["f"][0]+fc["f"][2]/2)/W
        # LIBRE exige aussi que la carte MESUREE autour du visage soit grande : 4D7 a
        # le visage du narrateur DANS le contenu (embed vertical w=0.23 a cx=0.33,
        # faceH=0.30, bounded=False) -> 3 scenes pip classees hero. Un vrai narrateur
        # libre n'a pas de petit panneau mesurable autour de lui ; un visage dont la
        # carte fait <0.5 d'un cote vit dans un panneau de contenu -> pas hero.
        return (fc["f"][3]/H > 0.40
                or (not fc.get("bounded", True) and fc["f"][3]/H > 0.20
                    and 0.25 < cx < 0.75
                    and fc["card"][2] > 0.5 and fc["card"][3] > 0.5))
    # carte plein cadre : traite A PART, avec DEUX gates (4D7 : scroll de grille de
    # vignettes -> card_extent gonfle a [0,0,1,1] -> 3 scenes pip classees hero) :
    #   1. visage central (vrai talking-head plein cadre est cadre central, TzJC
    #      cx=0.51 ; pip de coin cx~0.87 exclu) ;
    #   2. PERSISTANCE POSITIONNELLE : meme visage, meme place, 2 samples consecutifs.
    #      Une vignette de contenu traversee par un scroll a carte plein cadre + visage
    #      central mais SE DEPLACE puis disparait ; un talking-head reste en place.
    #   3. VISAGE DOMINANT : un talking-head plein cadre REMPLIT le cadre. Un plan
    #      d'evenement (atelier filme, photo de groupe) a lui aussi une carte plein
    #      cadre, un visage central et stable pendant le panoramique — sans plancher
    #      de taille il devenait hero et on peignait du vert plein ecran sur du
    #      contenu qui doit rester INTACT (verdict Boss 27/07 07h50, eglV 31.5-34 et
    #      38-38.5). Mesures eglV : vrais heros h=0.376-0.491 (0-3, 17-28, 906-918,
    #      1387, live 39-42) contre b-roll d'evenement h=0.058-0.192 — seuil 0.25
    #      entre les deux nuages, aucun ne s'en approche.
    def _bigcard(fc):
        cx = (fc["f"][0]+fc["f"][2]/2)/W
        return (fc["card"][2] > 0.85 and fc["card"][3] > 0.85 and 0.25 < cx < 0.75
                and fc["f"][3]/H > 0.25)
    _cur_big = [((fc["f"][0]+fc["f"][2]/2)/W, (fc["f"][1]+fc["f"][3]/2)/H)
                for fc in s["faces"] if _bigcard(fc) and fc["mo"] >= MOTION_MIN]
    _stable_big = any(abs(cx-px) < 0.05 and abs(cy-py) < 0.05
                      for cx, cy in _cur_big for px, py in _prev_big)
    _prev_big = _cur_big
    bigface = _stable_big or any(_heroish(fc) and fc["mo"] >= MOTION_MIN for fc in s["faces"])
    if not cands and not bigface:
        decisions.append((s["t"], None, None)); continue
    if bigface or any(_heroish(fc) for fc in cands):
        decisions.append((s["t"], "hero", [0.0, 0.0, 1.0, 1.0]))
    else:
        x0 = min(fc["card"][0] for fc in cands); y0 = min(fc["card"][1] for fc in cands)
        x1 = max(fc["card"][0]+fc["card"][2] for fc in cands)
        y1 = max(fc["card"][1]+fc["card"][3] for fc in cands)
        pip_cos[s["t"]] = max(_cos(fc["feat"], narr_feat) for fc in cands)
        decisions.append((s["t"], "pip", [x0, y0, x1-x0, y1-y0]))

# ---- PASS 4 : clusters de POSITION + TAILLE globaux (stabilisation) ----
def _same_size(a, b):
    return min(a, b) / max(a, b, 1e-6) > SIZE_SAME

pclust = []
for t, kind, card in decisions:
    if kind != "pip": continue
    cx, cy = card[0]+card[2]/2, card[1]+card[3]/2
    hit = None
    for c in pclust:
        if (abs(cx-c["cx"]) < 0.10 and abs(cy-c["cy"]) < 0.10
                and _same_size(card[2], c["cw"]) and _same_size(card[3], c["ch"])):
            hit = c; break
    if hit is None:
        pclust.append({"cx": cx, "cy": cy, "cw": card[2], "ch": card[3],
                       "cards": [card], "cos": [pip_cos.get(t, 0.0)]})
    else:
        n = len(hit["cards"])
        hit["cx"] = (hit["cx"]*n+cx)/(n+1); hit["cy"] = (hit["cy"]*n+cy)/(n+1)
        hit["cw"] = (hit["cw"]*n+card[2])/(n+1); hit["ch"] = (hit["ch"]*n+card[3])/(n+1)
        hit["cards"].append(card); hit["cos"].append(pip_cos.get(t, 0.0))
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
    c["hero"] = (c["box"][2]*c["box"][3] > 0.85)
    # FANTOME : le narrateur apparait aussi DANS le contenu (photo de groupe, plan
    # d'atelier) ; son visage y matche, card_extent gonfle a la taille du visuel et on
    # peint du vert sur des images qui doivent rester INTACTES (verdict Boss 27/07 07h50).
    # Discriminant mesure : la carte pip vraie donne un cos MEDIAN eleve sur toute la
    # video (eglV 0.87, 4D7 0.86) ; le narrateur noye dans un visuel donne 0.55-0.71
    # (eglV 29.5-38.5). Median et non min : un pip legitime plonge quand il se detourne
    # (4D7 39-45 : cos 0.48, aucun visage detecte 40-43) — le filtrer par sample
    # fabriquerait un trou de sous-couverture.
    _cs = sorted(c["cos"])
    c["cosmed"] = _cs[len(_cs)//2] if _cs else 0.0
    c["ghost"] = c["cosmed"] < COS_PIP
print("clusters position (samples) : %d" % len(pclust))
for i, c in enumerate(pclust):
    print("  cluster %d : n=%d cos_median=%.3f box=[%.4f,%.4f,%.4f,%.4f]%s"
          % (i, len(c["cards"]), c["cosmed"], *c["box"], "  FANTOME (ignore)" if c["ghost"] else ""))

def _clid(card):
    # meme critere qu'a la construction, sinon un sample bati dans le cluster A serait
    # relu dans le cluster B (plus proche en centre) et recupererait son verdict FANTOME.
    cx, cy = card[0]+card[2]/2, card[1]+card[3]/2
    best, bi = 1e9, -1
    for i, c in enumerate(pclust):
        if not (_same_size(card[2], c["cw"]) and _same_size(card[3], c["ch"])): continue
        d = (cx-c["cx"])**2 + (cy-c["cy"])**2
        if d < best: best, bi = d, i
    if bi >= 0: return bi
    for i, c in enumerate(pclust):
        d = (cx-c["cx"])**2 + (cy-c["cy"])**2
        if d < best: best, bi = d, i
    return bi

# ---- PASS 5 : scenes = runs de (kind, cluster) ; trous <= 4 ; hero prioritaire ----
scenes = []; cur = None; miss = 0
for t, kind, card in decisions:
    if kind == "pip" and pclust and pclust[_clid(card)]["ghost"]:
        kind = None; card = None
    elif kind == "pip" and pclust and pclust[_clid(card)]["hero"]:
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

# micro-scene pip collee a un hero = transition/bruit -> devient hero (sur-couvre) :
# <2s avec UN voisin hero, ou <=5s en SANDWICH hero-hero (XzEg : card_extent accroche le
# decor par intermittence pendant un plan plein cadre continu -> flicker hero/pip 3s)
for i in range(1, len(out)-1):
    b = out[i]
    if b["region"] == "hero": continue
    dur = b["end"]-b["start"]
    prev_h = out[i-1]["region"] == "hero"; next_h = out[i+1]["region"] == "hero"
    if (dur < 2.0 and (prev_h or next_h)) or (dur <= 5.0 and prev_h and next_h):
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

# ---- PASS 6 : verite-carte aux frontieres de scene (verdict Boss 27/07 07h25) ----
# un vert sans carte dessous = meme gravite qu'une carte sans vert. Deux trous :
# (1) les runs STEP=1.0 pontent <=4 samples manques : la carte eglV disparait
#     ~6.3-7.7 (contenu seul, sonde 27/07) mais la scene 5.5-9 reste verte a 7.2 ;
# (2) le clip "au precedent" des chevauchements colle le start pip a la fin du
#     hero alors que la carte source est DEJA revenue sous lui (eglV : carte des
#     42.75, hero jusqu'a 43). Sonde fine 0.25s aux bords (3s) de chaque scene pip,
#     + 2.5s dans le voisin hero : presence = visage cos>=COS_SAME dans la box
#     (marge 6%). Trim start a la premiere presence / end a la derniere ; absence
#     >=~1s ENCADREE de presences -> scission de la scene ; hero voisin rogne quand
#     la carte est prouvee sous lui. 4 sondes consecutives absentes = carte vraiment
#     partie (YuNet ne rate jamais autant une carte statique : 8-10s = 100% hits).
#     Sonde illisible (NOFRAME) = presence : on sur-couvre, jamais l'inverse.
FINE = 0.25

def _card_at(t, box):
    fr = _frame(t)
    if fr is None: return True
    _, fs = yfd.detect(fr)
    if fs is None: return False
    x0 = (box[0]-0.06)*W; y0 = (box[1]-0.06)*H
    x1 = (box[0]+box[2]+0.06)*W; y1 = (box[1]+box[3]+0.06)*H
    for f in fs:
        cx, cy = f[0]+f[2]/2.0, f[1]+f[3]/2.0
        if not (x0 <= cx <= x1 and y0 <= cy <= y1): continue
        try:
            ft = rec.feature(rec.alignCrop(fr, f)).flatten().astype(np.float32)
        except Exception:
            continue
        if _cos(ft, narr_feat) >= COS_SAME: return True
    return False

def _probe_zone(s, lo, hi):
    out_ = []
    t = lo
    while t <= hi + 1e-6:
        out_.append((round(t, 3), _card_at(round(t, 3), s["box"])))
        t += FINE
    return out_

refined = []
for i, s in enumerate(out):
    if s["region"] == "hero" or s["end"] - s["start"] < 1.0:
        refined.append(s); continue
    prev_h = i > 0 and out[i-1]["region"] == "hero"
    next_h = i + 1 < len(out) and out[i+1]["region"] == "hero"
    lo = max(0.0, s["start"] - (2.5 if prev_h else 0.0))
    hi = min(DUR - 0.1, s["end"] + (2.5 if next_h else 0.0))
    if s["end"] - s["start"] <= 8.0:
        seq = _probe_zone(s, lo, hi)
    else:
        head = _probe_zone(s, lo, s["start"] + 3.0)
        while (not head[-1][1]) and head[-1][0] + FINE < s["end"] - 3.0:
            nt = round(head[-1][0] + FINE, 3)
            head.append((nt, _card_at(nt, s["box"])))
        tail = _probe_zone(s, s["end"] - 3.0, hi)
        while (not tail[0][1]) and tail[0][0] - FINE > head[-1][0]:
            nt = round(tail[0][0] - FINE, 3)
            tail.insert(0, (nt, _card_at(nt, s["box"])))
        seq = list(head) + list(tail)
    trues = [t for t, p in seq if p]
    if not trues:
        refined.append(s); continue
    ns = s["start"]; ne = s["end"]
    if not (i == 0 and s["start"] <= 5.0):  # fade-in : bord temporel gere plus bas
        ns = max(lo, trues[0] - FINE / 2.0)
    ne = min(hi, trues[-1] + FINE / 2.0)
    if prev_h and ns < out[i-1]["end"]:
        out[i-1]["end"] = round(max(out[i-1]["start"] + 0.5, ns), 2)
    if next_h and ne > out[i+1]["start"]:
        out[i+1]["start"] = round(min(out[i+1]["end"] - 0.5, ne), 2)
    # scission uniquement sur trous PROUVES par sondes contigues : un trou qui
    # chevauche le coeur non sonde (scenes > 8s) n'est pas une absence de carte,
    # c'est une absence de SONDES (bug 27/07 09h40 : le point synthetique
    # fragmentait 4D7 0-86 en 0-3.1 + 82.9-85.1).
    if s["end"] - s["start"] <= 8.0:
        head_hi = seq[-1][0]; tail_lo = seq[0][0]
    else:
        head_hi = head[-1][0]; tail_lo = tail[0][0]
    pieces = []; cur = ns
    for a, b in zip(trues, trues[1:]):
        if b - a >= 0.9 + FINE and (b <= head_hi + 1e-6 or a >= tail_lo - 1e-6):
            pieces.append((cur, a + FINE / 2.0)); cur = b - FINE / 2.0
    pieces.append((cur, ne))
    for p0, p1 in pieces:
        if p1 - p0 < 0.5: continue
        ns2 = dict(s)
        ns2["start"] = round(float(p0), 2); ns2["end"] = round(float(p1), 2)
        refined.append(ns2)
out = refined
out.sort(key=lambda s: s["start"])

# NON-CHEVAUCHEMENT : build_seg concatene les scenes bout a bout. Deux scenes qui se
# recouvrent dupliquent donc leur intersection dans le montage : la piste video s'allonge
# et TOUT ce qui suit est decale (eglV passe 8 : scenes 36-40.5 / 37.5-42.6 / 39.5-40 ->
# DESYNC(1401.6) sur une source de 1395s, et le vert se retrouve peint sur les mauvaises
# frames). Le padding de debut de PASS 5 (t-0.5 / t-1.0) et les scissions de PASS 6 peuvent
# reculer un debut sous la fin precedente : on rogne la scene precedente, jamais la suivante
# (une carte prouvee tard prime sur du padding).
norm = []
for s in out:
    if norm and s["start"] < norm[-1]["end"] - 1e-6:
        norm[-1]["end"] = round(max(norm[-1]["start"], s["start"]), 2)
        if norm[-1]["end"] - norm[-1]["start"] < 0.3: norm.pop()
    if s["end"] - s["start"] >= 0.3: norm.append(s)
out = norm

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
