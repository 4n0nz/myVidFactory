#!/usr/bin/env python3
# pinpoint.py <workdir> — POSITION EXACTE du pip + DUREE par scene.
# Division du travail (Boss) :
#   VLM coarse (~4s) = quel COIN / "aucune" -> structure des scenes + tue les decoys (robuste).
#   OpenCV (0.5s dans le coin) = box EXACTE ; MEDIANE sur la scene stable = rock-solid, bien centre.
#   Edge-snap : cote a <2.5% d'un bord ecran = cam collee -> etend au bord ; sinon flottant -> serre.
# Sortie : tableau + host_map_pin.json (start,end,box,region,edges).
import sys, os, json, cv2, numpy as np, statistics
sys.path.insert(0, "/home/boss/videogen/agent_yt")
sys.path.insert(0, "/home/boss/yolo/scripts")
import vlm_probe
try: import card_extent
except Exception: card_extent = None

wd = sys.argv[1]
src = os.path.join(wd, "source.mp4")
YUNET = "/home/boss/videogen/face_detection_yunet_2023mar.onnx"
cap = cv2.VideoCapture(src)
W = int(cap.get(3)); H = int(cap.get(4)); FPS = cap.get(5) or 30; DUR = cap.get(7)/FPS
yfd = cv2.FaceDetectorYN.create(YUNET, "", (W, H), score_threshold=0.6)

VLM_STEP = 4.0     # VLM coarse
BOX_STEP = 0.5     # OpenCV fin
EDGE = 0.025
QUAD = {"top-left":(0,0,0.55,0.6),"top-right":(0.45,0,0.55,0.6),
        "bottom-left":(0,0.4,0.55,0.6),"bottom-right":(0.45,0.4,0.55,0.6),
        "center":(0.2,0.2,0.6,0.6)}

def _frame(t):
    cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0); ok, fr = cap.read()
    return fr if ok else None

# 1. VLM coarse -> region par sample. region()=="none" peut vouloir dire deux choses :
#    vrai contenu (off) OU narrateur PLEIN ECRAN sans overlay (hero) -> fullface tranche.
#    Sans ca, les heros tombent dans les trous "off" et restent a decouvert (0sqC 1ere minute).
vs = []
t = 0.0
while t < DUR:
    fr = _frame(t)
    r = vlm_probe.region(fr) if fr is not None else None
    r = r if r in QUAD else "none"
    if r == "none" and fr is not None and vlm_probe.fullface(fr) is True:
        r = "hero"
    vs.append((round(t,2), r))
    t += VLM_STEP

# 2. scenes = runs de region identique (!= none)
scenes = []; cur = None
for tt, r in vs:
    if r == "none":
        if cur: scenes.append(cur); cur = None
        continue
    if cur and cur["region"] == r:
        cur["end"] = tt
    else:
        if cur: scenes.append(cur)
        cur = {"region": r, "start": tt, "end": tt}
if cur: scenes.append(cur)

def box_in_quad(fr, quad):
    """box webcam dans le quadrant : plus gros visage DEDANS -> carte (contour) ou ancre-visage."""
    _, faces = yfd.detect(fr)
    if faces is None: return None
    qx,qy,qw,qh = quad; cand = []
    for f in faces:
        cx,cy = (f[0]+f[2]/2)/W, (f[1]+f[3]/2)/H
        if qx<=cx<=qx+qw and qy<=cy<=qy+qh: cand.append(f)
    if not cand: return None
    f = max(cand, key=lambda z: z[2]*z[3])
    if card_extent is not None and hasattr(card_extent,"_card_one"):
        cb = card_extent._card_one(fr, int(f[0]),int(f[1]),int(f[2]),int(f[3]))
        if cb is not None:
            x,y,w,h = cb; return [x/W,y/H,w/W,h/H]
    fx,fy,fw,fh = f[0]/W,f[1]/H,f[2]/W,f[3]/H
    return [max(0,fx-fw*0.6),max(0,fy-fh*0.7),min(1,fw*2.2),min(1,fh*3.0)]

# 3. box exacte par scene. EXTENT = percentile HAUT (sur-couvre, garantit couverture — Boss).
#    Position (x0,y0) = percentile BAS ; coin oppose (x1,y1) = percentile HAUT -> box englobante.
def pc(vals, q):
    v = sorted(vals); i = min(len(v)-1, max(0, int(q*(len(v)-1)))); return v[i]
out = []
for sc in scenes:
    t0, t1 = sc["start"], sc["end"] + VLM_STEP
    t1 = min(round(t1,2), round(DUR,2))
    if t1 - t0 < 1.0: continue
    if sc["region"] == "hero":
        # hero = avatar plein ecran, pas de box a detecter (etapes OpenCV/fusion box sautees)
        out.append({"start":float(t0),"end":float(t1),"region":"hero",
                    "box":[0.0,0.0,1.0,1.0],"edges":["L","T","R","B"],"n":0})
        continue
    quad = QUAD[sc["region"]]
    boxes = []
    t = t0 + 0.3
    while t < t1:
        fr = _frame(t)
        if fr is not None:
            b = box_in_quad(fr, quad)
            if b: boxes.append(b)
        t += BOX_STEP
    if len(boxes) < 3: continue
    x0 = pc([b[0] for b in boxes], 0.15); y0 = pc([b[1] for b in boxes], 0.15)
    x1 = pc([b[0]+b[2] for b in boxes], 0.85); y1 = pc([b[1]+b[3] for b in boxes], 0.85)
    edges = []
    if x0<EDGE: x0=0.0; edges.append("L")
    if y0<EDGE: y0=0.0; edges.append("T")
    if x1>1-EDGE: x1=1.0; edges.append("R")
    if y1>1-EDGE: y1=1.0; edges.append("B")
    out.append({"start":float(t0),"end":float(t1),"region":sc["region"],
                "box":[round(float(x0),4),round(float(y0),4),round(float(x1-x0),4),round(float(y1-y0),4)],
                "edges":edges,"n":len(boxes)})

# 3b. FUSION des scenes adjacentes meme coin + box proche (les fragments de 4s se recollent)
def close(a, b):
    return (a["region"]==b["region"] and abs(a["box"][0]-b["box"][0])<0.06
            and abs(a["box"][1]-b["box"][1])<0.06 and abs(a["box"][2]-b["box"][2])<0.08
            and abs(a["box"][3]-b["box"][3])<0.08 and b["start"]-a["end"] < 6.0)
merged = []
for s in out:
    if merged and close(merged[-1], s):
        p = merged[-1]; p["end"] = s["end"]
        p["box"] = [max(p["box"][0],0) if False else (p["box"][k]) for k in range(4)]
        # box fusionnee = englobante des deux (sur-couvre)
        px0,py0,px1,py1 = p["box"][0],p["box"][1],p["box"][0]+p["box"][2],p["box"][1]+p["box"][3]
        sx0,sy0,sx1,sy1 = s["box"][0],s["box"][1],s["box"][0]+s["box"][2],s["box"][1]+s["box"][3]
        nx0,ny0,nx1,ny1 = min(px0,sx0),min(py0,sy0),max(px1,sx1),max(py1,sy1)
        p["box"] = [round(nx0,4),round(ny0,4),round(nx1-nx0,4),round(ny1-ny0,4)]
        p["edges"] = sorted(set(p["edges"]) | set(s["edges"]))
    else:
        merged.append(dict(s))
out = merged

# 3c. CLUSTERING POSITION (Boss : le pip est STATIQUE -> UNE box canonique par position).
# Toutes les scenes d'une meme position (centres proches) recoivent la MEME box = l'ENVELOPPE
# (union) de leurs box -> la plus grande vue. Sur-couvre = OK, plus de variation entre scenes,
# plus de fuite tardive quand une scene avait detecte trop petit.
def _center(b): return (b[0]+b[2]/2, b[1]+b[3]/2)
clusters = []
for s in out:
    if s["region"] == "hero": continue   # box plein ecran, ne doit pas avaler un cluster "center"
    cx, cy = _center(s["box"])
    hit = None
    for c in clusters:
        ccx, ccy = _center(c["box"])
        if abs(cx-ccx) < 0.09 and abs(cy-ccy) < 0.09:
            hit = c; break
    if hit is None:
        clusters.append({"box": list(s["box"]), "members": [s]})
    else:
        b = hit["box"]; nb = s["box"]
        x0 = min(b[0], nb[0]); y0 = min(b[1], nb[1])
        x1 = max(b[0]+b[2], nb[0]+nb[2]); y1 = max(b[1]+b[3], nb[1]+nb[3])
        hit["box"] = [x0, y0, x1-x0, y1-y0]
        hit["members"].append(s)
for c in clusters:
    x0,y0 = c["box"][0], c["box"][1]
    x1,y1 = x0+c["box"][2], y0+c["box"][3]
    edges = []
    if x0<EDGE: x0=0.0; edges.append("L")
    if y0<EDGE: y0=0.0; edges.append("T")
    if x1>1-EDGE: x1=1.0; edges.append("R")
    if y1>1-EDGE: y1=1.0; edges.append("B")
    cb = [round(float(x0),4),round(float(y0),4),round(float(x1-x0),4),round(float(y1-y0),4)]
    for s in c["members"]:
        s["box"] = list(cb); s["edges"] = edges
print("clusters position : %d (box canonique partagee)" % len(clusters))

# 3d. RECUPERATION DES TROUS (v2) : toute probe VLM SEULE flake (mesure 0sqC : region() rate
# des pips 3 samples de suite, fullface rate un plein-ecran, crop-verify rate une cam cercle).
# Donc DETERMINISTE d'abord : YuNet echantillonne chaque trou. Pas de visage -> vrai off, zero
# VLM. Visage assez gros (>= 0.09*H, tue vignettes/avatars decoys) DANS une box canonique ->
# crop-verify 2 frames (OR) -> scene pip inseree a la box canonique. Gros visage plein cadre
# hors canonique -> fullface 2 frames (OR) -> scene hero. Sous-couvrir interdit (regle Boss).
FACE_MIN = 0.09
canon = {}
for s in out:
    if s["region"] != "hero":
        canon.setdefault(tuple(s["box"]), s["region"])

def _faces(fr):
    _, fs = yfd.detect(fr)
    return [] if fs is None else fs

def _motion(t, box, fr=None):
    """diff moyenne du crop entre t et t+0.8s. Cam LIVE qui parle >> 0 ; vignette/photo
    statique ~ 0. Tranche ce que le VLM ne peut pas : photo du narrateur vs cam live (876s)."""
    fr1 = fr if fr is not None else _frame(t)
    fr2 = _frame(min(t+0.8, DUR-0.05))
    if fr1 is None or fr2 is None: return 99.0
    x=int(box[0]*W); y=int(box[1]*H); w=max(4,int(box[2]*W)); h=max(4,int(box[3]*H))
    x=max(0,min(W-w,x)); y=max(0,min(H-h,y))
    a=cv2.cvtColor(fr1[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
    b=cv2.cvtColor(fr2[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
    return float(cv2.absdiff(a,b).mean())

def _card_box(fr, f):
    """carte webcam autour du visage f (card_extent, fallback ancre-visage)."""
    if card_extent is not None and hasattr(card_extent,"_card_one"):
        cb = card_extent._card_one(fr, int(f[0]),int(f[1]),int(f[2]),int(f[3]))
        if cb is not None:
            x,y,w,h = cb; return [x/W,y/H,w/W,h/H]
    fx,fy,fw,fh = f[0]/W,f[1]/H,f[2]/W,f[3]/H
    return [max(0,fx-fw*0.6),max(0,fy-fh*0.7),min(1,fw*2.2),min(1,fh*3.0)]

def _covers(cb, card):
    """cb couvre >= 85% de la carte (sinon bas de carte a decouvert — fuite 26s pin5)."""
    ix0=max(cb[0],card[0]); iy0=max(cb[1],card[1])
    ix1=min(cb[0]+cb[2],card[0]+card[2]); iy1=min(cb[1]+cb[3],card[1]+card[3])
    return max(0,ix1-ix0)*max(0,iy1-iy0) >= 0.85*card[2]*card[3]

MOTION_MIN = 1.5

def _gap_probe(t):
    """decision pour un sample de trou : ("pip", box, region) | ("hero",) | None.
    Ordre : gros visage -> hero d'abord (fuite plein ecran = pire cas), sinon match canonique."""
    fr = _frame(t)
    if fr is None: return None
    faces = [f for f in _faces(fr) if f[3]/H >= FACE_MIN]
    if not faces: return None
    fr2 = _frame(min(t+1.5, DUR-0.1))
    big = max(faces, key=lambda f: f[3])
    if big[3]/H > 0.22:
        if vlm_probe.fullface(fr) is True: return ("hero",)
        if fr2 is not None and vlm_probe.fullface(fr2) is True: return ("hero",)
    # box la plus PETITE d'abord : les grosses colonnes chevauchent tout l'ecran, sans tri
    # le choix alterne d'un sample a l'autre -> avatar qui saute (flicker pin4 512-528s).
    # Gates : box doit COUVRIR la carte detectee (fuite bas de carte 26s pin5) + MOUVEMENT
    # dans la box (tue les photos/vignettes du narrateur que crop-verify prend pour une cam).
    order = sorted(canon.items(), key=lambda kv: kv[0][2]*kv[0][3])
    for f in faces:
        cx, cy = (f[0]+f[2]/2)/W, (f[1]+f[3]/2)/H
        card = _card_box(fr, f)
        for cb, reg in order:
            if not (cb[0] <= cx <= cb[0]+cb[2] and cb[1] <= cy <= cb[1]+cb[3]): continue
            if not _covers(cb, card): continue
            if _motion(t, list(cb), fr) < MOTION_MIN: continue
            if vlm_probe.webcam_crop(fr, list(cb)) is True:
                return ("pip", list(cb), reg)
            if fr2 is not None and vlm_probe.webcam_crop(fr2, list(cb)) is True:
                return ("pip", list(cb), reg)
        # aucune canonique ne couvre la carte -> la carte elle-meme (+4% marge), memes gates
        px0=max(0.0,float(card[0])-0.04); py0=max(0.0,float(card[1])-0.04)
        px1=min(1.0,float(card[0]+card[2])+0.04); py1=min(1.0,float(card[1]+card[3])+0.04)
        pad=[round(px0,4),round(py0,4),round(px1-px0,4),round(py1-py0,4)]
        if _motion(t, pad, fr) >= MOTION_MIN:
            if vlm_probe.webcam_crop(fr, pad) is True:
                return ("pip", pad, "recup")
            if fr2 is not None and vlm_probe.webcam_crop(fr2, pad) is True:
                return ("pip", pad, "recup")
    return None

out.sort(key=lambda s: s["start"])
extra = []
edges_gap = [(0.0, out[0]["start"] if out else DUR)] + \
            [(out[i]["end"], out[i+1]["start"]) for i in range(len(out)-1)] + \
            ([(out[-1]["end"], DUR)] if out else [])
for gap0, gap1 in edges_gap:
    if gap1 - gap0 < 2.0: continue
    t = gap0 + 1.0; cur = None
    while t < gap1 - 0.5:
        # STICKY : si une scene pip est en cours, verifier d'abord SA box (stabilite > re-decision).
        # Gate mouvement AVANT le VLM : une box devenue statique = plus une cam, on re-decide.
        if cur is not None and cur["region"] != "hero":
            frs = _frame(t)
            if (frs is not None and _motion(t, cur["box"], frs) >= MOTION_MIN
                    and vlm_probe.webcam_crop(frs, cur["box"]) is True):
                cur["end"] = round(float(min(t + 2.0, gap1)), 2)
                t += 2.0; continue
        d = _gap_probe(t)
        key = None if d is None else (d[0], tuple(d[1]) if d[0] == "pip" else None)
        end_t = round(float(min(t + 2.0, gap1)), 2)
        if d is not None and cur is not None and cur["_key"] == key:
            cur["end"] = end_t
        elif d is not None:
            if cur: extra.append(cur)
            if d[0] == "hero":
                cur = {"start": round(float(max(t-1.0, gap0)),2), "end": end_t, "region": "hero",
                       "box": [0.0,0.0,1.0,1.0], "edges": ["L","T","R","B"], "n": 0, "_key": key}
            else:
                cur = {"start": round(float(max(t-1.0, gap0)),2), "end": end_t, "region": d[2],
                       "box": list(d[1]), "edges": [], "n": 0, "_key": key}
        else:
            if cur: extra.append(cur); cur = None
        t += 2.0
    if cur: extra.append(cur)
for e in extra:
    e.pop("_key", None)
    print("  RECUP trou : %.1f-%.1fs %s %s" % (e["start"], e["end"], e["region"], e["box"]))
out.extend(extra)
out.sort(key=lambda s: s["start"])

# 3e. CHEVAUCHEMENTS + MERGE : les fenetres de recup se recouvrent (pin4 : videoonly 1035s vs
# source 1002s -> desync A/V). Clip au precedent, fusion des adjacents meme box, vides jetes.
clean = []
for s in out:
    if clean:
        p = clean[-1]
        same = (s["region"] == p["region"] and s["box"] == p["box"])
        if s["start"] < p["end"] or (same and s["start"] - p["end"] < 0.5):
            if same:
                p["end"] = max(p["end"], s["end"]); continue
            s["start"] = p["end"]
            if s["end"] - s["start"] < 0.5: continue
    clean.append(s)
out = clean

json.dump(out,open(os.path.join(wd,"host_map_pin.json"),"w"),indent=2)
print("SCENES pip (%d) : start-end | duree | box | coin | bords" % len(out))
for s in out:
    print("  %.1f-%.1fs | %4.1fs | [%.3f,%.3f,%.3f,%.3f] | %-12s | %s"
          % (s["start"],s["end"],s["end"]-s["start"],*s["box"],s["region"],"".join(s["edges"]) or "flottant"))
cap.release()
