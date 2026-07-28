#!/usr/bin/env python3
# qc_fix.py <workdir> — boucle corrective "points rouges" (idee Boss) : les fuites detectees
# par qc_ident (qc_leaks.json) sont injectees comme scenes correctives dans host_map_pin.json.
# Peu importe POURQUOI la detection a rate : la fuite observee EST couverte au tour suivant.
# - fuites groupees (temps proche + zone proche) -> un patch englobant, pas 40 micro-scenes
# - patch sur scene existante -> la box devient l'UNION (jamais retrecir, sous-couvrir interdit)
# - patch sur trou "off" -> scene pip inseree
import sys, os, json, cv2
sys.path.insert(0, "/home/boss/videogen")
import narr_role

wd = sys.argv[1]
leaks = json.load(open(os.path.join(wd, "qc_leaks.json")))
pinf = os.path.join(wd, "host_map_pin.json")
pin = json.load(open(pinf))
if not leaks:
    print("qc_fix: aucune fuite a corriger"); sys.exit(0)

PAD_T = 1.5     # marge temporelle autour de la fuite
MARG = 0.35     # la box fuite = le VISAGE ; la carte autour = visage elargi
# Un LAYOUT, c'est une position ET une taille (meme doctrine que pinpoint3, commit
# db6c3d4). Sans ce gate, qc_fix melange des layouts differents et le vert deborde :
# eglV, fuite "fenetre navigateur" 1365-1383 (carte 0.64x0.745) unie a la scene pip
# bas-gauche (0.178x0.501), puis l'unification par position propage la box gonflee aux
# 860 s de la video -> vert de 0.385 de large sur une carte mesuree a 0.227, marge
# gauche mangee (verdict Boss 27/07 00h20). La scene devenait aussi patched_ident, ce
# qui lui fait SAUTER le consensus dans pin_render : la box consensus, elle, etait juste.
SIZE_SAME = 0.55

def _same_size(a, b):
    return min(a, b) / max(a, b, 1e-6) > SIZE_SAME

def _same_layout(a, b):
    return _same_size(a[2], b[2]) and _same_size(a[3], b[3])

def face_to_card(b):
    x0 = max(0.0, b[0] - b[2]*0.8); y0 = max(0.0, b[1] - b[3]*0.6)
    x1 = min(1.0, b[0] + b[2]*1.8); y1 = min(1.0, b[1] + b[3]*2.6)
    return [x0, y0, x1-x0, y1-y0]

def union(a, b):
    x0 = min(a[0], b[0]); y0 = min(a[1], b[1])
    x1 = max(a[0]+a[2], b[0]+b[2]); y1 = max(a[1]+a[3], b[1]+b[3])
    return [round(x0,4), round(y0,4), round(x1-x0,4), round(y1-y0,4)]

# groupage : fuites triees par t ; meme groupe si gap <= 3s, centres a < 0.2 ET MEME
# LAYOUT. Le centre du groupe est la MOYENNE des centres de cartes, pas le centre de la
# box unie : recalcule sur l'union il DERIVE vers le milieu et avale la fuite suivante,
# si bien que deux layouts eloignes de 0.44 finissaient dans le meme groupe.
groups = []
for L in sorted(leaks, key=lambda z: z["t"]):
    card = face_to_card(L["box"])
    cx, cy = card[0]+card[2]/2, card[1]+card[3]/2
    g = groups[-1] if groups else None
    if (g and L["t"] - g["t1"] <= 3.0 and abs(cx-g["cx"]) < 0.2 and abs(cy-g["cy"]) < 0.2
            and _same_size(card[2], g["cw"]) and _same_size(card[3], g["ch"])):
        n = g["n"]
        g["t1"] = L["t"]; g["box"] = union(g["box"], card)
        g["cx"] = (g["cx"]*n+cx)/(n+1); g["cy"] = (g["cy"]*n+cy)/(n+1)
        g["cw"] = (g["cw"]*n+card[2])/(n+1); g["ch"] = (g["ch"]*n+card[3])/(n+1)
        g["n"] = n+1
    else:
        groups.append({"t0": L["t"], "t1": L["t"], "box": card, "cx": cx, "cy": cy,
                       "cw": card[2], "ch": card[3], "n": 1})

_cap = None; _nr = None; _W = 0; _H = 0
def _preuve(pbox, a, b):
    """PREUVE avant de peindre du vert sur un trou 'off'. Le patch pur est le seul endroit
    du pipeline qui CREE du vert la ou il n y en avait aucun : c est donc le seul qui peut
    violer "un vert sans carte dessous est aussi grave qu une carte sans vert". La fuite
    vient du RENDU (qc_ident), qui peut se tromper ; la SOURCE tranche.
    Mesure du 28/07 08h10, XzEg 508.00-508.88 : capture du site BLOCKFIELD, AUCUN narrateur
    nulle part (probe plein cadre ET dans la box : present=False faceH=0.000) — pourtant un
    pip de 0.269x0.810 y etait insere et peint en plein milieu de page.
    Recensement des 41 scenes patchees du corpus (4D7 0, mCE 0, eglV 5, XzEg 34) : 40 ont le
    narrateur present dans leur box, une seule ne l a pas et c est 508. Les quatre
    qc_leaks.json du corpus sont vides a l etat final (qc_fix y sort en no-op), le garde est
    donc inerte sur les etalons par construction en plus de l etre par mesure.
    La preuve porte sur le NARRATEUR SEUL et pas sur cardness, pour une raison mesuree :
    sur une capture de site cardness est une passoire. Rejoue du trou 508.0-508.88 avec une
    box reconstruite : cardness=True sur le seul bord bas (hit 0.68 std 4.9) — un filet
    horizontal de page, pas une carte — alors qu il n y a aucun narrateur. Une page web est
    pleine de lignes droites ; un bord franc n y prouve rien. Et cardness n aurait rien
    sauve : les 40 patches legitimes du corpus passent tous par le narrateur.
    Echec ouvert : narr_role.FACE_MIN=0.10 jette les petits visages, donc une carte legitime
    dont le visage est minuscule dans la box serait refusee. Aucun cas dans le corpus."""
    global _cap, _nr, _W, _H
    if _cap is None:
        _cap = cv2.VideoCapture(os.path.join(wd, "source.mp4"))
        if not _cap.isOpened(): return True      # source illisible : on ne juge pas a l aveugle
        _W = int(_cap.get(3)); _H = int(_cap.get(4))
        _nr = narr_role.NarrRole(wd, _W, _H)
    if not _cap.isOpened(): return True
    r = [int(pbox[0]*_W), int(pbox[1]*_H), int((pbox[0]+pbox[2])*_W), int((pbox[1]+pbox[3])*_H)]
    return bool(_nr.probe(_cap, a, b, rect=r, n=3)[0])

def insert_patch(pin, t0, t1, pbox, raw0=None, raw1=None):
    # [t0,t1] est la fuite ELARGIE de PAD_T. Une scene d'un AUTRE layout que la fuite,
    # touchee par ce seul rembourrage, ne doit pas etre unie : eglV, la fuite narrateur
    # live plein cadre (39-42.5, carte 0.35x1.0) debordait sur la scene pip bas-gauche
    # qui commence a 43 et lui donnait la box [0.060,0,0.702,1.0] -> vert geant sur le
    # conferencier alors que seul le pip bas-gauche doit etre couvert (verdict Boss
    # 27/07 07h50). Sur la fuite REELLE l'union reste (jamais sous-couvrir).
    if raw0 is None: raw0, raw1 = t0, t1
    out = []
    for s in pin:
        if s["end"] <= t0 or s["start"] >= t1:
            out.append(s); continue
        if (s["region"] != "hero" and not _same_layout(pbox, s["box"])
                and not (s["end"] > raw0 and s["start"] < raw1)):
            out.append(s); continue
        if s["start"] < t0:
            out.append(dict(s, end=round(t0,2)))
        mid = dict(s, start=round(max(s["start"], t0),2), end=round(min(s["end"], t1),2))
        if s["region"] != "hero":
            ub = union(s["box"], pbox)
            if ub[2]*ub[3] > 0.85:
                # union quasi plein ecran : une ellipse/rect geant ne couvre pas ses coins
                # (KKni 56.5s, boucle QC sterile) -> hero plein ecran
                mid["region"] = "hero"; mid["box"] = [0.0,0.0,1.0,1.0]; mid["edges"] = ["L","T","R","B"]
            else:
                mid["box"] = ub; mid["patched"] = True; mid["patched_ident"] = True
        out.append(mid)
        if s["end"] > t1:
            out.append(dict(s, start=round(t1,2)))
    # sous-intervalles de [t0,t1] sans scene -> patch pur
    out.sort(key=lambda s: s["start"])
    covered = [(s["start"], s["end"]) for s in out if s["start"] < t1 and s["end"] > t0]
    holes = []; cur = t0
    for a, b in sorted(covered):
        if a > cur + 0.05: holes.append((cur, min(a, t1)))
        cur = max(cur, b)
        if cur >= t1: break
    if cur < t1 - 0.05: holes.append((cur, t1))
    for a, b in holes:
        if not _preuve(pbox, a, b):
            print("REFUS patch pur %.2f-%.2f : aucun narrateur dans la box a la SOURCE" % (a, b))
            continue
        out.append({"start": round(a,2), "end": round(b,2), "region": "patch",
                    "box": [round(v,4) for v in pbox], "edges": [], "n": 0, "src": "ident"})
    out.sort(key=lambda s: s["start"])
    return out

dur_max = max(s["end"] for s in pin) if pin else 0
for g in groups:
    t0 = max(0.0, g["t0"] - PAD_T); t1 = min(dur_max, g["t1"] + PAD_T) if dur_max else g["t1"] + PAD_T
    pin = insert_patch(pin, t0, t1, g["box"], g["t0"], g["t1"])
    print("PATCH %.1f-%.1fs box=%s (%d->" % (t0, t1, g["box"], len(pin)))

# UNIFICATION des patches par position : chaque tour QC produisait des unions de tailles
# differentes -> l'avatar changeait sans que le pip source bouge (pLos popout). Toutes les
# scenes pip d'une position patchee recoivent LA box du groupe (le popout est permanent).
pgroups = []
for s in pin:
    if not s.get("patched"): continue
    b = s["box"]; cx, cy = b[0]+b[2]/2, b[1]+b[3]/2
    hit = None
    for g in pgroups:
        if (abs(cx-g["cx"]) < 0.08 and abs(cy-g["cy"]) < 0.08
                and _same_size(b[2], g["cw"]) and _same_size(b[3], g["ch"])): hit = g; break
    if hit is None:
        pgroups.append({"cx": cx, "cy": cy, "cw": b[2], "ch": b[3], "box": list(b), "n": 1})
    else:
        n = hit["n"]
        hit["box"] = union(hit["box"], b)
        hit["cx"] = (hit["cx"]*n+cx)/(n+1); hit["cy"] = (hit["cy"]*n+cy)/(n+1)
        hit["cw"] = (hit["cw"]*n+b[2])/(n+1); hit["ch"] = (hit["ch"]*n+b[3])/(n+1)
        hit["n"] = n+1
for g in pgroups:
    for s in pin:
        if s["region"] == "hero": continue
        b = s["box"]; cx, cy = b[0]+b[2]/2, b[1]+b[3]/2
        if (abs(cx-g["cx"]) < 0.08 and abs(cy-g["cy"]) < 0.08
                and _same_size(b[2], g["cw"]) and _same_size(b[3], g["ch"])):
            s["box"] = [round(v,4) for v in g["box"]]; s["patched"] = True; s["patched_ident"] = True

# scenes degeneres jetees
pin = [s for s in pin if s["end"] - s["start"] >= 0.3]
json.dump(pin, open(pinf, "w"), indent=2)
print("qc_fix: %d groupes patches, %d scenes total" % (len(groups), len(pin)))
