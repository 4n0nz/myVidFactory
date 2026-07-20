#!/usr/bin/env python3
# qc_fix.py <workdir> — boucle corrective "points rouges" (idee Boss) : les fuites detectees
# par qc_ident (qc_leaks.json) sont injectees comme scenes correctives dans host_map_pin.json.
# Peu importe POURQUOI la detection a rate : la fuite observee EST couverte au tour suivant.
# - fuites groupees (temps proche + zone proche) -> un patch englobant, pas 40 micro-scenes
# - patch sur scene existante -> la box devient l'UNION (jamais retrecir, sous-couvrir interdit)
# - patch sur trou "off" -> scene pip inseree
import sys, os, json

wd = sys.argv[1]
leaks = json.load(open(os.path.join(wd, "qc_leaks.json")))
pinf = os.path.join(wd, "host_map_pin.json")
pin = json.load(open(pinf))
if not leaks:
    print("qc_fix: aucune fuite a corriger"); sys.exit(0)

PAD_T = 1.5     # marge temporelle autour de la fuite
MARG = 0.35     # la box fuite = le VISAGE ; la carte autour = visage elargi

def face_to_card(b):
    x0 = max(0.0, b[0] - b[2]*0.8); y0 = max(0.0, b[1] - b[3]*0.6)
    x1 = min(1.0, b[0] + b[2]*1.8); y1 = min(1.0, b[1] + b[3]*2.6)
    return [x0, y0, x1-x0, y1-y0]

def union(a, b):
    x0 = min(a[0], b[0]); y0 = min(a[1], b[1])
    x1 = max(a[0]+a[2], b[0]+b[2]); y1 = max(a[1]+a[3], b[1]+b[3])
    return [round(x0,4), round(y0,4), round(x1-x0,4), round(y1-y0,4)]

# groupage : fuites triees par t ; meme groupe si gap <= 3s et centres a < 0.2
groups = []
for L in sorted(leaks, key=lambda z: z["t"]):
    card = face_to_card(L["box"])
    cx, cy = card[0]+card[2]/2, card[1]+card[3]/2
    g = groups[-1] if groups else None
    if g and L["t"] - g["t1"] <= 3.0 and abs(cx-g["cx"]) < 0.2 and abs(cy-g["cy"]) < 0.2:
        g["t1"] = L["t"]; g["box"] = union(g["box"], card)
        g["cx"], g["cy"] = g["box"][0]+g["box"][2]/2, g["box"][1]+g["box"][3]/2
    else:
        groups.append({"t0": L["t"], "t1": L["t"], "box": card, "cx": cx, "cy": cy})

def insert_patch(pin, t0, t1, pbox):
    out = []
    for s in pin:
        if s["end"] <= t0 or s["start"] >= t1:
            out.append(s); continue
        if s["start"] < t0:
            out.append(dict(s, end=round(t0,2)))
        mid = dict(s, start=round(max(s["start"], t0),2), end=round(min(s["end"], t1),2))
        if s["region"] != "hero":
            ub = union(s["box"], pbox)
            if ub[2]*ub[3] > 0.5:
                # union quasi plein ecran : une ellipse/rect geant ne couvre pas ses coins
                # (KKni 56.5s, boucle QC sterile) -> hero plein ecran
                mid["region"] = "hero"; mid["box"] = [0.0,0.0,1.0,1.0]; mid["edges"] = ["L","T","R","B"]
            else:
                mid["box"] = ub; mid["patched"] = True
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
        out.append({"start": round(a,2), "end": round(b,2), "region": "patch",
                    "box": [round(v,4) for v in pbox], "edges": [], "n": 0, "src": "ident"})
    out.sort(key=lambda s: s["start"])
    return out

dur_max = max(s["end"] for s in pin) if pin else 0
for g in groups:
    t0 = max(0.0, g["t0"] - PAD_T); t1 = min(dur_max, g["t1"] + PAD_T) if dur_max else g["t1"] + PAD_T
    pin = insert_patch(pin, t0, t1, g["box"])
    print("PATCH %.1f-%.1fs box=%s (%d->" % (t0, t1, g["box"], len(pin)))

# scenes degeneres jetees
pin = [s for s in pin if s["end"] - s["start"] >= 0.3]
json.dump(pin, open(pinf, "w"), indent=2)
print("qc_fix: %d groupes patches, %d scenes total" % (len(groups), len(pin)))
