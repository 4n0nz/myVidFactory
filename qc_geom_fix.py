#!/usr/bin/env python3
# qc_geom_fix.py <workdir> — appliquer les corrections du QC geometrique :
#   SOUS-COUVERTURE -> box scene = union(box, carte reelle mesuree), patched (TzJC : torse
#                      sous l'avatar, invisible au QC identite car pas de visage)
#   TROP-GRAND      -> box resserree a la carte reelle (+2% marge) si pas deja patchee
import sys, os, json

wd = sys.argv[1]
fails = json.load(open(os.path.join(wd, "qc_geom.json")))
pinf = os.path.join(wd, "host_map_pin.json")
pin = json.load(open(pinf))

def union(a, b):
    x0 = min(a[0], b[0]); y0 = min(a[1], b[1])
    x1 = max(a[0]+a[2], b[0]+b[2]); y1 = max(a[1]+a[3], b[1]+b[3])
    return [round(x0,4), round(y0,4), round(x1-x0,4), round(y1-y0,4)]

n = 0
adjusted = {}   # position (cx,cy arrondis) -> nouvelle box, pour harmoniser les voisines
for f in fails:
    t = f["t"]
    for s in pin:
        if s["region"] != "hero" and s["start"] <= t <= s["end"]:
            if f["type"] == "SOUS-COUVERTURE":
                s["box"] = union(s["box"], f["carte"]); s["patched"] = True; n += 1
            # TROP-GRAND : PAS de resserrage automatique — les deux boucles correctives
            # s'ecrasaient mutuellement (ident elargit, geom resserre) -> oscillation
            # destructrice, gb5 LEAK_242. Corrections MONOTONES (grandir seulement) =
            # convergence garantie ; trop-grand reste au rapport, traite a la main.
            b = s["box"]
            adjusted[(round(b[0]+b[2]/2,1), round(b[1]+b[3]/2,1))] = (list(b), bool(s.get("patched")))
            break

# harmonisation : les scenes de la meme position prennent la box ajustee (pas de saut)
for s in pin:
    if s["region"] == "hero": continue
    b = s["box"]; key = (round(b[0]+b[2]/2,1), round(b[1]+b[3]/2,1))
    if key in adjusted:
        nb, pt = adjusted[key]
        s["box"] = list(nb)
        if pt: s["patched"] = True

json.dump(pin, open(pinf, "w"), indent=2)
print("qc_geom_fix: %d scenes ajustees" % n)
