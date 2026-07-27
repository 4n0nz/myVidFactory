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

# CONSENSUS AUTORITAIRE : une scene qui matche un cluster box_consensus ne se fait PAS
# patcher par le QC geometrique — son true_rect SURESTIME la carte sur fond sombre et
# ecrasait la mesure globale validee (4D7 : consensus 0.21x0.25 correct -> patch 0.28x0.28
# -> +marge = vert +60% d'aire, verdict Boss). Sous-couverture reelle -> rapport seulement.
_cons = []
_cp = os.path.join(wd, "box_consensus.json")
if os.path.exists(_cp):
    try: _cons = json.load(open(_cp))
    except Exception: _cons = []
def _cons_matched(b):
    cx = b[0]+b[2]/2; cy = b[1]+b[3]/2
    for c in _cons:
        cb = c["box"]
        if abs(cx-(cb[0]+cb[2]/2)) < 0.15 and abs(cy-(cb[1]+cb[3]/2)) < 0.15:
            return True
    return False

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
                ub = union(s["box"], f["carte"])
                # patched_keep SYSTEMATIQUE : pin_render remplace toute box non-keep
                # par le cluster consensus -> les unions des tours geom etaient
                # ecrasees au re-render (eglV passe 7 : 33/35/37 unionnees aux tours
                # 1-2, masque final = box pinpoint intacte, boucle sterile x3).
                # Le flag SOUS-COUVERTURE est deja gate par strip_act (bande vivante
                # prouvee) et true_rect est a fenetre adaptative : la mesure vaut
                # plus que le consensus la ou une fuite est PROUVEE. (l'ancien
                # blocage des grosses unions protegeait 4D7 des true_rect surestimes
                # d'AVANT le gate strip_act ; regression 4D7 re-verifiee 27/07.)
                s["box"] = ub; s["patched"] = True; s["patched_keep"] = True; n += 1
            # TROP-GRAND : PAS de resserrage automatique — les deux boucles correctives
            # s'ecrasaient mutuellement (ident elargit, geom resserre) -> oscillation
            # destructrice, gb5 LEAK_242. Corrections MONOTONES (grandir seulement) =
            # convergence garantie ; trop-grand reste au rapport, traite a la main.
            b = s["box"]
            adjusted[(round(b[0]+b[2]/2,1), round(b[1]+b[3]/2,1))] = \
                (list(b), bool(s.get("patched")), bool(s.get("patched_keep")))
            break

# harmonisation : les scenes de la meme position prennent la box ajustee (pas de saut)
for s in pin:
    if s["region"] == "hero": continue
    b = s["box"]; key = (round(b[0]+b[2]/2,1), round(b[1]+b[3]/2,1))
    if key in adjusted:
        nb, pt, pk = adjusted[key]
        s["box"] = list(nb)
        if pt: s["patched"] = True
        if pk: s["patched_keep"] = True

# FAUX-HERO -> la scene hero redevient pip avec la box du cluster pip dominant
# (le narrateur ne se teleporte pas ; la scene hero etait une misclassification
# scroll/contenu ou une promotion par patch identite fantome)
_fh = [f for f in fails if f["type"] == "FAUX-HERO"]
if _fh:
    from collections import Counter
    _bc = Counter(tuple(s["box"]) for s in pin if s["region"] != "hero")
    if _bc:
        _db = list(_bc.most_common(1)[0][0])
        _ref = next(s for s in pin if s["region"] != "hero" and list(s["box"]) == _db)
        for f in _fh:
            for s in pin:
                if s["region"] == "hero" and s["start"] <= f["t"] <= s["end"]:
                    s["region"] = _ref["region"]; s["box"] = list(_db)
                    s["edges"] = list(_ref.get("edges", [])); s["n"] = _ref.get("n", 0)
                    s["patched"] = True; s["patched_faux_hero"] = True; n += 1
                    break

json.dump(pin, open(pinf, "w"), indent=2)
print("qc_geom_fix: %d scenes ajustees" % n)
