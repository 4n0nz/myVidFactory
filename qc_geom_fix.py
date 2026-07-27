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
    return _cons_box(b) is not None

def _cons_box(b):
    # box consensus du cluster qui correspond a cette scene, ou None.
    cx = b[0]+b[2]/2; cy = b[1]+b[3]/2
    for c in _cons:
        cb = c["box"]
        if cb[2]*cb[3] > 0.85 or c.get("kind") == "hero":
            continue   # cluster geant : pin_render ne l'applique pas non plus
        if abs(cx-(cb[0]+cb[2]/2)) < 0.15 and abs(cy-(cb[1]+cb[3]/2)) < 0.15:
            return list(cb)
    return None

SIZE_CONS = 0.70

def _same_dim(a, b):
    return min(a, b) / max(a, b, 1e-6) > SIZE_CONS

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
                # REFERENTIEL : la carte a ete mesuree sur le RENDU, et le rendu d'une
                # scene qui matche un cluster suit la box CONSENSUS, pas la box pinpoint.
                # Unir dans le mauvais referentiel additionne les defauts de la box
                # pinpoint a la correction : la carte-fallback de pinpoint3 (visage x3.0
                # en hauteur, card_extent absent) descend jusqu'au bord bas du cadre, si
                # bien que eglV rendait [0.016,0.4989,0.251,0.5011] la ou consensus U
                # carte donne [0.015,0.5463,0.241,0.4297] -> vert colle au bas du cadre
                # et marge gauche mangee alors que la carte source, elle, a ses marges
                # (verdict Boss 27/07 00h20). Et comme le patch pose patched_keep, la
                # box consensus ne pouvait plus jamais reprendre la main : le defaut
                # etait fige pour toutes les passes suivantes.
                cb = _cons_box(s["box"])
                base = cb or s["box"]
                ub = union(base, f["carte"])
                # patched_keep SYSTEMATIQUE : pin_render remplace toute box non-keep
                # par le cluster consensus -> les unions des tours geom etaient
                # ecrasees au re-render (eglV passe 7 : 33/35/37 unionnees aux tours
                # 1-2, masque final = box pinpoint intacte, boucle sterile x3).
                # Le flag SOUS-COUVERTURE est deja gate par strip_act (bande vivante
                # prouvee) et true_rect est a fenetre adaptative : la mesure vaut
                # plus que le consensus la ou une fuite est PROUVEE. (l'ancien
                # blocage des grosses unions protegeait 4D7 des true_rect surestimes
                # d'AVANT le gate strip_act ; regression 4D7 re-verifiee 27/07.)
                # GARDE-FOU TAILLE : true_rect SURESTIME sur fond sombre (dit deja en
                # tete de fichier) et sa fenetre adaptative peut pousser 4x0.12 par cote,
                # si bien qu'une mesure ratee ne rate pas de peu : eglV passe 14, t=9.7
                # renvoie carte=[0.004,0.261,0.645,0.731] alors que t=8.3 mesure la MEME
                # carte a [0.024,0.558,0.211,0.427] et que le consensus (n=1335) dit
                # 0.227x0.405. L'union a gonfle l'intro a 0.347 de large -> 300px de vert
                # sur la page du navigateur, que le verdict Boss du 27/07 07h25 compte
                # aussi grave qu'une carte a nu, et la meme scene ressortait TROP-GRAND
                # (ratio 1.81) au tour suivant : les deux flags se contredisaient.
                # Pire, la correction est MONOTONE et posee patched_keep : chaque tour
                # ne pouvait que regonfler, jamais revenir. Une mesure qui contredit le
                # consensus EN TAILLE (doctrine "un layout = position ET taille") est un
                # artefact de mesure, pas une fuite : on ne patche pas, la scene garde sa
                # box et pin_render lui rendra le consensus.
                if cb is None or (_same_dim(cb[2], ub[2]) and _same_dim(cb[3], ub[3])):
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
