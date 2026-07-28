#!/usr/bin/env python3
# cardness.py — LE discriminant qui manquait : « est-ce une CARTE ? »
#
# Doctrine Boss (2026-07-27) : on ne remplace que (a) une carte pip du narrateur,
# (b) un narrateur LIVE plein cadre. Un visage dans du CONTENU = on ne touche pas.
# La decision se prenait sur la TAILLE du visage (seuil 0.30 H) — critere fragile :
# XzEg (narrateur plein cadre filme large, visage ~0.26 H) est tombe du mauvais cote
# et a recu une box etroite sur le torse (fail total du 22h00).
#
# Critere robuste et generique : un bord de CARTE est une DROITE. Pour chaque cote on
# cherche, le long de 40 lignes, la transition la plus forte dans une bande de +/-12 px
# et on regarde si ces positions sont ALIGNEES. On ecarte 20% a chaque extremite du
# cote : les cartes ont des COINS ARRONDIS qui font deriver la mesure (eglV).
#   carte reelle  -> beaucoup de lignes transitent ET l ecart-type des positions est petit
#   corps/contenu -> peu de lignes, positions dispersees
# Un cote colle au bord de l ecran n est pas mesurable (carte tronquee, eglV intro) : ignore.
import numpy as np
import cv2

GRAD_MIN = 26.0   # gradient RGB minimal pour compter une transition
HIT_MIN  = 0.50   # fraction des lignes qui doivent transiter pour mesurer un std
HIT_STRAIGHT = 0.58  # fraction requise pour parler d un vrai bord de carte
STD_MAX  = 5.0    # ecart-type max des positions (px) pour parler de DROITE
BAND     = 12     # demi-largeur de recherche autour du bord suppose


def _side_score(f, rect, side, W, H):
    x0, y0, x1, y1 = rect
    if side == 'L' and x0 <= 1: return None
    if side == 'R' and x1 >= W - 1: return None
    if side == 'T' and y0 <= 1: return None
    if side == 'B' and y1 >= H - 1: return None
    pos = []; tot = 0
    if side in ('L', 'R'):
        x = x0 if side == 'L' else x1
        for y in np.linspace(y0 + 0.20 * (y1 - y0), y1 - 0.20 * (y1 - y0), 40).astype(int):
            if y < 1 or y >= H - 1: continue
            tot += 1
            a = max(2, x - BAND); b = min(W - 2, x + BAND)
            if b - a < 4: continue
            seg = f[y, a - 2:b + 2]
            g = np.linalg.norm(seg[4:] - seg[:-4], axis=1)
            if g.size and g.max() > GRAD_MIN: pos.append(a + int(np.argmax(g)))
    else:
        y = y0 if side == 'T' else y1
        for x in np.linspace(x0 + 0.20 * (x1 - x0), x1 - 0.20 * (x1 - x0), 40).astype(int):
            if x < 1 or x >= W - 1: continue
            tot += 1
            a = max(2, y - BAND); b = min(H - 2, y + BAND)
            if b - a < 4: continue
            seg = f[a - 2:b + 2, x]
            g = np.linalg.norm(seg[4:] - seg[:-4], axis=1)
            if g.size and g.max() > GRAD_MIN: pos.append(a + int(np.argmax(g)))
    if not tot: return None
    hit = len(pos) / float(tot)
    if hit < HIT_MIN: return (hit, None)
    return (hit, float(np.std(pos)))


def card_score(cap, rect, t0, t1, W, H, nframes=3):
    """(is_card, detail) — detail = {cote: (hit, std)} median sur nframes."""
    acc = []
    for frac in np.linspace(0.25, 0.75, nframes):
        cap.set(cv2.CAP_PROP_POS_MSEC, (t0 + (t1 - t0) * frac) * 1000.0)
        ok, img = cap.read()
        if not ok: continue
        f = img.astype('float32')
        acc.append({s: _side_score(f, [int(v) for v in rect], s, W, H) for s in 'LTRB'})
    if not acc: return False, {}
    det = {}
    for s in 'LTRB':
        vals = [a[s] for a in acc if a.get(s) is not None]
        if not vals: det[s] = None; continue
        hits = sorted(v[0] for v in vals); stds = [v[1] for v in vals if v[1] is not None]
        det[s] = (round(hits[len(hits) // 2], 2),
                  round(sorted(stds)[len(stds) // 2], 1) if stds else None)
    usable = [v for v in det.values() if v is not None]
    if not usable: return False, det          # 4 cotes a l ecran = plein cadre, pas une carte
    # REGLE : il suffit d UN cote qui soit une VRAIE DROITE (transition sur >=58% de sa
    # longueur ET positions alignees a <=5 px). Une carte en a toujours au moins un, meme
    # si la box testee est decalee ou si 2 cotes sont a l ecran (4D7 coin bas-droit).
    # Un corps/contenu n en a aucun. Mesure sur les 4 videos etalons :
    #   4D7 2/2 · eglV 5/5 · XzEg 0/6 (le fail total du 22h00, narrateur plein cadre).
    ok_sides = [v for v in usable if v[1] is not None and v[1] <= STD_MAX and v[0] >= HIT_STRAIGHT]
    return (len(ok_sides) >= 1), det
