#!/usr/bin/env python3
# qc_fid.py <workdir> <rendu.mp4> — JUGE DE FIDELITE + CLASSIFICATION.
#
# Boss 2026-07-28 : « arrete de tourner en rond, corrige l auto correction », puis
# « il le corrige pas encore ? alors arrange pour qu il le corrige ».
#
# Ni qc_ident (fuites d identite) ni qc_geom (sous-couverture seule, doctrine monotone)
# ne voyaient la SUR-couverture, les faux pips ni les heros rates. Ce juge les voit ET
# les rend corrigibles, en repondant a deux questions mesurables :
#   1. cardness  : le vert repose-t-il sur une VRAIE CARTE ? (un bord de carte est une DROITE)
#   2. narr_role : le narrateur est-il le SUJET de ce plan / de cette carte ?
#
# Les 4 cas de la doctrine Boss en decoulent, sans seuil de taille de visage fragile :
#   carte + narrateur sujet          -> PIP    : box = la carte mesuree  (TROP-GRAND/SOUS-COUV)
#   carte + narrateur non sujet      -> OFF    : contenu (photo, b-roll) -> image intacte
#   pas de carte + narrateur dominant-> HERO   : narrateur live plein cadre (XzEg)
#   pas de carte + sinon             -> OFF    : image intacte
#
# Non-oscillant : la cible est mesuree dans la SOURCE, invariante d un tour a l autre.
import json, os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cardness, narr_role

wd = sys.argv[1]
rendu = sys.argv[2]
hm = json.load(open(os.path.join(wd, 'host_map.json')))
cons = []
cp = os.path.join(wd, 'box_consensus.json')
if os.path.exists(cp):
    try: cons = json.load(open(cp))
    except Exception: cons = []

cr = cv2.VideoCapture(rendu)
cs = cv2.VideoCapture(os.path.join(wd, 'source.mp4'))
W = int(cs.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cs.get(cv2.CAP_PROP_FRAME_HEIGHT))
NR = narr_role.NarrRole(wd, W, H)

TOL_BIG = 0.020     # le vert deborde la carte de >2.0% de l ecran -> TROP-GRAND
TOL_SMALL = 0.015   # la carte deborde le vert de >1.5% -> SOUS-COUVERTURE
SUJET_MIN = 0.25    # faceH / hauteur de carte : pip webcam 0.38-0.47, contenu 0.00-0.11


def green_box(t):
    cr.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, f = cr.read()
    if not ok: return None
    b, g, r = f[:, :, 0].astype(int), f[:, :, 1].astype(int), f[:, :, 2].astype(int)
    m = ((g > 140) & (r < 110) & (b < 110)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2: return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = st[i]
    if a < 500: return None
    return [int(x), int(y), int(x + w), int(y + h)]


def _scan(line):
    if len(line) < 10: return None
    ref = line[:4].mean(axis=0)
    for i in range(4, len(line)):
        if np.linalg.norm(line[i] - ref) > 35: return i
    return None


def card_rect(gb, t0, t1):
    gw = gb[2] - gb[0]; gh = gb[3] - gb[1]
    fx0 = max(0, gb[0] - int(gw * 0.30)); fy0 = max(0, gb[1] - int(gh * 0.30))
    fx1 = min(W, gb[2] + int(gw * 0.30)); fy1 = min(H, gb[3] + int(gh * 0.30))
    fw = fx1 - fx0; fh = fy1 - fy0
    if fw < 16 or fh < 16: return None
    res = []
    for frac in (0.2, 0.35, 0.5, 0.65, 0.8):
        cs.set(cv2.CAP_PROP_POS_MSEC, (t0 + (t1 - t0) * frac) * 1000.0)
        ok, img = cs.read()
        if not ok: continue
        img = img.astype('float32')
        row = img[fy0 + fh // 2, fx0:fx1]; col = img[fy0:fy1, fx0 + fw // 2]
        L = 0 if fx0 == 0 else _scan(row[:fw // 2])
        R = 0 if fx1 == W else _scan(row[::-1][:fw // 2])
        T = 0 if fy0 == 0 else _scan(col[:fh // 2])
        B = 0 if fy1 == H else _scan(col[::-1][:fh // 2])
        if None in (L, R, T, B): continue
        res.append((L, R, T, B))
    if len(res) < 3: return None
    med = [sorted(r[i] for r in res)[len(res) // 2] for i in range(4)]
    for i in range(4):
        if max(abs(r[i] - med[i]) for r in res) > 0.10 * (fw if i < 2 else fh):
            return None
    L, R, T, B = med
    c = [fx0 + L, fy0 + T, fx1 - R, fy1 - B]
    if c[2] - c[0] < 0.35 * fw or c[3] - c[1] < 0.35 * fh: return None
    return c


def cons_match(card):
    if not cons: return None
    ccx = (card[0] + card[2]) / 2.0 / W; ccy = (card[1] + card[3]) / 2.0 / H
    best = None; bd = 9.0
    for c in cons:
        b = c['box']; d = abs(ccx - (b[0] + b[2] / 2)) + abs(ccy - (b[1] + b[3] / 2))
        if d < bd: best, bd = c, d
    return best if bd < 0.20 else None


def nb(r):
    return [round(r[0] / W, 4), round(r[1] / H, 4),
            round((r[2] - r[0]) / W, 4), round((r[3] - r[1]) / H, 4)]


fails = []
# --- controle des scenes HERO : un hero couvre TOUT l ecran, c est l action la plus
# destructrice du pipeline. Angle mort constate le 28/07 : le juge ne regardait que les
# pips, donc une video entierement mal classee en hero sortait « fid=OK ».
# On ne convertit que le cas CERTAIN : le narrateur est absent des 3 sondes ET
# quelqu un d AUTRE est le sujet du plan (visage dominant non-narrateur). Un hero ou le
# narrateur est de dos / hors champ reste intact (doute -> on ne touche pas).
for e in hm:
    if e['host'] != 'hero': continue
    t0, t1 = e['start'], e['end']
    pres, dom, fh = NR.probe(cs, t0, t1)
    if pres or dom: continue
    other = NR.other_subject(cs, t0, t1)
    if other:
        fails.append({'t0': t0, 't1': t1, 'type': 'FAUX-HERO', 'green': None,
                      'card': None, 'faceH': round(other, 3)})

for e in hm:
    if e['host'] != 'pip' or not e.get('bbox'): continue
    t0, t1 = e['start'], e['end']
    gb = green_box((t0 + t1) / 2.0)
    if gb is None: continue

    ok_card, det = cardness.card_score(cs, gb, t0, t1, W, H)

    if not ok_card:
        # aucune carte sous le vert : soit narrateur LIVE plein cadre (hero rate),
        # soit du contenu qu on n aurait jamais du toucher.
        # GARDE (28/07) : un HERO couvre TOUT l ecran. Si le vert est petit, ce n est
        # pas un hero rate — c est un pip dont cardness a rate les bords parce qu il
        # teste le VERT et non la carte. Sans cette garde, eglV 42.6-906.6 (15 minutes
        # de pip correct, vert 15% de l ecran) partait en hero plein ecran.
        ga = (gb[2] - gb[0]) * (gb[3] - gb[1]) / float(W * H)
        pres, dom, fh = NR.probe(cs, t0, t1)
        if ga < 0.25:
            if not pres: 
                fails.append({'t0': t0, 't1': t1, 'type': 'FAUX-PIP', 'green': nb(gb),
                              'card': None, 'faceH': round(fh, 3), 'dominant': bool(dom)})
            continue
        typ = 'HERO-RATE' if dom else 'FAUX-PIP'
        fails.append({'t0': t0, 't1': t1, 'type': typ, 'green': nb(gb),
                      'card': None, 'faceH': round(fh, 3), 'dominant': bool(dom)})
        continue

    card = card_rect(gb, t0, t1)
    if card is None: continue

    # PREUVE BORD PAR BORD (28/07). qc_fid_fix pose la carte mesuree en AUTORITE
    # (patched_keep, non monotone : elle peut AGRANDIR le vert). Une mesure fausse est
    # donc gravee et devient invisible a tous les juges. Mesure du 28/07 sur
    # mCE 13.0-20.12 : card_rect annonce 0.236 de large la ou le bord reel de la carte
    # est a 0.180 (+31%), et cardness sur cette meme carte donne hit=0.30 a droite —
    # le juge pouvait s invalider lui-meme. On exige donc que chaque cote NON colle au
    # bord de l ecran soit une VRAIE DROITE avant d emettre le moindre verdict.
    # Conservateur par construction : un cote non prouve = aucun verdict = image inchangee.
    _, cdet = cardness.card_score(cs, card, t0, t1, W, H)
    bad = [s for s, v in cdet.items()
           if v is not None and (v[1] is None or v[1] > cardness.STD_MAX
                                 or v[0] < cardness.HIT_STRAIGHT)]
    if bad:
        print('  t=%.1f-%.1f carte %s REJETEE : cote(s) %s pas une droite (%s)'
              % (t0, t1, nb(card), ','.join(sorted(bad)),
                 ' '.join('%s=%s' % (s, cdet[s]) for s in sorted(bad))))
        continue

    # le narrateur est-il le SUJET de cette carte ? (pip webcam 0.38-0.47 de la hauteur
    # de carte ; photo/b-roll ou il apparait 0.00-0.11)
    pres, dom, fh = NR.probe(cs, t0, t1, rect=card)
    ch = (card[3] - card[1]) / float(H)
    if ch > 0 and (fh / ch) < SUJET_MIN:
        fails.append({'t0': t0, 't1': t1, 'type': 'PAS-NARRATEUR', 'green': nb(gb),
                      'card': nb(card), 'faceH': round(fh, 3), 'ratio': round(fh / ch, 2)})
        continue

    cm = cons_match(card)
    if cm is not None:
        ca = (card[2] - card[0]) * (card[3] - card[1]) / float(W * H)
        ka = cm['box'][2] * cm['box'][3]
        # garde resserree a 1.8 : une carte mesuree qui contredit le consensus en aire
        # d un facteur >1.8 est un artefact de scan (eglV t=7.6 mesurait 0.432 de large
        # contre 0.227 au consensus). Sous 1.8 on laisse passer les vraies variations
        # de layout (intro eglV, carte collee aux bords : 1.6).
        if ka > 0 and (ca / ka > 1.8 or ka / ca > 1.8): continue
    over = [(card[0] - gb[0]) / W, (card[1] - gb[1]) / H,
            (gb[2] - card[2]) / W, (gb[3] - card[3]) / H]
    big = max(over); small = max(-v for v in over)
    typ = 'TROP-GRAND' if big > TOL_BIG else ('SOUS-COUVERTURE' if small > TOL_SMALL else None)
    if typ:
        fails.append({'t0': t0, 't1': t1, 'type': typ, 'green': nb(gb), 'card': nb(card),
                      'ecart_pct': [round(v * 100, 1) for v in over]})

json.dump(fails, open(os.path.join(wd, 'qc_fid.json'), 'w'), indent=1)
from collections import Counter
c = Counter(f['type'] for f in fails)
print('QC-FID ECHECS : %d  %s' % (len(fails), dict(c)))
for f in fails[:40]:
    print('  t=%.1f-%.1f %-15s vert=%s carte=%s %s'
          % (f['t0'], f['t1'], f['type'], f['green'], f.get('card'),
             f.get('ecart_pct') or ('faceH=%s dom=%s' % (f.get('faceH'), f.get('dominant')))))
sys.exit(1 if fails else 0)
