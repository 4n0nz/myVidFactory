#!/usr/bin/env python3
# qc_fid.py <workdir> <rendu.mp4> — JUGE DE FIDELITE (le juge qui manquait).
#
# Pourquoi (Boss, 2026-07-28 « arrete de tourner en rond, corrige l auto correction ») :
#   - qc_ident ne voit que les fuites d IDENTITE (un visage a decouvert).
#   - qc_geom ne corrige QUE la sous-couverture (doctrine monotone anti-oscillation).
#   => la SUR-couverture et les faux pips n avaient AUCUN juge : Boss devait les voir a
#      l oeil, passe apres passe. C est cette boucle qu on casse.
#
# Non-oscillant par construction : la cible est mesuree dans la SOURCE, qui ne change
# jamais d un tour a l autre. L oscillation historique (gb5 LEAK_242) venait de mesurer
# sur le RENDU, qui bouge a chaque tour. Cible fixe => convergence.
#
# Trois verdicts :
#   TROP-GRAND      le vert deborde la carte           -> corrigible (box = carte)
#   SOUS-COUVERTURE la carte deborde le vert           -> corrigible (box = carte)
#   PAS-DE-CARTE    aucun bord de carte sous le vert   -> NON corrigible ici, mais
#                   signale : c est un faux pip (contenu) ou un hero rate (XzEg).
import json, os, sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cardness

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

TOL_BIG = 0.020
TOL_SMALL = 0.015


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
    """bords REELS de la carte dans la SOURCE. Un cote colle au bord de l ecran n est
    pas scannable : la carte y touche (eglV intro). None si non reproductible."""
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


fails = []
for e in hm:
    if e['host'] != 'pip' or not e.get('bbox'): continue
    t = (e['start'] + e['end']) / 2.0
    gb = green_box(t)
    if gb is None: continue

    # 1) le vert repose-t-il sur une VRAIE carte ? (cardness : un bord de carte est
    #    une DROITE ; un corps/contenu n en a aucune)
    ok_card, det = cardness.card_score(cs, gb, e['start'], e['end'], W, H)
    if not ok_card:
        fails.append({'t0': e['start'], 't1': e['end'], 'type': 'PAS-DE-CARTE',
                      'green': [round(gb[0] / W, 4), round(gb[1] / H, 4),
                                round((gb[2] - gb[0]) / W, 4), round((gb[3] - gb[1]) / H, 4)],
                      'card': None, 'ecart_pct': None, 'bords': str(det)})
        continue

    # 2) le vert coincide-t-il avec la carte ?
    card = card_rect(gb, e['start'], e['end'])
    if card is None: continue
    cm = cons_match(card)
    if cm is not None:
        ca = (card[2] - card[0]) * (card[3] - card[1]) / float(W * H)
        ka = cm['box'][2] * cm['box'][3]
        if ka > 0 and (ca / ka > 2.2 or ka / ca > 2.2): continue
    over = [(card[0] - gb[0]) / W, (card[1] - gb[1]) / H,
            (gb[2] - card[2]) / W, (gb[3] - card[3]) / H]
    big = max(over); small = max(-v for v in over)
    typ = 'TROP-GRAND' if big > TOL_BIG else ('SOUS-COUVERTURE' if small > TOL_SMALL else None)
    if typ:
        fails.append({'t0': e['start'], 't1': e['end'], 'type': typ,
                      'green': [round(gb[0] / W, 4), round(gb[1] / H, 4),
                                round((gb[2] - gb[0]) / W, 4), round((gb[3] - gb[1]) / H, 4)],
                      'card': [round(card[0] / W, 4), round(card[1] / H, 4),
                               round((card[2] - card[0]) / W, 4), round((card[3] - card[1]) / H, 4)],
                      'ecart_pct': [round(v * 100, 1) for v in over]})

json.dump(fails, open(os.path.join(wd, 'qc_fid.json'), 'w'), indent=1)
nc = sum(1 for f in fails if f['type'] == 'PAS-DE-CARTE')
print('QC-FID ECHECS : %d  (dont PAS-DE-CARTE : %d)' % (len(fails), nc))
for f in fails:
    if f['type'] == 'PAS-DE-CARTE':
        print('  t=%.1f-%.1f PAS-DE-CARTE vert=%s bords=%s' % (f['t0'], f['t1'], f['green'], f['bords']))
    else:
        print('  t=%.1f-%.1f %s vert=%s carte=%s ecart%%(L,T,R,B)=%s'
              % (f['t0'], f['t1'], f['type'], f['green'], f['card'], f['ecart_pct']))
sys.exit(1 if fails else 0)
