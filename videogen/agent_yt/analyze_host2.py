import cv2, json, sys, os, statistics
import numpy as np

# DETECTEUR PIP v2.2 — DEUX PASSES.
# PASS A = classification V2.0 (YuNet + live-motion) -> segments STABLES (motion-box bbox).
# PASS B = etendue exacte par segment : edge-scan de la bordure sur ~6 frames -> MEDIANE.
#   Gere border ET "coupe sec" (les deux = un gradient a la frontiere). Edge faible -> fallback
#   sur la box mouvement (= comportement V2.0). Un seul rectangle stable/segment -> pas de jitter.
video_path = sys.argv[1]
out_dir = os.path.dirname(os.path.abspath(video_path))
YUNET = '/home/boss/videogen/face_detection_yunet_2023mar.onnx'

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(3)); H = int(cap.get(4)); dur = total / fps
print("Video: %.1fs %dx%d" % (dur, W, H))

fd = cv2.FaceDetectorYN.create(YUNET, "", (W, H), score_threshold=0.5, nms_threshold=0.3, top_k=5000)
fd.setInputSize((W, H))
K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
MOT_DT = 0.15; MOT_TH = 12; LIVE = 0.12; SAMPLE = 0.5; EDGE_TH = 18; MARGIN = 6

def frame_at(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, f = cap.read(); return f if ok else None

def motion_and_faces(t):
    f0 = frame_at(t)
    if f0 is None: return None
    f1 = frame_at(min(dur - 0.05, t + MOT_DT)); f1 = f1 if f1 is not None else f0
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    mot = (cv2.absdiff(g0, g1) > MOT_TH).astype(np.uint8) * 255
    motc = cv2.dilate(cv2.morphologyEx(mot, cv2.MORPH_CLOSE, K, iterations=2), K, iterations=1)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(motc)
    _, faces = fd.detect(f0)
    return f0, g0, motc, lab, stats, (faces if faces is not None else [])

# ---------- PASS A : classification V2.0 (motion-box bbox, stable) ----------
def detect(t):
    r = motion_and_faces(t)
    if r is None: return None
    f0, g0, motc, lab, stats, faces = r
    live = []
    for fc in faces:
        fx, fy, fw, fh = int(fc[0]), int(fc[1]), int(fc[2]), int(fc[3])
        if fw < 8 or fh < 8: continue
        cx = min(W - 1, max(0, fx + fw // 2)); cy = min(H - 1, max(0, fy + fh // 2))
        rx0 = max(0, fx - fw // 2); ry0 = max(0, fy - fh // 2)
        rx1 = min(W, fx + fw + fw // 2); ry1 = min(H, fy + fh + 2 * fh)
        fill = motc[ry0:ry1, rx0:rx1].mean() / 255.0
        if fill < LIVE: continue
        lid = int(lab[cy, cx])
        if lid > 0:
            mx, my, mw, mh = int(stats[lid, 0]), int(stats[lid, 1]), int(stats[lid, 2]), int(stats[lid, 3])
        else:
            mx, my, mw, mh = fx, fy, fw, fh
        cx0 = max(0, fx - fw); cy0 = max(0, fy - fh); cx1 = min(W, fx + 2 * fw); cy1 = min(H, fy + 3 * fh)
        ax = max(mx, cx0); ay = max(my, cy0); ax2 = min(mx + mw, cx1); ay2 = min(my + mh, cy1)
        aw = ax2 - ax; ah = ay2 - ay
        if aw < 16 or ah < 16: continue
        live.append((ax, ay, aw, ah, fx + fw / 2, fw, fh))
    if not live: return ('off', None, None, 0.0)
    # dominant = le plus GROS VISAGE (le narrateur plein ecran gagne sur un pip de coin)
    ax, ay, aw, ah, fcx, fw, fh = max(live, key=lambda r: r[5] * r[6])
    frac = aw * ah / (W * H); ccx = (ax + aw / 2) / W
    box_w = aw / W
    # face_big => hero SEULEMENT si la box remplit l'ecran (large OU centree) ;
    # un gros visage dans une box etroite de coin = pip, pas hero (fix 2026-07-07 vid 1DOLq)
    face_big = (fh / H) > 0.18 and (box_w > 0.55 or 0.25 <= ccx <= 0.75)
    bbox = [round(ax / W, 4), round(ay / H, 4), round(aw / W, 4), round(ah / H, 4)]
    if frac >= 0.30 or (0.30 <= ccx <= 0.70 and frac >= 0.12) or face_big:
        return ('hero', bbox, round(fcx / W, 4), round(fh / H, 4))
    return ('pip', bbox, round(fcx / W, 4), round(fh / H, 4))

# ---------- PASS B : etendue exacte (edge-scan) ----------
def strongest(prof, a, b):
    a = max(0, a); b = min(len(prof), b)
    if b <= a: return None, 0.0
    seg = prof[a:b]; i = int(np.argmax(seg)); return a + i, float(seg[i])

def extent_at(t):
    """Rectangle webcam exact a l'instant t (edge-scan autour de la region live). None si pas de pip."""
    r = motion_and_faces(t)
    if r is None: return None
    f0, g0, motc, lab, stats, faces = r
    cand = None
    for fc in faces:
        fx, fy, fw, fh = int(fc[0]), int(fc[1]), int(fc[2]), int(fc[3])
        if fw < 8 or fh < 8: continue
        cx = min(W - 1, max(0, fx + fw // 2)); cy = min(H - 1, max(0, fy + fh // 2))
        rx0 = max(0, fx - fw // 2); ry0 = max(0, fy - fh // 2)
        rx1 = min(W, fx + fw + fw // 2); ry1 = min(H, fy + fh + 2 * fh)
        if motc[ry0:ry1, rx0:rx1].mean() / 255.0 < LIVE: continue
        lid = int(lab[cy, cx])
        if lid > 0:
            mx, my, mw, mh = int(stats[lid, 0]), int(stats[lid, 1]), int(stats[lid, 2]), int(stats[lid, 3])
        else:
            mx, my, mw, mh = fx, fy, fw, fh
        if cand is None or mw * mh > cand[2] * cand[3]:
            cand = (mx, my, mw, mh)
    if cand is None: return None
    mx, my, mw, mh = cand
    agx = np.abs(cv2.Sobel(g0, cv2.CV_32F, 1, 0, ksize=3))
    agy = np.abs(cv2.Sobel(g0, cv2.CV_32F, 0, 1, ksize=3))
    ys, ye, xs, xe = my, my + mh, mx, mx + mw
    SL = max(0, mx - int(1.5 * mw)); ST = max(0, my - int(1.2 * mh))
    SR = min(W, mx + mw + int(0.5 * mw)); SB = min(H, my + mh + int(0.4 * mh))
    profx = agx[ys:ye, :].sum(axis=0) / max(1, ye - ys)
    profy = agy[:, xs:xe].sum(axis=1) / max(1, xe - xs)
    L, lv = strongest(profx, SL, mx + 1); R, rv = strongest(profx, mx + mw, SR)
    Tp, tv = strongest(profy, ST, my + 1); B, bv = strongest(profy, my + mh, SB)
    # bordure si edge fort ; sinon snap frame si on touche le bord, sinon fallback modere
    L = L if (L is not None and lv > EDGE_TH) else (0 if SL == 0 else mx - int(1.0 * mw))
    R = R if (R is not None and rv > EDGE_TH) else (W if SR == W else mx + mw + int(0.35 * mw))
    Tp = Tp if (Tp is not None and tv > EDGE_TH) else (0 if ST == 0 else my - int(0.8 * mh))
    B = B if (B is not None and bv > EDGE_TH) else (H if SB == H else my + mh + int(0.25 * mh))
    L = max(0, min(L, mx)); R = min(W, max(R, mx + mw)); Tp = max(0, min(Tp, my)); B = min(H, max(B, my + mh))
    return [max(0, L - MARGIN), max(0, Tp - MARGIN), min(W, R + MARGIN), min(H, B + MARGIN)]

# ---------- PASS A run ----------
raw = []; t = 0.0
while t < dur:
    r = detect(t)
    if r is None: t += SAMPLE; continue
    host, bbox, cx, fhH = r
    raw.append({'t': round(t, 2), 'host': host, 'bbox': bbox, 'cx': cx, 'fhH': fhH})
    t += SAMPLE
pip_n = sum(1 for r in raw if r['host'] == 'pip'); hero_n = sum(1 for r in raw if r['host'] == 'hero')
print("raw: %d samples (pip=%d hero=%d)" % (len(raw), pip_n, hero_n))

def med(L): return statistics.median(L)

hosts = [r['host'] for r in raw]; sm = list(hosts)
# un vrai plan plein-ecran = gros visage centre : on le VERROUILLE hero. Le lissage median
# effacait les cutaways hero courts noyes dans un long regime pip (narrateur decouvert,
# ex. ofr 1:54). Cle sur le VISAGE (fhH/fcx), pas la box mouvement. NO_HEROLOCK=1 = ancien.
HEROLOCK = os.environ.get('NO_HEROLOCK') != '1'
STRONG_H = float(os.environ.get('STRONG_HERO', '0.30'))
MINRUN = int(os.environ.get('HEROLOCK_MINRUN', '3'))
def _strong_hero(r): return r['host'] == 'hero' and r.get('fhH', 0) > STRONG_H and 0.28 <= (r['cx'] or 0) <= 0.72
# min-run : un sample strong ISOLE (decoy, frame de cross-dissolve) reste lisse ;
# seuls les runs de >=MINRUN samples consecutifs (>=1.5s) sont verrouilles
# (vrai cutaway plein ecran >=1.5s ; faux positifs YuNet vus jusqu a 2 samples consecutifs, cf Ethx trou noir 431s).
_strong = [_strong_hero(r) for r in raw]
_lockmask = [False] * len(raw)
_i = 0
while _i < len(raw):
    if _strong[_i]:
        _j = _i
        while _j < len(raw) and _strong[_j]: _j += 1
        if _j - _i >= MINRUN:
            for _k in range(_i, _j): _lockmask[_k] = True
        _i = _j
    else: _i += 1
locked = 0
for i in range(len(hosts)):
    if HEROLOCK and _lockmask[i]:
        sm[i] = 'hero'; locked += 1; continue
    if hosts[i] in ('hero', 'pip'):
        win = [hosts[j] for j in range(max(0, i-4), min(len(hosts), i+5)) if hosts[j] in ('hero', 'pip')]
        if win: sm[i] = 'hero' if win.count('hero') >= win.count('pip') else 'pip'
for i, r in enumerate(raw): r['host'] = sm[i]
print('smoothing: %d strong-hero verrouilles (gros visage centre)' % locked)

pip_idx = [(i, r) for i, r in enumerate(raw) if r['host'] == 'pip' and r['bbox']]
cells = {}
for i, r in pip_idx:
    fx, fy, fw, fh = r['bbox']; cx_, cy_ = fx + fw/2, fy + fh/2
    cell = (min(3, int(cx_ * 4)), min(3, int(cy_ * 4)))
    cells.setdefault(cell, []).append((i, r))
for cell, items in cells.items():
    if len(items) < 2: continue
    idxs = [i for i, _ in items]; bxs = [r['bbox'] for _, r in items]
    i_first, i_last = min(idxs), max(idxs); mbb = [round(med([b[k] for b in bxs]), 4) for k in range(4)]
    last_real = i_first
    for j in range(i_first, i_last + 1):
        if j in idxs: last_real = j; continue
        if j - last_real > 4: continue
        if raw[j]['host'] == 'off': raw[j]['host'] = 'pip'; raw[j]['bbox'] = mbb

hero_idxs = [i for i, r in enumerate(raw) if r['host'] == 'hero']
for k in range(len(hero_idxs) - 1):
    a, b = hero_idxs[k], hero_idxs[k+1]
    if 1 < b - a <= 10 and all(raw[j]['host'] != 'pip' for j in range(a+1, b)):
        for j in range(a+1, b):
            if raw[j]['host'] == 'off': raw[j]['host'] = 'hero'; raw[j]['bbox'] = None

for i in range(1, len(raw) - 1):
    if raw[i]['host'] == 'off' and raw[i-1]['host'] == raw[i+1]['host'] and raw[i-1]['host'] in ('hero', 'pip'):
        raw[i]['host'] = raw[i-1]['host']
        if raw[i-1]['host'] == 'pip': raw[i]['bbox'] = raw[i-1]['bbox']

WIN = 3
for i in [i for i, r in enumerate(raw) if r['host'] == 'pip' and r['bbox']]:
    neigh = [raw[j]['bbox'] for j in range(i-WIN, i+WIN+1) if 0 <= j < len(raw) and raw[j]['host'] == 'pip' and raw[j]['bbox']]
    if len(neigh) >= 2: raw[i]['bbox'] = [round(med([b[k] for b in neigh]), 4) for k in range(4)]

def center(bxs): return med([b[0]+b[2]/2 for b in bxs]), med([b[1]+b[3]/2 for b in bxs])
PIP_MOVE = 0.12
segs = []
if raw:
    ch = raw[0]['host']; cs = raw[0]['t']; bxs = [raw[0]['bbox']] if raw[0]['bbox'] else []
    for r in raw[1:]:
        split = False
        if r['host'] == ch == 'pip' and r['bbox'] and bxs:
            ccx, ccy = center(bxs); nx, ny = r['bbox'][0]+r['bbox'][2]/2, r['bbox'][1]+r['bbox'][3]/2
            if abs(nx-ccx) > PIP_MOVE or abs(ny-ccy) > PIP_MOVE: split = True
        if r['host'] == ch and not split:
            if r['bbox']: bxs.append(r['bbox'])
        else:
            avg = [round(med([b[i] for b in bxs]), 4) for i in range(4)] if bxs else None
            segs.append({'start': cs, 'end': round(r['t'], 2), 'host': ch, 'bbox': avg})
            ch = r['host']; cs = r['t']; bxs = [r['bbox']] if r['bbox'] else []
    avg = [round(med([b[i] for b in bxs]), 4) for i in range(4)] if bxs else None
    segs.append({'start': cs, 'end': round(dur, 2), 'host': ch, 'bbox': avg})

merged = []
for s in segs:
    if merged and s['host'] != 'off' and merged[-1]['host'] == s['host'] and s['start'] - merged[-1]['end'] < 2.0:
        if s['host'] == 'pip' and s['bbox'] and merged[-1]['bbox']:
            pcx = merged[-1]['bbox'][0]+merged[-1]['bbox'][2]/2; pcy = merged[-1]['bbox'][1]+merged[-1]['bbox'][3]/2
            ncx = s['bbox'][0]+s['bbox'][2]/2; ncy = s['bbox'][1]+s['bbox'][3]/2
            if abs(ncx-pcx) > PIP_MOVE or abs(ncy-pcy) > PIP_MOVE: merged.append(dict(s)); continue
        merged[-1]['end'] = s['end']
    else: merged.append(dict(s))

clean = []
for s in merged:
    if s['host'] == 'pip' and s['bbox']:
        bx, by, bw, bh = s['bbox']; frac = bw*bh
        if frac > 0.45 or bw > 0.55 or bw < 0.04 or bh < 0.04:
            s = {'start': s['start'], 'end': s['end'], 'host': 'off', 'bbox': None}
    if s['host'] == 'pip' and (s['end']-s['start']) < 1.0:
        s = {'start': s['start'], 'end': s['end'], 'host': 'off', 'bbox': None}
    clean.append(s)
merged2 = []
for s in clean:
    if merged2 and merged2[-1]['host'] == s['host'] == 'off': merged2[-1]['end'] = s['end']
    elif (merged2 and merged2[-1]['host'] == s['host'] == 'pip' and s['bbox'] and merged2[-1]['bbox']
          and abs((s['bbox'][0]+s['bbox'][2]/2)-(merged2[-1]['bbox'][0]+merged2[-1]['bbox'][2]/2)) <= PIP_MOVE
          and abs((s['bbox'][1]+s['bbox'][3]/2)-(merged2[-1]['bbox'][1]+merged2[-1]['bbox'][3]/2)) <= PIP_MOVE
          and s['start']-merged2[-1]['end'] < 2.0): merged2[-1]['end'] = s['end']
    else: merged2.append(dict(s))
merged = merged2

# ---------- PASS B : etendue exacte par segment (mediane des edge-scans) ----------
refined = 0; skipped = 0
for s in merged:
    if s['host'] != 'pip' or not s['bbox']: continue
    a, b = s['start'], s['end']
    ts = [a + (b - a) * f for f in (0.2, 0.35, 0.5, 0.65, 0.8)]
    rects = [extent_at(t) for t in ts]
    rects = [r for r in rects if r]
    if len(rects) >= 3:
        L = med([r[0] for r in rects]); T = med([r[1] for r in rects])
        R = med([r[2] for r in rects]); B = med([r[3] for r in rects])
        nb = [round(L / W, 4), round(T / H, 4), round((R - L) / W, 4), round((B - T) / H, 4)]
        # garde-fou : une webcam pip n'est jamais pleine largeur. Si l'edge-scan derape
        # (screen-share : le contenu en mouvement fusionne avec la cam -> blob full-width,
        # pas de bordure nette -> fallback L=0/R=W), on JETTE le raffinement et on garde
        # la box Pass A (ancree-visage, clampee a ~3*fw = correcte, cf t=700). Sinon over-cover.
        if nb[2] <= 0.55 and nb[2] * nb[3] <= 0.45:
            s['bbox'] = nb
            refined += 1
        else:
            skipped += 1
print("PASS B: %d pip segments raffines (edge-scan median), %d jetes (full-width -> box Pass A)" % (refined, skipped))

# ---------- PASS C : avaler les cross-dissolve de la source ----------
# La source fond-enchaine entre scenes : le narrateur plein ecran apparait EN FONDU
# avant la frontiere hero -> son vrai visage fuiterait ~1s (avatar encore en pip coin).
# On etend chaque hero de HERO_PAD des deux cotes en rognant les voisins non-hero,
# pour que l'avatar plein ecran couvre le narrateur des qu'il commence a apparaitre.
HERO_PAD = 1.0; MIN_SEG = 0.3
for i, s in enumerate(merged):
    if s['host'] != 'hero': continue
    if i > 0 and merged[i-1]['host'] != 'hero':
        prev = merged[i-1]
        ns = max(prev['start'], round(s['start'] - HERO_PAD, 2))
        if ns - prev['start'] < MIN_SEG: ns = prev['start']   # absorbe le voisin trop court
        s['start'] = ns; prev['end'] = ns
    if i < len(merged) - 1 and merged[i+1]['host'] != 'hero':
        nxt = merged[i+1]
        ne = min(nxt['end'], round(s['end'] + HERO_PAD, 2))
        if nxt['end'] - ne < MIN_SEG: ne = nxt['end']
        s['end'] = ne; nxt['start'] = ne
merged = [s for s in merged if s['end'] - s['start'] > 0.05]
print("PASS C: hero elargis +-%.1fs (avale les cross-dissolve source)" % HERO_PAD)

json.dump(merged, open(os.path.join(out_dir, 'host_map.json'), 'w'), indent=2)
hc = sum(1 for s in merged if s['host'] == 'hero'); pc = sum(1 for s in merged if s['host'] == 'pip')
oc = sum(1 for s in merged if s['host'] == 'off')
print("host_map: %d segs hero=%d pip=%d off=%d" % (len(merged), hc, pc, oc))
for s in merged:
    if s['host'] == 'pip': print("  PIP %.1f-%.1fs bbox=%s" % (s['start'], s['end'], s['bbox']))
cap.release()
