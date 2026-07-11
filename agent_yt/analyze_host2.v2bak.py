import cv2, json, sys, os, statistics
import numpy as np

# DETECTEUR PIP v2 : YuNet (visage multi-echelle) + LIVE-MOTION.
# Une vraie webcam = visage DANS une region qui bouge (flux vivant).
# Une thumbnail/photo dans le contenu = visage STATIQUE -> rejete.
# Reutilise la machinerie temporelle de analyze_host.py (lissage/era-fill/split/merge/cleanup).
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
MOT_DT = 0.15; MOT_TH = 12; LIVE = 0.12
SAMPLE = 0.5

def frame_at(t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, f = cap.read(); return f if ok else None

def detect(t):
    """-> (host, bbox_norm|None, cx|None). bbox = rect webcam (live-motion autour du visage)."""
    f0 = frame_at(t)
    if f0 is None: return None
    f1 = frame_at(min(dur - 0.05, t + MOT_DT)); f1 = f1 if f1 is not None else f0
    g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY); g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    mot = (cv2.absdiff(g0, g1) > MOT_TH).astype(np.uint8) * 255
    motc = cv2.dilate(cv2.morphologyEx(mot, cv2.MORPH_CLOSE, K, iterations=2), K, iterations=1)
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(motc)
    _, faces = fd.detect(f0)
    if faces is None: return ('off', None, None)
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
        # borne le rect au voisinage du visage (anti-bridge vers du mouvement UI lointain)
        cx0 = max(0, fx - fw); cy0 = max(0, fy - fh)
        cx1 = min(W, fx + 2 * fw); cy1 = min(H, fy + 3 * fh)
        ax = max(mx, cx0); ay = max(my, cy0); ax2 = min(mx + mw, cx1); ay2 = min(my + mh, cy1)
        aw = ax2 - ax; ah = ay2 - ay
        if aw < 16 or ah < 16: continue
        live.append((ax, ay, aw, ah, fx + fw / 2))
    if not live: return ('off', None, None)
    ax, ay, aw, ah, fcx = max(live, key=lambda r: r[2] * r[3])  # dominant = plus grand rect live
    frac = aw * ah / (W * H); ccx = (ax + aw / 2) / W
    bbox = [round(ax / W, 4), round(ay / H, 4), round(aw / W, 4), round(ah / H, 4)]
    if frac >= 0.30 or (0.30 <= ccx <= 0.70 and frac >= 0.12):
        return ('hero', bbox, round(fcx / W, 4))
    return ('pip', bbox, round(fcx / W, 4))

raw = []; t = 0.0
while t < dur:
    r = detect(t)
    if r is None: t += SAMPLE; continue
    host, bbox, cx = r
    raw.append({'t': round(t, 2), 'host': host, 'bbox': bbox, 'cx': cx})
    t += SAMPLE
cap.release()
pip_n = sum(1 for r in raw if r['host'] == 'pip'); hero_n = sum(1 for r in raw if r['host'] == 'hero')
print("raw: %d samples (pip=%d hero=%d)" % (len(raw), pip_n, hero_n))

# ---- PASS 1: smoothing vote +-4 hero<->pip ----
hosts = [r['host'] for r in raw]; sm = list(hosts)
for i in range(len(hosts)):
    if hosts[i] in ('hero', 'pip'):
        win = [hosts[j] for j in range(max(0, i-4), min(len(hosts), i+5)) if hosts[j] in ('hero', 'pip')]
        if win: sm[i] = 'hero' if win.count('hero') >= win.count('pip') else 'pip'
for i, r in enumerate(raw): r['host'] = sm[i]

def med(L): return statistics.median(L)

# ---- PASS 2: PIP era-fill (cellule 4x4, comble micro-trous off) ----
pip_idx = [(i, r) for i, r in enumerate(raw) if r['host'] == 'pip' and r['bbox']]
cells = {}
for i, r in pip_idx:
    fx, fy, fw, fh = r['bbox']; cx_, cy_ = fx + fw/2, fy + fh/2
    cell = (min(3, int(cx_ * 4)), min(3, int(cy_ * 4)))
    cells.setdefault(cell, []).append((i, r))
for cell, items in cells.items():
    if len(items) < 2: continue
    idxs = [i for i, _ in items]; bxs = [r['bbox'] for _, r in items]
    i_first, i_last = min(idxs), max(idxs)
    mbb = [round(med([b[k] for b in bxs]), 4) for k in range(4)]
    last_real = i_first
    for j in range(i_first, i_last + 1):
        if j in idxs: last_real = j; continue
        if j - last_real > 4: continue
        if raw[j]['host'] == 'off': raw[j]['host'] = 'pip'; raw[j]['bbox'] = mbb

# ---- PASS 3: HERO era-fill (bridge <=5s off entre hero) ----
hero_idxs = [i for i, r in enumerate(raw) if r['host'] == 'hero']
for k in range(len(hero_idxs) - 1):
    a, b = hero_idxs[k], hero_idxs[k+1]
    if 1 < b - a <= 10 and all(raw[j]['host'] != 'pip' for j in range(a+1, b)):
        for j in range(a+1, b):
            if raw[j]['host'] == 'off': raw[j]['host'] = 'hero'; raw[j]['bbox'] = None

# ---- PASS 4: lone-off absorption ----
for i in range(1, len(raw) - 1):
    if raw[i]['host'] == 'off' and raw[i-1]['host'] == raw[i+1]['host'] and raw[i-1]['host'] in ('hero', 'pip'):
        raw[i]['host'] = raw[i-1]['host']
        if raw[i-1]['host'] == 'pip': raw[i]['bbox'] = raw[i-1]['bbox']

# ---- LISSAGE bbox pip: median glissant ----
WIN = 3
for i in [i for i, r in enumerate(raw) if r['host'] == 'pip' and r['bbox']]:
    neigh = [raw[j]['bbox'] for j in range(i-WIN, i+WIN+1) if 0 <= j < len(raw) and raw[j]['host'] == 'pip' and raw[j]['bbox']]
    if len(neigh) >= 2: raw[i]['bbox'] = [round(med([b[k] for b in neigh]), 4) for k in range(4)]

# ---- MERGE (split pip quand le centre saute = scene differente) ----
def center(bxs):
    return med([b[0]+b[2]/2 for b in bxs]), med([b[1]+b[3]/2 for b in bxs])
PIP_MOVE = 0.12
segs = []
if raw:
    ch = raw[0]['host']; cs = raw[0]['t']; bxs = [raw[0]['bbox']] if raw[0]['bbox'] else []
    for r in raw[1:]:
        split = False
        if r['host'] == ch == 'pip' and r['bbox'] and bxs:
            ccx, ccy = center(bxs)
            nx, ny = r['bbox'][0]+r['bbox'][2]/2, r['bbox'][1]+r['bbox'][3]/2
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

# ---- NETTOYAGE FINAL: drop pip aberrants + micro-segs <1s ----
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

json.dump(merged, open(os.path.join(out_dir, 'host_map.json'), 'w'), indent=2)
hc = sum(1 for s in merged if s['host'] == 'hero'); pc = sum(1 for s in merged if s['host'] == 'pip')
oc = sum(1 for s in merged if s['host'] == 'off')
print("host_map: %d segs hero=%d pip=%d off=%d" % (len(merged), hc, pc, oc))
for s in merged:
    if s['host'] == 'pip': print("  PIP %.1f-%.1fs bbox=%s" % (s['start'], s['end'], s['bbox']))
