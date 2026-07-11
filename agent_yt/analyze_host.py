import cv2, json, sys, os, statistics
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as P
from mediapipe.tasks.python import vision as V

video_path = sys.argv[1]
out_dir = os.path.dirname(os.path.abspath(video_path))
MODEL = '/home/boss/videogen/efficientdet.tflite'

# ---------- PIP FRAME DETECTOR (glow orange de la source) ----------
# 2 etages: NEON strict DECIDE qu'il y a un cadre (vs peau/bureau/cartoon mats),
# masque LARGE mesure l'etendue complete du cadre (le glow s'attenue en haut).
_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
def _closed(mask):
    return cv2.dilate(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K, iterations=4), _K, iterations=1)

def detect_pip_frame(bgr):
    H, W = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    strict = cv2.inRange(hsv, (14, 120, 180), (40, 255, 255))   # neon glow seul
    loose  = cv2.inRange(hsv, (10, 70, 110),  (42, 255, 255))   # tout l'orange
    cnts, _ = cv2.findContours(_closed(strict), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w*h; frac = area/(W*H); ar = w/h if h else 0
        edge = (x <= 0.15*W or x+w >= 0.85*W or y <= 0.15*H or y+h >= 0.85*H)
        if not (0.03 <= frac <= 0.45 and 0.35 <= ar <= 1.8 and edge):
            continue
        inset = int(min(w, h)*0.22)
        if inset <= 2 or w-2*inset < 4 or h-2*inset < 4:
            continue
        int_fill = strict[y+inset:y+h-inset, x+inset:x+w-inset].mean()/255.0
        if int_fill < 0.55 and (cand is None or area > cand[4]):
            cand = (x, y, w, h, area)
    if cand is None:
        return None
    sx, sy, sw, sh, _ = cand
    cx, cy = sx+sw//2, sy+sh//2
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(_closed(loose))
    lid = lab[min(H-1, cy), min(W-1, cx)]
    if lid == 0:
        sub = lab[sy:sy+sh, sx:sx+sw]
        vals, cc = np.unique(sub[sub > 0], return_counts=True)
        lid = vals[np.argmax(cc)] if len(vals) else 0
    if lid > 0:
        cxp, cyp, cwp, chp = int(stats[lid, 0]), int(stats[lid, 1]), int(stats[lid, 2]), int(stats[lid, 3])
        # ETENDUE BORNEE autour du strict: grandir vers la composante loose mais
        # jamais plus de 0.45x le strict de chaque cote (anti-bridge vers contenu chaud voisin).
        mx, my = int(0.45*sw), int(0.45*sh)
        ex = min(sx, max(cxp, sx-mx))
        ey = min(sy, max(cyp, sy-my))
        ex2 = max(sx+sw, min(cxp+cwp, sx+sw+mx))
        ey2 = max(sy+sh, min(cyp+chp, sy+sh+my))
        ex, ey, ew, eh = ex, ey, ex2-ex, ey2-ey
    else:
        ex, ey, ew, eh = sx, sy, sw, sh
    # REJET: un vrai pip est un petit insert de coin. Une etendue trop grande
    # = scene plein-ecran chaude (lampes, b-roll) faussement captee -> pas un pip.
    if (ew*eh)/(W*H) > 0.38:
        return None
    return [round(ex/W, 4), round(ey/H, 4), round(ew/W, 4), round(eh/H, 4)]

# ---------- PERSON DETECTOR (hero plein ecran) ----------
det = V.ObjectDetector.create_from_options(V.ObjectDetectorOptions(
    base_options=P.BaseOptions(model_asset_path=MODEL),
    score_threshold=0.22, category_allowlist=['person'], max_results=5))

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(3)); H = int(cap.get(4)); dur = total / fps
print("Video: %.1fs %dx%d" % (dur, W, H))

SAMPLE = 0.5
raw = []
t = 0.0
while t < dur:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, fr = cap.read()
    if not ok:
        t += SAMPLE; continue
    # 1) PIP frame glow orange (style de CETTE source)
    pip_bb = detect_pip_frame(fr)
    # detection personne TOUJOURS (pour reconcilier un faux cadre orange = lumiere chaude/sunset)
    rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
    res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    person = None
    if res.detections:
        b = max(res.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height).bounding_box
        fx = max(0.0, b.origin_x / W); fy = max(0.0, b.origin_y / H)
        fw = min(b.width / W, 1 - fx); fh = min(b.height / H, 1 - fy)
        person = (fx, fy, fw, fh)

    def _hero_like(p):
        px, py, pw, ph = p; a = pw * ph; c = px + pw / 2
        return a >= 0.28 or (0.28 <= c <= 0.72 and a >= 0.06)

    if pip_bb:
        # REJET faux cadre orange: si une grande personne (hero) est HORS de la boite orange,
        # c'est le narrateur plein cadre + lumiere chaude, pas un vrai insert -> hero.
        if person and _hero_like(person):
            pcx, pcy = person[0] + person[2] / 2, person[1] + person[3] / 2
            bx, by, bw, bh = pip_bb
            if not (bx <= pcx <= bx + bw and by <= pcy <= by + bh):
                fx, fy, fw, fh = person
                raw.append({'t': round(t, 2), 'host': 'hero',
                            'bbox': [round(fx, 4), round(fy, 4), round(fw, 4), round(fh, 4)],
                            'cx': round(fx + fw / 2, 4), 'src': 'reconcile'})
                t += SAMPLE; continue
        raw.append({'t': round(t, 2), 'host': 'pip', 'bbox': pip_bb, 'cx': None, 'src': 'frame'})
        t += SAMPLE; continue
    # 2) FALLBACK general (sources SANS glow): person -> hero(grand/centre) / pip(coin) / off
    if person:
        fx, fy, fw, fh = person
        area = fw * fh; cx = fx + fw / 2
        if area >= 0.28 or (0.28 <= cx <= 0.72 and area >= 0.06):
            host = 'hero'
        elif 0.015 <= area < 0.22:
            host = 'pip'
        else:
            host = 'off'
        raw.append({'t': round(t, 2), 'host': host,
                    'bbox': [round(fx, 4), round(fy, 4), round(fw, 4), round(fh, 4)] if host in ('hero', 'pip') else None,
                    'cx': round(cx, 4), 'src': 'person'})
    else:
        raw.append({'t': round(t, 2), 'host': 'off', 'bbox': None, 'cx': None, 'src': None})
    t += SAMPLE
cap.release(); det.close()
pip_n = sum(1 for r in raw if r['host'] == 'pip')
print("raw: %d samples  (pip-frame hits=%d)" % (len(raw), pip_n))

# ---------- PASS 1: smoothing vote +-4 hero<->pip (sans toucher off) ----------
hosts = [r['host'] for r in raw]
sm = list(hosts)
for i in range(len(hosts)):
    if hosts[i] in ('hero', 'pip'):
        win = [hosts[j] for j in range(max(0, i-4), min(len(hosts), i+5)) if hosts[j] in ('hero', 'pip')]
        if win:
            sm[i] = 'hero' if win.count('hero') >= win.count('pip') else 'pip'
for i, r in enumerate(raw):
    r['host'] = sm[i]

# ---------- PASS 2: PIP era-fill (cluster cellule 4x4, comble off) ----------
def med(L): return statistics.median(L)
pip_idx = [(i, r) for i, r in enumerate(raw) if r['host'] == 'pip' and r['bbox']]
cells = {}
for i, r in pip_idx:
    fx, fy, fw, fh = r['bbox']; cx_, cy_ = fx + fw/2, fy + fh/2
    cell = (min(3, int(cx_ * 4)), min(3, int(cy_ * 4)))
    cells.setdefault(cell, []).append((i, r))
filled_pip = 0
for cell, items in cells.items():
    if len(items) < 2:
        continue
    idxs = [i for i, _ in items]; bxs = [r['bbox'] for _, r in items]
    i_first, i_last = min(idxs), max(idxs)
    mbb = [round(med([b[k] for b in bxs]), 4) for k in range(4)]
    last_real = i_first
    for j in range(i_first, i_last + 1):
        if j in idxs:
            last_real = j; continue
        if j - last_real > 4:   # ne comble que les micro-trous (<=2s), pas le contenu off long
            continue
        if raw[j]['host'] == 'off':
            raw[j]['host'] = 'pip'; raw[j]['bbox'] = mbb; filled_pip += 1
print("PIP era-fill: %d off frames filled" % filled_pip)

# ---------- PASS 3: HERO era-fill (bridge <=5s off entre hero) ----------
hero_idxs = [i for i, r in enumerate(raw) if r['host'] == 'hero']
filled_hero = 0
for k in range(len(hero_idxs) - 1):
    a, b = hero_idxs[k], hero_idxs[k+1]
    if 1 < b - a <= 10 and all(raw[j]['host'] != 'pip' for j in range(a+1, b)):
        for j in range(a+1, b):
            if raw[j]['host'] == 'off':
                raw[j]['host'] = 'hero'; raw[j]['bbox'] = None; filled_hero += 1
print("HERO era-fill: %d off frames filled" % filled_hero)

# ---------- PASS 4: lone-off absorption ----------
for i in range(1, len(raw) - 1):
    if raw[i]['host'] == 'off' and raw[i-1]['host'] == raw[i+1]['host'] and raw[i-1]['host'] in ('hero', 'pip'):
        raw[i]['host'] = raw[i-1]['host']
        if raw[i-1]['host'] == 'pip':
            raw[i]['bbox'] = raw[i-1]['bbox']

# ---------- LISSAGE bbox pip: median glissant (tue les outliers transitoires) ----------
# Un vrai deplacement est soutenu; le bruit (accroche texte gauche 1-2 frames) est transitoire.
WIN = 3  # +-3 = fenetre de 7 samples (~3.5s)
pip_positions = [i for i, r in enumerate(raw) if r['host'] == 'pip' and r['bbox']]
smoothed = {}
for i in pip_positions:
    lo, hi = i-WIN, i+WIN
    neigh = [raw[j]['bbox'] for j in range(lo, hi+1) if 0 <= j < len(raw) and raw[j]['host'] == 'pip' and raw[j]['bbox']]
    if len(neigh) >= 2:
        smoothed[i] = [round(med([b[k] for b in neigh]), 4) for k in range(4)]
for i, bb in smoothed.items():
    raw[i]['bbox'] = bb

# ---------- MERGE (split pip quand le bbox saute = scene differente) ----------
def center(bxs):
    cx = med([b[0]+b[2]/2 for b in bxs]); cy = med([b[1]+b[3]/2 for b in bxs])
    return cx, cy
PIP_MOVE = 0.12  # saut de centre normalise -> nouvelle sous-era

segs = []
if raw:
    ch = raw[0]['host']; cs = raw[0]['t']; bxs = [raw[0]['bbox']] if raw[0]['bbox'] else []
    prev_t = raw[0]['t']
    for r in raw[1:]:
        split = False
        if r['host'] == ch == 'pip' and r['bbox'] and bxs:
            ccx, ccy = center(bxs)
            nx, ny = r['bbox'][0]+r['bbox'][2]/2, r['bbox'][1]+r['bbox'][3]/2
            if abs(nx-ccx) > PIP_MOVE or abs(ny-ccy) > PIP_MOVE:
                split = True
        if r['host'] == ch and not split:
            if r['bbox']: bxs.append(r['bbox'])
        else:
            avg = [round(med([b[i] for b in bxs]), 4) for i in range(4)] if bxs else None
            segs.append({'start': cs, 'end': round(r['t'], 2), 'host': ch, 'bbox': avg})
            ch = r['host']; cs = r['t']; bxs = [r['bbox']] if r['bbox'] else []
    avg = [round(med([b[i] for b in bxs]), 4) for i in range(4)] if bxs else None
    segs.append({'start': cs, 'end': round(dur, 2), 'host': ch, 'bbox': avg})

# absorber petits trous off (<2.0s) entre meme host ET meme position (pip)
merged = []
for s in segs:
    if merged and s['host'] != 'off' and merged[-1]['host'] == s['host'] and s['start'] - merged[-1]['end'] < 2.0:
        if s['host'] == 'pip' and s['bbox'] and merged[-1]['bbox']:
            pcx = merged[-1]['bbox'][0]+merged[-1]['bbox'][2]/2
            pcy = merged[-1]['bbox'][1]+merged[-1]['bbox'][3]/2
            ncx = s['bbox'][0]+s['bbox'][2]/2; ncy = s['bbox'][1]+s['bbox'][3]/2
            if abs(ncx-pcx) > PIP_MOVE or abs(ncy-pcy) > PIP_MOVE:
                merged.append(dict(s)); continue
        merged[-1]['end'] = s['end']
    else:
        merged.append(dict(s))

# ---------- NETTOYAGE FINAL ----------
# 1) drop pip aberrants (trop gros = scene chaude faussement captee)
# 2) drop micro-segs pip <1.0s (residus de bruit), fusionne le trou dans le voisin off
clean = []
for s in merged:
    if s['host'] == 'pip' and s['bbox']:
        frac = s['bbox'][2]*s['bbox'][3]
        bx, by, bw, bh = s['bbox']
        far_left_wide = (bx <= 0.06 and bw > 0.30)   # bande gauche large = texte/dessin, pas un pip webcam
        if frac > 0.35 or bw > 0.42 or bw < 0.06 or bh < 0.06 or far_left_wide:
            s = {'start': s['start'], 'end': s['end'], 'host': 'off', 'bbox': None}
    if s['host'] == 'pip' and (s['end']-s['start']) < 1.0:
        s = {'start': s['start'], 'end': s['end'], 'host': 'off', 'bbox': None}
    clean.append(s)
# re-merge off consecutifs / re-merge pip meme position adjacents
merged2 = []
for s in clean:
    if merged2 and merged2[-1]['host'] == s['host'] == 'off':
        merged2[-1]['end'] = s['end']
    elif (merged2 and merged2[-1]['host'] == s['host'] == 'pip' and s['bbox'] and merged2[-1]['bbox']
          and abs((s['bbox'][0]+s['bbox'][2]/2)-(merged2[-1]['bbox'][0]+merged2[-1]['bbox'][2]/2)) <= PIP_MOVE
          and abs((s['bbox'][1]+s['bbox'][3]/2)-(merged2[-1]['bbox'][1]+merged2[-1]['bbox'][3]/2)) <= PIP_MOVE
          and s['start']-merged2[-1]['end'] < 2.0):
        merged2[-1]['end'] = s['end']
    else:
        merged2.append(dict(s))
merged = merged2

json.dump(merged, open(os.path.join(out_dir, 'host_map.json'), 'w'), indent=2)
hc = sum(1 for s in merged if s['host'] == 'hero')
pc = sum(1 for s in merged if s['host'] == 'pip')
oc = sum(1 for s in merged if s['host'] == 'off')
print("host_map: %d segs hero=%d pip=%d off=%d" % (len(merged), hc, pc, oc))
for s in merged:
    if s['host'] == 'pip':
        print("  PIP %.1f-%.1fs bbox=%s" % (s['start'], s['end'], s['bbox']))
