#!/usr/bin/env python3
# Compositor SEGMENTE : un ffmpeg par segment (seul l'overlay actif est calcule),
# puis concat + mux audio. ~100x moins de travail/frame que le filtergraph geant.
# PIP, deux modes :
#   - webcam FIXE  : env/arg PIP_RECT="x,y,w,h" -> meme rect pour tous les pips (coins nets).
#   - webcam MOBILE: pas de PIP_RECT -> bbox PAR-SEGMENT de la detection (suit la webcam),
#     avec extension du haut (la box ancree-visage sous-evalue le haut du cadre webcam).
import json, subprocess, os, sys

workdir  = sys.argv[1] if len(sys.argv) > 1 else '/home/boss/videogen/wk_full'
out_name = sys.argv[2] if len(sys.argv) > 2 else 'my_remix.mp4'
hmap   = json.load(open(workdir + '/host_map.json'))
avatar = '/home/boss/videogen/public/avatar.mp4'
source = workdir + '/source.mp4'
outp   = '/home/boss/videogen/out/' + out_name
os.makedirs('/home/boss/videogen/out', exist_ok=True)
segdir = workdir + '/segs'; os.makedirs(segdir, exist_ok=True)
for f in os.listdir(segdir):
    if f.endswith(('.png', '.mp4', '.txt')): os.remove(segdir + '/' + f)

r = subprocess.check_output(
    'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 ' + source,
    shell=True).decode().strip().split(',')
W, H = int(r[0]), int(r[1])
num, den = r[2].split('/'); FPS = round(float(num) / float(den), 4)

NV = "-c:v h264_nvenc -preset p4 -rc vbr -cq 23 -b:v 0"

# Mode FIXE si PIP_RECT fourni (env "x,y,w,h"), sinon mode PAR-SEGMENT (bbox detectee).
_env = os.environ.get('PIP_RECT', '').strip()
FIXED = None
if _env:
    FIXED = tuple(int(v) for v in _env.split(','))
    assert len(FIXED) == 4, "PIP_RECT doit etre 'x,y,w,h'"

TOP_EXT = 0.0    # v2.1: la bbox = deja le vrai rectangle webcam (edge-scan + marge) -> pas d'extension
SIDE    = 0.0

# coins arrondis sur l'avatar pip (les webcams ont un border-radius) -> l'overlay epouse
# le cadre au lieu de deborder aux coins. Rayon = RADIUS_FRAC du petit cote de la box.
# PIP_ROUND: '0' = jamais rond, '1' = toujours rond, 'auto' (defaut) = detection par
# segment : coin arrondi SEULEMENT si la source montre un coin arrondi (sinon un avatar
# rond sur une webcam carree laisse depasser les coins du narrateur).
PIP_ROUND   = os.environ.get('PIP_ROUND', 'auto').strip().lower()
RADIUS_FRAC = 0.08

def _patch_mean(img, px, py):
    h, w = img.shape[:2]
    x0 = max(0, px - 1); y0 = max(0, py - 1)
    p = img[y0:min(h, py + 2), x0:min(w, px + 2)]
    if p.size == 0: return None
    return p.reshape(-1, 3).mean(axis=0)

def detect_round(x, y, w, h, ss, dur):
    # coin arrondi si le pixel de coin (interieur bbox) ressemble au fond exterieur
    # et PAS au contenu webcam du bord. Vote 4 coins x 3 frames ; defaut = carre.
    import cv2, numpy as np
    r = max(2, int(min(w, h) * RADIUS_FRAC))
    d = max(1, int(r * 0.25))
    cap = cv2.VideoCapture(source)
    votes = 0; valid = 0
    for frac in (0.25, 0.5, 0.75):
        cap.set(cv2.CAP_PROP_POS_MSEC, (ss + dur * frac) * 1000.0)
        ok, img = cap.read()
        if not ok: continue
        img = img.astype('float32')
        corners = ((x, y, 1, 1), (x + w - 1, y, -1, 1), (x, y + h - 1, 1, -1), (x + w - 1, y + h - 1, -1, -1))
        for cx, cy, sx, sy in corners:
            pc = _patch_mean(img, cx + sx * d, cy + sy * d)            # coin, interieur bbox
            po = _patch_mean(img, cx - sx * 4, cy - sy * 4)            # fond, exterieur bbox
            pe1 = _patch_mean(img, x + w // 2, cy + sy * d)            # bord haut/bas, milieu
            pe2 = _patch_mean(img, cx + sx * d, y + h // 2)            # bord gauche/droit, milieu
            if pc is None or po is None or pe1 is None or pe2 is None: continue
            import numpy as _np
            d_out = float(_np.linalg.norm(pc - po))
            d_edge = min(float(_np.linalg.norm(pc - pe1)), float(_np.linalg.norm(pc - pe2)))
            valid += 1
            if d_out + 15 < d_edge: votes += 1
    cap.release()
    return valid >= 4 and votes * 2 >= valid

def even(v): return v - (v % 2)

def rounded_lum(w, h, r):
    # expression geq (alpha 0-255) : masque rectangle a coins arrondis rayon r sur wxh.
    # un pixel dans un coin ET hors du cercle du coin -> alpha 0 (montre le fond).
    W1 = w - r; H1 = h - r
    return ("255*(1-("
            "lt(X,%d)*lt(Y,%d)*gt(hypot(%d-X,%d-Y),%d)+"
            "gt(X,%d)*lt(Y,%d)*gt(hypot(X-%d,%d-Y),%d)+"
            "lt(X,%d)*gt(Y,%d)*gt(hypot(%d-X,Y-%d),%d)+"
            "gt(X,%d)*gt(Y,%d)*gt(hypot(X-%d,Y-%d),%d)))"
            % (r, r, r, r, r,
               W1, r, W1, r, r,
               r, H1, r, H1, r,
               W1, H1, W1, H1, r))

def pip_fc(w, h, x, y, rnd):
    """filter_complex pour un pip : avatar cover -> (coins arrondis si rnd) -> overlay."""
    if rnd:
        r = max(2, int(min(w, h) * RADIUS_FRAC))
        expr = rounded_lum(w, h, r)
        return ("[1:v]%s,format=rgba[base];"
                "color=c=black:s=%dx%d:r=%s,format=gray,geq=lum='%s',loop=loop=-1:size=1[m];"
                "[base][m]alphamerge[av];[0:v][av]overlay=%d:%d:shortest=1[vo]"
                % (cover(w, h), w, h, FPS, expr, x, y))
    return "[1:v]%s[av];[0:v][av]overlay=%d:%d:shortest=1[vo]" % (cover(w, h), x, y)

def seg_rect(bbox):
    fx, fy, fw, fh = bbox
    x = int(fx * W); y = int(fy * H); w = int(fw * W); h = int(fh * H)
    top = int(h * TOP_EXT); side = int(min(w, h) * SIDE)
    nx = max(0, x - side); ny = max(0, y - top)
    nw = min(w + 2 * side, W - nx); nh = min(h + top + side, H - ny)
    nw = even(nw); nh = even(nh)
    if nw < 16 or nh < 16: return None
    return nx, ny, nw, nh

def cover(w, h):
    return "scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d" % (w, h, w, h)

lines = ["#!/bin/bash", "set -e", "exec > %s/seg.log 2>&1" % workdir, "echo '=== START SEG RENDER ==='",
         "date", 'T0=$(date +%s)']
concat = []
for si, s in enumerate(hmap):
    d = round(s['end'] - s['start'], 3)
    if d <= 0: continue
    sf = "%s/seg%04d.mp4" % (segdir, si)
    ss = s['start']; host = s['host']

    if host == 'hero':
        fc = "[1:v]%s[av];[0:v][av]overlay=0:0:shortest=1[vo]" % cover(W, H)
        cmd = ('ffmpeg -y -ss %s -t %s -i %s -stream_loop -1 -i %s '
               '-filter_complex "%s" -map "[vo]" -an -r %s -t %s %s "%s"'
               % (ss, d, source, avatar, fc, FPS, d, NV, sf))
    elif host == 'pip':
        rect = FIXED if FIXED else (seg_rect(s['bbox']) if s.get('bbox') else None)
        if rect:
            x, y, w, h = rect
            if PIP_ROUND == '1': rnd = True
            elif PIP_ROUND == '0': rnd = False
            else: rnd = detect_round(x, y, w, h, ss, d)
            fc = pip_fc(w, h, x, y, rnd)
            cmd = ('ffmpeg -y -ss %s -t %s -i %s -stream_loop -1 -i %s '
                   '-filter_complex "%s" -map "[vo]" -an -r %s -t %s %s "%s"'
                   % (ss, d, source, avatar, fc, FPS, d, NV, sf))
        else:  # pip sans bbox exploitable -> source brute
            cmd = ('ffmpeg -y -ss %s -t %s -i %s -an -r %s -t %s %s "%s"'
                   % (ss, d, source, FPS, d, NV, sf))
    else:  # off -> source brut
        cmd = ('ffmpeg -y -ss %s -t %s -i %s -an -r %s -t %s %s "%s"'
               % (ss, d, source, FPS, d, NV, sf))

    concat.append("file '%s'" % sf)
    lines.append("echo '--- seg %d/%d  %s  %.1f-%.1fs ---'" % (si + 1, len(hmap), host, s['start'], s['end']))
    lines.append(cmd)

listf = segdir + '/concat.txt'
open(listf, 'w').write('\n'.join(concat) + '\n')
lines.append("echo '=== CONCAT ==='")
lines.append('ffmpeg -y -f concat -safe 0 -i %s -c copy %s/videoonly.mp4' % (listf, segdir))
lines.append("echo '=== MUX AUDIO ==='")
lines.append('ffmpeg -y -i %s/videoonly.mp4 -i %s -map 0:v -map 1:a -c:v copy -c:a copy -metadata:s:a:0 language=und -shortest "%s"'
             % (segdir, source, outp))
lines.append("echo '=== EXTRACT WAV ==='")
lines.append('ffmpeg -y -i "%s" -vn -acodec pcm_s16le -ar 48000 -ac 2 "%s"'
             % (outp, outp.rsplit('.', 1)[0] + '.wav'))
lines.append('T1=$(date +%s); echo "=== DONE in $((T1-T0))s ==="')
lines.append('ls -lh "%s"' % outp)

open(workdir + '/run_seg.sh', 'w').write('\n'.join(lines) + '\n')
os.chmod(workdir + '/run_seg.sh', 0o755)
npip = sum(1 for s in hmap if s['host'] == 'pip')
mode = ("FIXE %s" % (FIXED,)) if FIXED else "PAR-SEGMENT (bbox detectee, top+%.0f%%)" % (TOP_EXT * 100)
print("SEG: %d segments, %d pip, mode=%s, %dx%d @%sfps" % (len(hmap), npip, mode, W, H, FPS))
