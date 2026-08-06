#!/usr/bin/env python3
# Compositor SEGMENTE : un ffmpeg par segment (seul l'overlay actif est calcule),
# puis concat + mux audio. ~100x moins de travail/frame que le filtergraph geant.
# PIP, deux modes :
#   - webcam FIXE  : env/arg PIP_RECT="x,y,w,h" -> meme rect pour tous les pips (coins nets).
#   - webcam MOBILE: pas de PIP_RECT -> bbox PAR-SEGMENT de la detection (suit la webcam),
#     avec extension du haut (la box ancree-visage sous-evalue le haut du cadre webcam).
import json, subprocess, os, sys, hashlib

workdir  = sys.argv[1] if len(sys.argv) > 1 else '/home/boss/videogen/wk_full'
out_name = sys.argv[2] if len(sys.argv) > 2 else 'my_remix.mp4'
hmap   = json.load(open(workdir + '/host_map.json'))
avatar = '/home/boss/videogen/public/avatar.mp4'
source = workdir + '/source.mp4'
outp   = '/home/boss/videogen/out/' + out_name
os.makedirs('/home/boss/videogen/out', exist_ok=True)
segdir = workdir + '/segs'; os.makedirs(segdir, exist_ok=True)
# RENDER INCREMENTAL (branche optimisation 2026-08-06) : on ne supprime PLUS les segs.
# Un seg est re-rendu SEULEMENT si sa cle (commande ffmpeg + contenu du masque) a change
# depuis le dernier rendu reussi — la cle est ecrite en seg%04d.hash par run_seg.sh
# APRES le ffmpeg (donc un rendu interrompu n'a pas de hash = re-rendu au tour suivant).
# Un tour de QC ne re-rend ainsi que les segments touches par le fix (~90% du temps de
# render economise sur les tours 2+). Les .txt sont regeneres a chaque passe.
for f in os.listdir(segdir):
    if f.endswith('.txt'): os.remove(segdir + '/' + f)

def _sha(s):
    return hashlib.sha1(s.encode() if isinstance(s, str) else s).hexdigest()

_mask_sha = {}
def mask_sha(p):
    """empreinte du CONTENU du masque (le chemin peut garder le meme nom avec des
    pixels differents apres un qc_fix + gen_masks)."""
    if p not in _mask_sha:
        try: _mask_sha[p] = _sha(open(p, 'rb').read())
        except Exception: _mask_sha[p] = 'NOMASK'
    return _mask_sha[p]

# cache des verdicts detect_round : 3 seeks video par pip candidat a chaque passe,
# resultat pourtant deterministe pour (rect, fenetre temporelle) donnes.
_round_path = segdir + '/round_cache.json'
try: _round_cache = json.load(open(_round_path))
except Exception: _round_cache = {}

r = subprocess.check_output(
    'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of csv=p=0 ' + source,
    shell=True).decode().strip().split(',')
W, H = int(r[0]), int(r[1])
num, den = r[2].split('/'); FPS = round(float(num) / float(den), 4)
RATE = '%s/%s' % (num, den)     # cadence EXACTE pour -r : 29.97 != 30000/1001
FPSX = float(num) / float(den)  # grille de frames de la source
try: NF = int(r[3])
except Exception: NF = 0

NV = "-c:v h264_nvenc -preset p4 -rc vbr -cq 23 -b:v 0"
# VF_ENC=cpu : fallback libx264 quand NVENC est mort (driver upgrade sous module charge,
# 2026-07-23 — reload = sudo, indisponible). Le contenu vert compresse vite en CPU.
_enc = os.environ.get('VF_ENC', '').strip()
if not _enc:
    # pas d'indication de l'appelant -> sonde NVENC directe (les chaines unitaires
    # n'exportent pas VF_ENC ; gOQZ run_seg mort sur NVENC HS, 2026-07-25)
    _enc = 'gpu' if os.system(
        "ffmpeg -y -v error -f lavfi -i color=c=red:s=320x180:r=30 -t 1 "
        "-c:v h264_nvenc /tmp/nvenc_probe.mp4 2>/dev/null") == 0 else 'cpu'
if _enc == 'cpu':
    NV = "-c:v libx264 -preset fast -crf 23"
AVIN = ('-f lavfi -i color=c=0x00FF00:s=%dx%d:r=%s' % (W, H, RATE)) if os.environ.get('GREEN_PIP','0').strip()=='1' else None

# Mode FIXE si PIP_RECT fourni (env "x,y,w,h"), sinon mode PAR-SEGMENT (bbox detectee).
_env = os.environ.get('PIP_RECT', '').strip()
FIXED = None
if _env:
    FIXED = tuple(int(v) for v in _env.split(','))
    assert len(FIXED) == 4, "PIP_RECT doit etre 'x,y,w,h'"

TOP_EXT = 0.0    # v2.1: la bbox = deja le vrai rectangle webcam (edge-scan + marge) -> pas d'extension
SIDE    = 0.0

# Filtre couleur applique a l'AVATAR (pip et hero) avant overlay. Env AVATAR_FX pour
# override ; defaut = tint vert #00ff00 (Boss 2026-07-20) : G booste, R/B reduits.
AVFX = os.environ.get('AVATAR_FX', 'colorchannelmixer=rr=0.6:gg=1.25:bb=0.6').strip()

# GREEN_PIP=1 : le pip original est rempli de VERT CHROMA pur (#00ff00) au lieu de
# l'avatar (Boss 2026-07-20) — validation visuelle immediate de la geometrie + keying
# possible en post. L'entree [1:v] devient une source couleur lavfi ; formes, coins,
# masques, overlay : inchanges.
GREEN = os.environ.get('GREEN_PIP', '0').strip() == '1'
if GREEN:
    AVFX = ''
def av_in():
    """entree avatar [1:v] avec le filtre couleur si defini."""
    return "[1:v]%s," % AVFX if AVFX else "[1:v]"

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
        return (av_in()+"%s,format=rgba[base];"
                "color=c=black:s=%dx%d:r=%s,format=gray,geq=lum='%s',loop=loop=-1:size=1[m];"
                "[base][m]alphamerge[av];[0:v][av]overlay=%d:%d:shortest=1[vo]"
                % (cover_auto(w, h), w, h, FPS, expr, x, y))
    return av_in()+"%s[av];[0:v][av]overlay=%d:%d:shortest=1[vo]" % (cover_auto(w, h), x, y)

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

# --- colonnes verticales (batch20 QQEg) : le crop-to-fill plein centre donne un visage geant
# coupe dans une box 1:3. A la place : fond = avatar zoome FLOU (crop-to-fill + boxblur),
# avatar net fit-WIDTH pose au tiers haut (comme une vraie cam colonne). Toggle env PIP_COL=0
# pour revenir a l'ancien crop-to-fill partout.
COL_RATIO = 0.70
PIP_COL = os.environ.get('PIP_COL', '0')  # DEFAUT OFF : Boss rejette le fond flou (2026-07-18). Crop-to-fill partout.

def cover_col(w, h):
    fgh = max(2, int(w * 9 / 16 / 2) * 2)  # hauteur avatar 16:9 fit-width, paire
    fy = max(0, int(h * 0.10))
    return ("split[cba][cbb];"
            "[cba]scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,"
            "boxblur=luma_radius=24:luma_power=2:chroma_radius=12[cbg];"
            "[cbb]scale=%d:%d[cfg];[cbg][cfg]overlay=0:%d"
            % (w, h, w, h, w, fgh, fy))

def cover_auto(w, h):
    # colonne = etroit ET quasi pleine hauteur ecran. Une petite box verticale (ex 0.11x0.29)
    # n'est PAS une colonne -> crop-to-fill normal (bug vu sur XzEg t=322, box decoy 219x314).
    if PIP_COL != '0' and h > 0 and (w / float(h)) < COL_RATIO and h >= 0.6 * H:
        return cover_col(w, h)
    return cover(w, h)

# PAVAGE SUR LA GRILLE DE FRAMES (2026-07-27). Les bornes etaient passees telles quelles
# a -t <duree arrondie au centieme> : ffmpeg sort ceil(d*fps) frames, donc le concat gagnait
# jusqu'a 1 frame par segment et le rendu DERIVAIT de la source. eglV passe 16 : 41821
# frames rendues pour 41817 dans la source, et le vert arrivait deja +0.21s trop tard a
# t=43 (sonde verte : carte a nu 42.9-43.2, verdict Boss 27/07 07h25) ; la video se
# desynchronisait aussi de l'audio, muxe sur l'horloge source. Chaque scene devient un
# INDEX de frame, la fin de chaque segment est le DEBUT du suivant (jamais sa propre borne
# arrondie) et la longueur est imposee en frames : les segments pavent la timeline sans
# trou ni recouvrement, derive nulle par construction, meme si une scene est trop courte
# pour survivre a l'arrondi.
_i0 = [int(round(s['start'] * FPSX)) for s in hmap]
_last = NF if NF > _i0[-1] else int(round(hmap[-1]['end'] * FPSX))
_i1 = _i0[1:] + [_last]

lines = ["#!/bin/bash", "set -e", "exec > %s/seg.log 2>&1" % workdir, "echo '=== START SEG RENDER ==='",
         "date", 'T0=$(date +%s)']
concat = []
skipped = 0
for si, s in enumerate(hmap):
    n = _i1[si] - _i0[si]
    if n <= 0: continue
    d = n / FPSX
    din = d + 2.0 / FPSX        # lecture un peu plus longue : -frames:v tranche net
    sf = "%s/seg%04d.mp4" % (segdir, si)
    ss = _i0[si] / FPSX; host = s['host']

    if host == 'hero':
        fc = av_in()+"%s[av];[0:v][av]overlay=0:0:shortest=1[vo]" % cover(W, H)
        cmd = ('ffmpeg -y -ss %s -t %s -i %s %s '
               '-filter_complex "%s" -map "[vo]" -an -r %s -frames:v %s %s "%s"'
               % (ss, din, source, AVIN or ('-stream_loop -1 -i ' + avatar), fc, RATE, n, NV, sf))
    elif host == 'pip' and s.get('mask') and not FIXED:
        # COMPOSITE MASQUE (pixel-exact) : avatar passe a travers le masque de la fenetre webcam
        # (grabCut) -> forme exacte (rond/arrondi), zero fuite. bbox = bbox du masque.
        fx, fy, fw, fh = s['bbox']
        x = int(fx * W); y = int(fy * H); w = even(int(fw * W)); h = even(int(fh * H))
        mp = s['mask']
        fc = (av_in()+"scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,format=rgba[av0];"
              "[2:v]scale=%d:%d,format=gray[mk];[av0][mk]alphamerge[av];"
              "[0:v][av]overlay=%d:%d:shortest=1[vo]" % (w, h, w, h, w, h, x, y))
        cmd = ('ffmpeg -y -ss %s -t %s -i %s %s -stream_loop -1 -i %s '
               '-filter_complex "%s" -map "[vo]" -an -r %s -frames:v %s %s "%s"'
               % (ss, din, source, AVIN or ('-stream_loop -1 -i ' + avatar), mp, fc, RATE, n, NV, sf))
    elif host == 'pip':
        rect = FIXED if FIXED else (seg_rect(s['bbox']) if s.get('bbox') else None)
        if rect:
            x, y, w, h = rect
            if PIP_ROUND == '1': rnd = True
            elif PIP_ROUND == '0': rnd = False
            else:
                _rk = '%d,%d,%d,%d,%.3f,%.3f' % (x, y, w, h, ss, d)
                if _rk not in _round_cache:
                    _round_cache[_rk] = detect_round(x, y, w, h, ss, d)
                rnd = _round_cache[_rk]
            fc = pip_fc(w, h, x, y, rnd)
            cmd = ('ffmpeg -y -ss %s -t %s -i %s %s '
                   '-filter_complex "%s" -map "[vo]" -an -r %s -frames:v %s %s "%s"'
                   % (ss, din, source, AVIN or ('-stream_loop -1 -i ' + avatar), fc, RATE, n, NV, sf))
        else:  # pip sans bbox exploitable -> source brute
            cmd = ('ffmpeg -y -ss %s -t %s -i %s -an -r %s -frames:v %s %s "%s"'
                   % (ss, din, source, RATE, n, NV, sf))
    else:  # off -> source brut
        cmd = ('ffmpeg -y -ss %s -t %s -i %s -an -r %s -frames:v %s %s "%s"'
               % (ss, din, source, RATE, n, NV, sf))

    concat.append("file '%s'" % sf)
    # cle d identite du seg : commande complete + contenu du masque. Hash present et
    # identique + fichier present -> seg reutilise tel quel (bit-identique).
    key = _sha(cmd + '|' + (mask_sha(s['mask']) if host == 'pip' and s.get('mask') and not FIXED else ''))
    hf = sf.rsplit('.', 1)[0] + '.hash'
    if os.path.exists(sf) and os.path.exists(hf) and open(hf).read().strip() == key:
        lines.append("echo '--- seg %d/%d  %s  %.1f-%.1fs [CACHE] ---'" % (si + 1, len(hmap), host, s['start'], s['end']))
        skipped += 1
        continue
    lines.append("echo '--- seg %d/%d  %s  %.1f-%.1fs ---'" % (si + 1, len(hmap), host, s['start'], s['end']))
    lines.append(cmd)
    lines.append("echo '%s' > '%s'" % (key, hf))

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
json.dump(_round_cache, open(_round_path, 'w'))
# menage : les segs/hashes qui ne font plus partie du plan (renumerotation apres un
# fix qui change le nombre de scenes) ne doivent pas trainer sur disque
_want = set(c.split("'")[1] for c in concat)
for f in os.listdir(segdir):
    p = segdir + '/' + f
    if f.endswith('.mp4') and f.startswith('seg') and p not in _want and f != 'videoonly.mp4':
        os.remove(p)
        hp = p.rsplit('.', 1)[0] + '.hash'
        if os.path.exists(hp): os.remove(hp)
npip = sum(1 for s in hmap if s['host'] == 'pip')
mode = ("FIXE %s" % (FIXED,)) if FIXED else "PAR-SEGMENT (bbox detectee, top+%.0f%%)" % (TOP_EXT * 100)
print("SEG: %d segments, %d pip, %d en cache, mode=%s, %dx%d @%sfps" % (len(hmap), npip, skipped, mode, W, H, FPS))
