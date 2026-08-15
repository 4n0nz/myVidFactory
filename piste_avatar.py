#!/usr/bin/env python3
# piste_avatar.py <sortie.mp4> <audio> --clips a.webm,b.webm --dur S [--mode banque|boucle]
#                 [--offset S] [--force 1.0]
#
# Produit une piste avatar de la duree EXACTE d une video, pilotee par la voix.
#
# Pourquoi une piste et pas une boucle : vf_avatar.py fait "-stream_loop -1" sur
# public/avatar.mp4 — l avatar tourne en rond, a vitesse fixe, sans aucun lien avec
# l audio. En fournissant une piste deja synchronisee ET plus longue que la video, la
# boucle infinie ne se declenche jamais : le pipeline n a pas besoin d etre modifie.
#
# Deux modes a comparer :
#   banque : le narrateur CHANGE de pose au fil de la video (mains jointes, clavier...)
#   boucle : une seule pose du debut a la fin, seule la vitesse suit la voix
#
# --offset : vf_finalize insere 1,5 s de neige AVANT l image. L audio de reference (pris
# sur le montage final) est donc en avance de 1,5 s sur le master vert ou l avatar se pose.
#
# Le fond est compose en NOIR : vf_avatar pose l avatar en cover dans les masques, donc
# son fond est visible ; l alpha reste le master pour composer ailleurs.
import os, subprocess, sys, wave
import cv2, numpy as np

DST, AUDIO = sys.argv[1], sys.argv[2]


def arg(n, d=None):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


CLIPS = [c for c in (arg('--clips', '') or '').split(',') if c]
DUR = float(arg('--dur', '60'))
MODE = arg('--mode', 'banque')
OFFSET = float(arg('--offset', '0'))
FORCE = float(arg('--force', '1.0'))
FPS, SR = 30, 16000
if not CLIPS:
    sys.exit('--clips requis')

# ---------------------------------------------------------------- enveloppe de la voix
wav = '/tmp/piste_%d.wav' % os.getpid()
ss = ['-ss', '%.3f' % OFFSET] if OFFSET > 0 else []
subprocess.run(['ffmpeg', '-y', '-v', 'error'] + ss + ['-i', AUDIO, '-t', '%.3f' % DUR,
                '-ac', '1', '-ar', str(SR), wav], check=True)
with wave.open(wav) as w:
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
NF = int(round(DUR * FPS))
pf = max(1, len(pcm) // NF)
rms = np.array([np.sqrt(np.mean(pcm[i*pf:(i+1)*pf]**2) + 1e-12) for i in range(NF)])
env = np.power(rms, 0.5)
lo, hi = np.percentile(env, 5), np.percentile(env, 95)
env = np.clip((env - lo) / max(1e-6, hi - lo), 0, 1)
env = np.convolve(env, np.ones(9) / 9, mode='same')
os.remove(wav)
print('%s | %.0f s | voix active %.0f%% du temps' % (MODE, DUR, 100 * (env > 0.25).mean()))

# ---------------------------------------------------------------- clips en memoire (BGR)
banque = []
for c in CLIPS:
    d = '/tmp/pa_%d' % abs(hash(c))
    subprocess.run(['rm', '-rf', d]); os.makedirs(d)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-c:v', 'libvpx-vp9', '-i', c,
                    '-vf', 'format=rgba', '%s/f%%04d.png' % d], check=True)
    frames = []
    for f in sorted(os.listdir(d)):
        im = cv2.imread(os.path.join(d, f), cv2.IMREAD_UNCHANGED)
        if im.shape[2] == 4:                      # compose sur noir
            a = im[:, :, 3:4].astype(np.float32) / 255.0
            im = (im[:, :, :3].astype(np.float32) * a).astype(np.uint8)
        frames.append(im)
    subprocess.run(['rm', '-rf', d])
    banque.append(frames)
    print('  %-34s %d images' % (os.path.basename(c), len(frames)))
H, W = banque[0][0].shape[:2]

# ---------------------------------------------------------------- frontieres de segments
# Mode "segments" (consigne Boss) : la pose ne change QU AUX FRONTIERES du montage —
# passage hero<->pip, ou changement de scene. Changer au bout d un clip (mode banque)
# donnait 559 bascules sur 12 min, souvent au milieu d une phrase.
# host_map.json est en temps du MASTER VERT, comme la piste avatar : aucun decalage ici.
# L OFFSET ne concerne que l audio, pris sur le montage final ou la neige decale tout.
FRONTIERES = []
HM = arg('--segments')
if HM and os.path.exists(HM):
    import json
    segs = sorted(json.load(open(HM)), key=lambda s: s['start'])
    FRONTIERES = sorted({int(round(s['start'] * FPS)) for s in segs if 0 < s['start'] < DUR})
    print('  %d segments -> %d frontieres de changement de pose' % (len(segs), len(FRONTIERES)))

# ---------------------------------------------------------------- parcours
base = 0.08 + 0.92 * (1 - FORCE)
vit = base + (1.0 - base) * np.power(env, 0.8) * (1 + 0.8 * FORCE)
rng = np.random.default_rng(11)
fset = set(FRONTIERES)
ci, pos, sens, plan = 0, 0.0, 1, []
for i in range(NF):
    if MODE == 'segments' and i in fset and len(banque) > 1:
        # nouvelle scene : nouvelle pose, on repart du debut du clip
        ci = int(rng.choice([k for k in range(len(banque)) if k != ci]))
        pos, sens = 0.0, 1
    n = len(banque[ci])
    pos += vit[i] * sens
    if pos >= n - 1 or pos <= 0:
        if MODE == 'banque' and len(banque) > 1:
            ci = int(rng.choice([k for k in range(len(banque)) if k != ci]))
            pos = 0.0 if sens > 0 else len(banque[ci]) - 1.0
        else:
            # dans un segment (ou en mode boucle) : aller-retour sur la meme pose
            sens = -sens
            pos = max(0.0, min(n - 1.0, pos))
    plan.append((ci, int(round(pos))))

# ---------------------------------------------------------------- ecriture directe
enc = ['h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '20', '-b:v', '0']
if subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                   'color=c=red:s=320x180:r=30', '-t', '1', '-c:v', 'h264_nvenc',
                   '/tmp/nvp4.mp4'], capture_output=True).returncode != 0:
    enc = ['libx264', '-preset', 'fast', '-crf', '20']
p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                      '-s', '%dx%d' % (W, H), '-r', str(FPS), '-i', '-', '-an',
                      '-c:v'] + enc + ['-pix_fmt', 'yuv420p', DST], stdin=subprocess.PIPE)
for c, f in plan:
    p.stdin.write(banque[c][min(f, len(banque[c]) - 1)].tobytes())
p.stdin.close(); p.wait()

chg = sum(1 for i in range(1, len(plan)) if plan[i][0] != plan[i-1][0])
d = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                   'format=duration', '-of', 'csv=p=0', DST]))
print('%s : %.1f s, %dx%d | %d changements de pose | %d images distinctes'
      % (os.path.basename(DST), d, W, H, chg, len(set(plan))))
