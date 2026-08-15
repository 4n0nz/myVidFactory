#!/usr/bin/env python3
# sync_alpha.py <sortie.webm> <audio> [--dur 30] [--force 1.0] [--clips a.webm,b.webm,...]
#
# Meme principe que sync_voix mais :
#   - FOND TRANSPARENT conserve de bout en bout (VP9 alpha) — c est le livrable retenu ;
#   - plusieurs clips sources au lieu d un seul, pour ne pas rejouer la meme boucle.
#
# Anti-repetition : on parcourt une BANQUE de gestes. A chaque fois qu on atteint un bout
# de clip, on saute vers un AUTRE clip (jamais celui qu on vient de jouer), avec un fondu
# court. Le personnage est le meme partout — seule la graine de generation change — donc
# le raccord ne se voit pas, contrairement a un aller-retour qui rejoue le geste a l envers.
import os, subprocess, sys, wave
import numpy as np

DST = sys.argv[1]
AUDIO = sys.argv[2]


def arg(n, d=None):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


DUREE = float(arg('--dur', '30'))
FORCE = float(arg('--force', '1.0'))
CLIPS = [c for c in (arg('--clips', '') or '').split(',') if c]
FPS = 30
SR = 16000
TMP = '/tmp/syncalpha'
subprocess.run(['rm', '-rf', TMP]); os.makedirs(TMP, exist_ok=True)

if not CLIPS:
    sys.exit('donner au moins un clip avec --clips')

# ------------------------------------------------------- 1. enveloppe de la voix
wav = TMP + '/voix.wav'
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', AUDIO, '-t', '%.3f' % DUREE,
                '-ac', '1', '-ar', str(SR), wav], check=True)
with wave.open(wav) as w:
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
NF = int(round(DUREE * FPS))
pf = max(1, len(pcm) // NF)
rms = np.array([np.sqrt(np.mean(pcm[i*pf:(i+1)*pf]**2) + 1e-12) for i in range(NF)])
env = np.power(rms, 0.5)
lo, hi = np.percentile(env, 5), np.percentile(env, 95)
env = np.clip((env - lo) / max(1e-6, hi - lo), 0, 1)
env = np.convolve(env, np.ones(9) / 9, mode='same')
print('voix : %.0f%% du temps au-dessus du seuil de parole' % (100 * (env > 0.25).mean()))

# ------------------------------------------------------- 2. banque d images RGBA
banque = []
for ci, c in enumerate(CLIPS):
    d = '%s/c%d' % (TMP, ci)
    os.makedirs(d, exist_ok=True)
    # -vf format=rgba : force la sortie en RGBA meme si la source est opaque
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-c:v', 'libvpx-vp9', '-i', c,
                    '-vf', 'format=rgba', '%s/f%%04d.png' % d], check=True)
    n = len(os.listdir(d))
    banque.append((d, n))
    print('  clip %d : %d images (%s)' % (ci + 1, n, os.path.basename(c)))
total = sum(n for _, n in banque)
print('banque : %d clips, %d images, %.1f s de geste distinct' % (len(banque), total, total / FPS))

# ------------------------------------------------------- 3. parcours pilote par la voix
base = 0.08 + 0.92 * (1 - FORCE)
vitesse = base + (1.0 - base) * np.power(env, 0.8) * (1 + 0.8 * FORCE)

rng = np.random.default_rng(7)          # deterministe : meme audio -> meme montage
ci = 0
pos = 0.0
sens = 1
plan = []
for i in range(NF):
    pos += vitesse[i] * sens
    n = banque[ci][1]
    if pos >= n - 1 or pos <= 0:
        # bout de clip atteint : on change de geste plutot que de rejouer a l envers
        autres = [k for k in range(len(banque)) if k != ci]
        if autres:
            ci = int(rng.choice(autres))
            pos = 0.0 if sens > 0 else banque[ci][1] - 1.0
        else:
            sens = -sens
            pos = max(0.0, min(banque[ci][1] - 1.0, pos))
    plan.append((ci, int(round(pos))))

# ------------------------------------------------------- 4. montage RGBA
seq = TMP + '/seq'
os.makedirs(seq, exist_ok=True)
for i, (c, f) in enumerate(plan):
    d, n = banque[c]
    src = '%s/f%04d.png' % (d, min(max(f, 0), n - 1) + 1)
    os.link(src, '%s/f%05d.png' % (seq, i)) if not os.path.exists('%s/f%05d.png' % (seq, i)) else None

# -auto-alt-ref 0 / -lag-in-frames 0 : SANS eux, VP9 utilise des images de reference
# alternees, incompatibles avec le canal alpha — l encodage reussit mais sort en yuv420p,
# l alpha est silencieusement perdu (verifie : alphaextract echoue sur la sortie).
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', str(FPS),
                '-i', '%s/f%%05d.png' % seq, '-i', wav,
                '-map', '0:v', '-map', '1:a', '-c:a', 'libopus', '-b:a', '128k',
                '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', '-b:v', '3M',
                '-auto-alt-ref', '0', '-lag-in-frames', '0',
                '-shortest', DST], check=True)

# on ne se fie pas au code de retour : on VERIFIE que l alpha a survecu
ok = subprocess.run(['ffmpeg', '-v', 'error', '-c:v', 'libvpx-vp9', '-i', DST,
                     '-vf', 'alphaextract', '-frames:v', '1', '-f', 'null', '-'],
                    capture_output=True).returncode == 0
print('canal alpha dans la sortie : %s' % ('PRESENT' if ok else 'PERDU'))

changements = sum(1 for i in range(1, len(plan)) if plan[i][0] != plan[i-1][0])
print('\n%s : %.0f s, fond transparent (VP9 alpha)' % (DST, DUREE))
print('%d changements de geste | %d images distinctes utilisees sur %d'
      % (changements, len(set(plan)), NF))
