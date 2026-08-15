#!/usr/bin/env python3
# sync_voix.py <clip.mp4|webm> <audio_source.mp4|wav> <sortie.mp4> [--dur 20] [--force 1.0]
#
# Synchronise les mouvements de tete sur la VOIX, sans rien regenerer.
#
# PRINCIPE : le clip Wan contient deja un mouvement de tete propre, avec un masque intact.
# Plutot que de demander a un modele audio-driven de refabriquer le visage (ce qui deforme
# le masque, cf SVD), on pilote la VITESSE DE LECTURE de ce clip par l enveloppe de la
# voix : ca bouge quand ca parle, ca se calme dans les silences. Aucun pixel n est
# recalcule, donc la coque ne peut pas se deformer.
#
# Le mouvement est en ALLER-RETOUR sur le clip source : on ne retombe jamais sur une
# coupure, quelle que soit la duree demandee.
import os, subprocess, sys
import cv2, numpy as np

CLIP, AUDIO, DST = sys.argv[1], sys.argv[2], sys.argv[3]


def arg(n, d):
    return float(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d


DUREE = arg('--dur', 20.0)
FORCE = arg('--force', 1.0)      # 0 = vitesse constante, 1 = fortement pilote par la voix
SR = 16000

# ---------------------------------------------------------------- 1. enveloppe de la voix
wav = '/tmp/sync_voix.wav'
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', AUDIO, '-t', '%.3f' % DUREE,
                '-ac', '1', '-ar', str(SR), wav], check=True)
import wave
with wave.open(wav) as w:
    n = w.getnframes()
    pcm = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32) / 32768.0

FPS_OUT = 30
NF = int(round(DUREE * FPS_OUT))
par_frame = max(1, len(pcm) // NF)
rms = np.array([np.sqrt(np.mean(pcm[i*par_frame:(i+1)*par_frame]**2) + 1e-12)
                for i in range(NF)])

# compression douce puis normalisation : la voix parlee a une dynamique large, on veut une
# commande de mouvement lisible, pas un signal qui saute
env = np.power(rms, 0.5)
lo, hi = np.percentile(env, 5), np.percentile(env, 95)
env = np.clip((env - lo) / max(1e-6, hi - lo), 0, 1)
# lissage : la tete a de l inertie, elle ne suit pas les syllabes
k = 9
env = np.convolve(env, np.ones(k) / k, mode='same')
parle = float((env > 0.25).mean())
print('voix : %.0f%% du temps au-dessus du seuil de parole' % (100 * parle))

# ---------------------------------------------------------------- 2. lecture du clip
cap = cv2.VideoCapture(CLIP)
frames = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    frames.append(fr)
cap.release()
NS = len(frames)
H, W = frames[0].shape[:2]
print('clip source : %d images %dx%d' % (NS, W, H))

# ---------------------------------------------------------------- 3. vitesse pilotee
# vitesse = un plancher tres bas + une part proportionnelle a l energie de la voix.
# Consigne Boss : dans le silence la tete ne bouge PRESQUE PAS, et ca bouge des que ca
# parle. Le plancher n est pas nul : une tete parfaitement figee lit comme une image
# arretee, pas comme quelqu un qui ecoute. A 0,08 le mouvement est a peine perceptible.
base = 0.08 + 0.92 * (1 - FORCE)
vitesse = base + (1.0 - base) * np.power(env, 0.8) * (1 + 0.8 * FORCE)
pos = np.cumsum(vitesse)
periode = 2 * (NS - 1)                       # aller-retour : jamais de coupure
idx = np.mod(pos, periode)
idx = np.where(idx >= NS, periode - idx, idx)  # repli = retour arriere

tmp = DST + '.raw.mp4'
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS_OUT, (W, H))
for i in range(NF):
    vw.write(frames[int(round(idx[i])) % NS])
vw.release()

ENC = ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '19', '-b:v', '0',
       '-pix_fmt', 'yuv420p']
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-i', wav,
                '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '160k',
                '-shortest'] + ENC + [DST], check=True)
os.remove(tmp)

# ---------------------------------------------------------------- 4. preuve de synchro
# On mesure la correlation entre l energie de la voix et la quantite de mouvement image
# par image. Une correlation nettement positive = le personnage bouge quand ca parle.
cap = cv2.VideoCapture(DST); ok, p = cap.read()
p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY); mouv = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    mouv.append(float(cv2.absdiff(g, p).mean())); p = g
cap.release()
m = np.array(mouv); e = env[1:1+len(m)]
r = float(np.corrcoef(m, e)[0, 1]) if len(m) > 10 else 0.0
print('%s : %.0f s @30 im/s' % (DST, DUREE))
print('correlation voix <-> mouvement : %+.2f  (0 = aucun lien, >0,5 = nettement synchro)' % r)
