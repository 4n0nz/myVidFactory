#!/usr/bin/env python3
# boucle60.py <clip.webm> <sortie.webm> [--dur 60] [--fps 30]
#
# Transforme un clip court (2 s) en boucle longue qui ne se voit pas tourner.
#
# NAIF : repeter le clip 30 fois -> saut a chaque raccord. Aller-retour simple -> pas de
# saut, mais 15 cycles identiques qui battent comme un metronome.
#
# ICI : parcours aller-retour dont la VITESSE varie. La phase avance selon
#     phi(t) = 2pi*m*t/D + a*sin(2pi*t/D) + b*sin(4pi*t/D + c)
# et la position suit (1 - cos(phi))/2. Toutes les composantes ont une periode qui DIVISE
# la duree D, donc phi(D) = phi(0) + 2pi*m : la boucle est exacte au raccord, sans fondu.
# Les termes en sin font que chaque aller-retour n a ni la meme duree ni la meme vitesse,
# ce qui casse la periodicite percue.
#
# L alpha est conserve de bout en bout (VP9), avec -auto-alt-ref 0 sans quoi il est perdu
# silencieusement.
import math, os, subprocess, sys
import numpy as np

SRC, DST = sys.argv[1], sys.argv[2]


def arg(n, d):
    return float(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d


D = arg('--dur', 60.0)
FPS = int(arg('--fps', 30))
TMP = '/tmp/boucle60_%d' % os.getpid()
os.makedirs(TMP, exist_ok=True)

# --- images source en RGBA (l alpha doit survivre a l extraction)
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-c:v', 'libvpx-vp9', '-i', SRC,
                '-vf', 'format=rgba', '%s/s%%04d.png' % TMP], check=True)
src = sorted(os.listdir(TMP))
NS = len(src)
if NS < 2:
    sys.exit('clip source trop court')

NF = int(round(D * FPS))
m = max(1, int(round(D / (2 * NS / FPS))))     # nombre d aller-retours sur la duree
a, b, c = 0.55, 0.28, 1.3                      # modulation de vitesse (radians)

t = np.arange(NF) / NF                          # 0..1 sur la boucle
phi = 2 * math.pi * m * t + a * np.sin(2 * math.pi * t) + b * np.sin(4 * math.pi * t + c)
pos = (1 - np.cos(phi)) / 2 * (NS - 1)          # 0..NS-1, aller-retour continu
idx = np.clip(np.round(pos).astype(int), 0, NS - 1)

seq = TMP + '/seq'
os.makedirs(seq, exist_ok=True)
for i, k in enumerate(idx):
    os.link(os.path.join(TMP, src[k]), '%s/f%05d.png' % (seq, i))

subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', str(FPS),
                '-i', '%s/f%%05d.png' % seq, '-c:v', 'libvpx-vp9',
                '-pix_fmt', 'yuva420p', '-b:v', '3M',
                '-auto-alt-ref', '0', '-lag-in-frames', '0', DST], check=True)

alpha_ok = subprocess.run(['ffmpeg', '-v', 'error', '-c:v', 'libvpx-vp9', '-i', DST,
                           '-vf', 'alphaextract', '-frames:v', '1', '-f', 'null', '-'],
                          capture_output=True).returncode == 0
subprocess.run(['rm', '-rf', TMP])

# le raccord est-il exact ? la position a la fin doit rejoindre celle du debut
ecart = abs(int(idx[0]) - int(idx[-1]))
distinctes = len(set(idx.tolist()))
print('%s : %.0f s @%d im/s | %d aller-retours | %d/%d images source utilisees'
      % (os.path.basename(DST), D, FPS, m, distinctes, NS))
print('  raccord : image %d -> %d (ecart %d) | alpha %s'
      % (idx[-1], idx[0], ecart, 'PRESENT' if alpha_ok else 'PERDU'))
