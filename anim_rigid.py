#!/usr/bin/env python3
# anim_rigid.py <image> <sortie.mp4> [--dur 12] [--fps 30] [--intensite 1.0]
#
# Anime une image fixe de narrateur a MASQUE INTEGRAL, sans aucun modele.
#
# Pourquoi pas LivePortrait ici : il deforme les traits du visage. Un masque rigide ne
# se deforme PAS — le seul mouvement physiquement juste est celui de la tete entiere.
# On fait donc bouger l image, pas le "visage" : rotation, translation, respiration.
#
# BOUCLE PARFAITE : toutes les frequences sont des multiples ENTIERS de 1/duree, donc
# l etat a t=duree est exactement celui de t=0. Le raccord est invisible, sans fondu.
# Les periodes sont premieres entre elles (2,3,5,7...) pour que la somme ne se repete
# pas a l oreille du regard avant la fin de la boucle.
#
# Le leger sur-cadrage (ZOOM) evite que la rotation decouvre les bords.
import math, os, subprocess, sys
import cv2, numpy as np

SRC, DST = sys.argv[1], sys.argv[2]


def arg(n, d):
    return float(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d


DUR = arg('--dur', 12.0)
FPS = int(arg('--fps', 30))
K = arg('--intensite', 1.0)
ZOOM = 1.06

img = cv2.imread(SRC)
if img is None:
    sys.exit('image illisible : %s' % SRC)
H, W = img.shape[:2]
N = int(round(DUR * FPS))

# amplitudes (en fraction de la largeur / degres), calibrees discretes : on doit sentir
# que ca vit, jamais que ca tangue.
A_ROT, A_TX, A_TY, A_RESP = 0.9 * K, 0.010 * K, 0.007 * K, 0.006 * K

tmp = DST + '.raw.mp4'
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
for i in range(N):
    p = 2 * math.pi * i / N                      # 0..2pi sur la boucle
    rot = A_ROT * (math.sin(2 * p) + 0.45 * math.sin(3 * p + 1.1))
    tx = A_TX * W * (math.sin(3 * p + 0.4) + 0.5 * math.sin(5 * p))
    ty = A_TY * H * (math.sin(2 * p + 2.0) + 0.5 * math.sin(7 * p))
    sc = ZOOM * (1.0 + A_RESP * math.sin(2 * p))  # respiration : 2 cycles par boucle
    M = cv2.getRotationMatrix2D((W / 2, H / 2), rot, sc)
    M[0, 2] += tx
    M[1, 2] += ty
    vw.write(cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REPLICATE))
vw.release()

enc = (['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '19', '-b:v', '0']
       if subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                          'color=c=red:s=320x180:r=30', '-t', '1', '-c:v', 'h264_nvenc',
                          '/tmp/nvprobe_anim.mp4'], capture_output=True).returncode == 0
       else ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18'])
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-an'] + enc +
               ['-pix_fmt', 'yuv420p', DST], check=True)
os.remove(tmp)

# preuve de boucle : la derniere frame doit etre quasi identique a la premiere
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', DST, '-vf', 'select=eq(n\\,0)',
                '-frames:v', '1', '/tmp/loop_f0.png'], check=True)
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-sseof', '-0.1', '-i', DST,
                '-frames:v', '1', '/tmp/loop_fn.png'], check=True)
a, b = cv2.imread('/tmp/loop_f0.png'), cv2.imread('/tmp/loop_fn.png')
mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
psnr = 10 * math.log10(255 ** 2 / mse) if mse > 0 else 99
print('%s  %.0fs @%dfps  %dx%d' % (DST, DUR, FPS, W, H))
print('raccord de boucle : PSNR %.1f dB entre derniere et premiere frame '
      '(>35 dB = raccord invisible)' % psnr)
