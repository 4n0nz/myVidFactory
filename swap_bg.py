#!/usr/bin/env python3
# swap_bg.py <video> [nom] — detoure le narrateur et le repose sur d autres fonds.
#
# On detoure APRES generation plutot que de regenerer avec un decor impose : Wan animerait
# aussi le nouveau fond, alors qu ici il reste exactement ce qu on veut.
#
# Piege du detourage image par image : le contour est recalcule a chaque image, donc les
# bords GRELOTTENT. On lisse le canal alpha dans le TEMPS (moyenne glissante sur 5 images)
# — le sujet bouge lentement, donc le lissage ne mange pas le mouvement, il ne tue que le
# scintillement.
import os, subprocess, sys
import cv2, numpy as np
from rembg import new_session, remove
from PIL import Image

SRC = sys.argv[1]
NOM = sys.argv[2] if len(sys.argv) > 2 else 'narrateur'
BASE = os.path.expanduser('~/avatar_gen')
OUT = '%s/out/bg_%s' % (BASE, NOM)
os.makedirs(OUT, exist_ok=True)
MATRIX = os.path.expanduser('~/videogen/assets/background.mp4')

cap = cv2.VideoCapture(SRC)
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print('source %dx%d @%.0f im/s' % (W, H, FPS))

sess = new_session('u2net')
rgb, alphas = [], []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    out = remove(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), session=sess)
    a = np.asarray(out)[:, :, 3].astype(np.float32) / 255.0
    rgb.append(fr); alphas.append(a)
cap.release()
N = len(rgb)
print('%d images detourees' % N)

# lissage temporel du contour (fenetre 5) — supprime le grelottement des bords
A = np.stack(alphas)
K = 5
lisse = np.empty_like(A)
for i in range(N):
    lo, hi = max(0, i - K // 2), min(N, i + K // 2 + 1)
    lisse[i] = A[lo:hi].mean(axis=0)
couv = float((lisse > 0.5).mean())
print('sujet = %.1f%% de l image | stabilite du contour : ecart moyen %.4f'
      % (100 * couv, float(np.abs(np.diff(lisse, axis=0)).mean())))

ENC = ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '19', '-b:v', '0',
       '-pix_fmt', 'yuv420p']


def ecrire(nom, fabrique_fond):
    tmp = '%s/%s.raw.mp4' % (OUT, nom)
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    for i in range(N):
        a = lisse[i][..., None]
        vw.write((rgb[i].astype(np.float32) * a +
                  fabrique_fond(i).astype(np.float32) * (1 - a)).astype(np.uint8))
    vw.release()
    dst = '%s/%s_%s.mp4' % (OUT, NOM, nom)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-an'] + ENC + [dst], check=True)
    os.remove(tmp)
    print('  ecrit %s' % os.path.basename(dst))


ecrire('noir', lambda i: np.zeros((H, W, 3), np.uint8))
# vert chroma : permet de composer n importe quel fond plus tard, y compris dans le pipeline
ecrire('vert', lambda i: np.full((H, W, 3), (0, 255, 0), np.uint8))

# fond Matrix du projet (assets/background.mp4), boucle s il est plus court
if os.path.exists(MATRIX):
    mc = cv2.VideoCapture(MATRIX)
    fonds = []
    while len(fonds) < N:
        ok, f = mc.read()
        if not ok:
            mc.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = mc.read()
            if not ok:
                break
        fonds.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA))
    mc.release()
    if len(fonds) >= N:
        ecrire('matrix', lambda i: fonds[i])

# version a fond transparent (VP9 alpha) : pour composer ailleurs sans repasser par ici
tmpdir = '%s/png' % OUT
os.makedirs(tmpdir, exist_ok=True)
for i in range(N):
    rgba = np.dstack([rgb[i], (lisse[i] * 255).astype(np.uint8)])
    cv2.imwrite('%s/f%04d.png' % (tmpdir, i), rgba)
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', str(int(FPS)),
                '-i', '%s/f%%04d.png' % tmpdir, '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p',
                '-b:v', '2M', '%s/%s_alpha.webm' % (OUT, NOM)], check=True)
subprocess.run(['rm', '-rf', tmpdir])
print('  ecrit %s_alpha.webm (fond transparent)' % NOM)
print('\nsorties dans %s' % OUT)
