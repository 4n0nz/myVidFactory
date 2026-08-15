#!/usr/bin/env python3
# anim_one.py <image> [nom] — chaine complete pour UNE image :
#   SVD (2 amplitudes) -> interpolation 30 im/s -> aller-retour -> mesures.
#
# Corrige un defaut des essais precedents : un resize direct vers 1024x576 (ratio 1,778)
# depuis une source de ratio different ETIRE le visage. On recadre au centre d abord.
import os, subprocess, sys
import numpy as np, torch, cv2
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

SRC = sys.argv[1]
NOM = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(SRC))[0]
BASE = os.path.expanduser('~/avatar_gen')
OUT = '%s/out/anim_%s' % (BASE, NOM)
os.makedirs(OUT, exist_ok=True)
W, H = 1024, 576
MOTIONS = [90, 160]


def nvenc_ok():
    return subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                           'color=c=red:s=320x180:r=30', '-t', '1', '-c:v', 'h264_nvenc',
                           '/tmp/nvp.mp4'], capture_output=True).returncode == 0


ENC = (['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '19', '-b:v', '0']
       if nvenc_ok() else ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18']) + \
      ['-pix_fmt', 'yuv420p']


def prepare(p):
    im = Image.open(p).convert('RGB')
    w, h = im.size
    cible = W / H
    if abs(w / h - cible) > 0.001:                 # recadrage centre, jamais d etirement
        if w / h > cible:
            nw = int(round(h * cible)); x = (w - nw) // 2
            im = im.crop((x, 0, x + nw, h))
        else:
            nh = int(round(w / cible)); y = (h - nh) // 2
            im = im.crop((0, y, w, y + nh))
        print('  recadre %dx%d -> %dx%d (ratio preserve)' % (w, h, *im.size))
    return im.resize((W, H), Image.LANCZOS)


def ff(args):
    subprocess.run(['ffmpeg', '-y', '-v', 'error'] + args, check=True)


def mesures(p):
    cap = cv2.VideoCapture(p); ok, prev = cap.read()
    if not ok:
        return None
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY); h, w = prev.shape
    c, b, d = [], [], []
    while len(d) < 120:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(g, prev).astype(np.float32)
        c.append(diff[h//4:3*h//4, w//3:2*w//3].mean())
        b.append(np.concatenate([diff[:, :w//8].ravel(), diff[:, -w//8:].ravel()]).mean())
        d.append(float(diff.mean())); prev = g
    cap.release()
    a = np.array(d)
    return dict(centre=float(np.mean(c)), bords=float(np.mean(b)),
                variation=float(a.std()/a.mean()) if a.mean() > 0 else 0,
                figees=int((a < 0.05).sum()))


print('chargement SVD...', flush=True)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    os.path.join(BASE, 'models/svd'), torch_dtype=torch.float16, variant='fp16')
pipe.enable_model_cpu_offload()

img = prepare(SRC)
img.save('%s/%s_source_1024x576.png' % (OUT, NOM))

for m in MOTIONS:
    brut = '%s/%s_m%d_brut7fps.mp4' % (OUT, NOM, m)
    interp = '%s/%s_m%d_30fps.mp4' % (OUT, NOM, m)
    loop = '%s/%s_m%d_30fps_boucle.mp4' % (OUT, NOM, m)
    if not os.path.exists(brut):
        frames = pipe(img, decode_chunk_size=4, motion_bucket_id=m,
                      noise_aug_strength=0.05, num_frames=25,
                      generator=torch.Generator('cuda').manual_seed(1)).frames[0]
        export_to_video(frames, brut, fps=7)
        print('  genere %s' % os.path.basename(brut), flush=True)
    ff(['-i', brut, '-vf',
        'minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1',
        '-an'] + ENC + [interp])
    ff(['-i', interp, '-filter_complex',
        '[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[out]',
        '-map', '[out]', '-an'] + ENC + [loop])
    for p in (brut, interp, loop):
        r = mesures(p)
        if r:
            print('  %-38s centre %5.2f | bords %5.2f | variation %4.2f | figees %2d'
                  % (os.path.basename(p), r['centre'], r['bords'], r['variation'], r['figees']))

print('\nsorties dans %s' % OUT)
