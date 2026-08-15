#!/usr/bin/env python3
# wan_chain.py <image> [nom] [--maillons 4] — clip long par CHAINAGE Wan 2.2.
#
# Wan sort 49 images a 24 im/s, soit 2 s. Trop court pour les segments hero (6,4 s en
# mediane). Deux facons d allonger :
#   - aller-retour : double la duree mais le mouvement se rejoue a l envers, ca se voit
#     des que le geste est marque ;
#   - CHAINAGE (retenu) : la DERNIERE image d un clip devient l image d entree du suivant.
#     Le mouvement continue vraiment, et la duree n est plus bornee.
#
# Limite du chainage, mesuree ici et pas supposee : chaque maillon repart d une image deja
# passee par le VAE, donc le grain et le contraste derivent un peu a chaque fois. On borne
# a quelques maillons et on affiche la derive (ecart de luminance/contraste au depart) pour
# savoir a partir de quand ca se degrade.
import os, subprocess, sys
import numpy as np, torch
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image

SRC = sys.argv[1]
NOM = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else 'chain'
MAILLONS = int(sys.argv[sys.argv.index('--maillons') + 1]) if '--maillons' in sys.argv else 4
# graine de depart : en la changeant on obtient un GESTE different depuis la MEME image,
# donc le meme personnage. C est ce qui permet de constituer une banque de mouvements
# plutot que de rejouer une seule boucle.
SEED0 = int(sys.argv[sys.argv.index('--seed0') + 1]) if '--seed0' in sys.argv else 1
BASE = os.path.expanduser('~/avatar_gen')
MODEL = os.path.join(BASE, 'models/wan22')
OUT = '%s/out/wanchain_%s' % (BASE, NOM)
os.makedirs(OUT, exist_ok=True)

W, H, NF = 1280, 704, 49
PROMPT = ('a hooded person wearing a rigid pale mask slowly turns their head and shoulders, '
          'subtle natural head motion, the mask stays a solid rigid object keeping its exact '
          'shape, static camera, fixed dark background, cinematic low light')
NEG = ('camera movement, zoom, panning, the mask deforming, melting or warping, face changing '
       'shape, distorted features, morphing, flickering, background moving, blurry, low quality')


def prepare(im):
    w, h = im.size
    cible = W / H
    if abs(w / h - cible) > 0.001:
        if w / h > cible:
            nw = int(round(h * cible)); x = (w - nw) // 2
            im = im.crop((x, 0, x + nw, h))
        else:
            nh = int(round(w / cible)); y = (h - nh) // 2
            im = im.crop((0, y, w, y + nh))
    return im.resize((W, H), Image.LANCZOS)


print('chargement Wan 2.2...', flush=True)
vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder='vae', torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(MODEL, vae=vae, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

img = prepare(Image.open(SRC).convert('RGB'))
img.save('%s/%s_source.png' % (OUT, NOM))
ref = np.asarray(img.convert('L'), dtype=np.float32)
print('depart : luminance %.1f | contraste %.1f' % (ref.mean(), ref.std()))

morceaux = []
for m in range(MAILLONS):
    frames = pipe(image=img, prompt=PROMPT, negative_prompt=NEG,
                  height=H, width=W, num_frames=NF,
                  guidance_scale=5.0, num_inference_steps=40,
                  generator=torch.Generator('cuda').manual_seed(SEED0 + m)).frames[0]
    # on retire la 1re image des maillons suivants : c est la derniere du precedent
    utiles = frames if m == 0 else frames[1:]
    morceaux.extend(utiles)
    img = frames[-1] if isinstance(frames[-1], Image.Image) else Image.fromarray(
        (np.asarray(frames[-1]) * 255).astype(np.uint8))
    img = prepare(img.convert('RGB'))
    g = np.asarray(img.convert('L'), dtype=np.float32)
    print('  maillon %d/%d : %d images | luminance %.1f (%+.1f) | contraste %.1f (%+.1f)'
          % (m + 1, MAILLONS, len(utiles), g.mean(), g.mean() - ref.mean(),
             g.std(), g.std() - ref.std()), flush=True)

brut = '%s/%s_24fps.mp4' % (OUT, NOM)
export_to_video(morceaux, brut, fps=24)

# 24 -> 30 im/s : le pipeline travaille en 30, et l interpolation avec compensation de
# mouvement est deja validee sur les essais SVD.
final = '%s/%s_30fps.mp4' % (OUT, NOM)
enc = ['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '19', '-b:v', '0',
       '-pix_fmt', 'yuv420p']
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', brut, '-vf',
                'minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1',
                '-an'] + enc + [final], check=True)

d = float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
                                   'format=duration', '-of', 'csv=p=0', final]))
print('\n%s : %d images, %.1f s a 30 im/s' % (final, len(morceaux), d))
