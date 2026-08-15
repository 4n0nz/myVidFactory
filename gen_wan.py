#!/usr/bin/env python3
# gen_wan.py <image> [nom] — image-to-video avec Wan 2.2 (Apache 2.0).
#
# POURQUOI CHANGER DE MODELE : SVD date de fin 2023 et fait fondre les objets rigides —
# c est ce qui deformait le masque ("pas coherent", verdict Boss). Wan 2.2 est de 2025 et
# tient bien mieux la coherence d objet. Meme entree, meme sortie : seul le modele change,
# ce qui rend la comparaison honnete.
#
# Le prompt sert a CADRER le mouvement, pas a inventer la scene : on demande explicitement
# une tete qui bouge et un masque rigide, et on interdit dans le negatif la deformation du
# masque et le mouvement de camera (les deux echecs precedents).
import os, sys, torch
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image

SRC = sys.argv[1]
NOM = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(os.path.basename(SRC))[0]
BASE = os.path.expanduser('~/avatar_gen')
MODEL = os.path.join(BASE, 'models/wan22')
OUT = '%s/out/wan_%s' % (BASE, NOM)
os.makedirs(OUT, exist_ok=True)

# dimensions parametrables : les sources ne sont pas toutes en paysage. Multiples de 16
# obligatoires. 1280x704 = paysage (cadre hero) ; 704x1056 = portrait (cadre pip, et
# cadrage naturel d une photo verticale).
W = int(sys.argv[sys.argv.index('--w') + 1]) if '--w' in sys.argv else 1280
H = int(sys.argv[sys.argv.index('--h') + 1]) if '--h' in sys.argv else 704
NUM_FRAMES = 49           # ~2 s a 24 im/s ; on allongera par boucle si le rendu convient

PROMPT = ('a hooded person wearing a rigid pale mask slowly turns their head and shoulders, '
          'subtle natural head motion, the mask stays a solid rigid object keeping its exact '
          'shape, static camera, fixed dark background, cinematic low light')
NEG = ('camera movement, zoom, panning, the mask deforming, melting or warping, face changing '
       'shape, distorted features, morphing, flickering, background moving, blurry, low quality')


def prepare(p):
    im = Image.open(p).convert('RGB')
    w, h = im.size
    cible = W / H
    if abs(w / h - cible) > 0.001:              # recadrage centre, jamais d etirement
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
pipe.enable_model_cpu_offload()                 # 16 Go de VRAM : marge obligatoire
# Le decodage VAE de toutes les images d un coup depasse la carte : OOM mesure a l etape
# finale (1,72 Go demandes, 1,57 disponible). En tuiles + tranches, on decode par morceaux
# au lieu du volume entier, et le pic retombe.
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

img = prepare(SRC)
img.save('%s/%s_source.png' % (OUT, NOM))

frames = pipe(image=img, prompt=PROMPT, negative_prompt=NEG,
              height=H, width=W, num_frames=NUM_FRAMES,
              guidance_scale=5.0, num_inference_steps=40,
              generator=torch.Generator('cuda').manual_seed(1)).frames[0]

dst = '%s/%s_wan22.mp4' % (OUT, NOM)
export_to_video(frames, dst, fps=24)
print('ecrit %s (%d images)' % (dst, len(frames)))
