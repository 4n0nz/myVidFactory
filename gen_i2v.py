#!/usr/bin/env python3
# gen_i2v.py — VRAIE animation : image-to-video (Stable Video Diffusion).
#
# Pourquoi on jette anim_rigid : il applique une transformation affine a l IMAGE ENTIERE.
# Fond, epaules et tete bougent ensemble = mouvement de camera sur une photo. Le sujet
# lui-meme ne bouge pas d un pixel. Verdict Boss, et il a raison.
#
# SVD conditionne UNIQUEMENT sur l image (aucun encodeur de texte a charger, ~9,5 Go) et
# genere un mouvement propre au sujet : la tete tourne, le tissu bouge, le fond reste.
# motion_bucket_id pilote l amplitude — on en sort plusieurs pour comparer a l oeil.
import os, sys, torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image

MODEL = os.path.expanduser('~/avatar_gen/models/svd')
OUT = os.path.expanduser('~/avatar_gen/out/i2v')
os.makedirs(OUT, exist_ok=True)

SOURCES = sys.argv[1:] or [
    os.path.expanduser('~/avatar_gen/out/anon4/anon4_porcelaine_s1.png'),
    os.path.expanduser('~/avatar_gen/out/anon4/anon4_lisse_s1.png'),
]
# amplitudes : 90 = discret (un narrateur qui ecoute), 160 = marque (il parle)
MOTIONS = [90, 160]

print('chargement SVD...', flush=True)
pipe = StableVideoDiffusionPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, variant='fp16')
pipe.enable_model_cpu_offload()          # 16 Go de VRAM : on garde de la marge

for src in SOURCES:
    if not os.path.exists(src):
        print('  absent : %s' % src); continue
    base = os.path.splitext(os.path.basename(src))[0]
    img = Image.open(src).convert('RGB').resize((1024, 576), Image.LANCZOS)
    for m in MOTIONS:
        dst = '%s/%s_m%d.mp4' % (OUT, base, m)
        if os.path.exists(dst):
            print('  saute %s' % os.path.basename(dst)); continue
        frames = pipe(img,
                      decode_chunk_size=4,          # VRAM : decode par petits paquets
                      motion_bucket_id=m,
                      noise_aug_strength=0.05,
                      num_frames=25,
                      generator=torch.Generator('cuda').manual_seed(1)).frames[0]
        export_to_video(frames, dst, fps=7)
        print('  ecrit %s (%d frames)' % (os.path.basename(dst), len(frames)), flush=True)

print('\n%d clips dans %s' % (len([f for f in os.listdir(OUT) if f.endswith('.mp4')]), OUT))
