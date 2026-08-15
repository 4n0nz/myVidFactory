#!/usr/bin/env python3
# gen_face_sdxl.py — candidats de narrateur masque (SDXL base, tient entier en 16 Go).
#
# Deux familles, parce qu elles n ont PAS la meme voie d animation :
#   INTEGRAL  — masque rigide couvrant tout le visage. Aucun repere facial exploitable :
#               LivePortrait est inadapte (il deformerait le masque comme de la chair,
#               ce qu un masque rigide ne fait jamais). Animation = geometrique.
#   PARTIEL   — bas du visage couvert, YEUX VISIBLES. Reperes detectables, donc
#               LivePortrait anime regard et clignements. Bouche cachee = zero lip sync.
#
# Masques ORIGINAUX volontairement : le Guy Fawkes de l avatar actuel est un design
# sous droits (Warner Bros), inconfortable des qu il y a monetisation.
import os, torch
from diffusers import StableDiffusionXLPipeline

OUT = os.path.expanduser('~/avatar_gen/out/faces')
os.makedirs(OUT, exist_ok=True)
MODEL = os.path.expanduser('~/avatar_gen/models/sdxl')

COMMUN = ('cinematic portrait, dark studio background with faint green code glow, '
          'moody low-key lighting, rim light, chest-up framing, facing camera, '
          'photorealistic, highly detailed, sharp focus, 35mm, shallow depth of field')
NEG = ('cartoon, anime, illustration, painting, cgi, 3d render, deformed, distorted, '
       'extra limbs, watermark, text, logo, blurry, low quality, guy fawkes mask, '
       'v for vendetta mask')

VARIANTES = [
    ('integral_lisse',   'a person wearing a black hoodie and a smooth matte black full-face mask '
                         'with thin glowing green line accents, featureless mask, no visible eyes, hood up'),
    ('integral_chrome',  'a person wearing a dark hoodie and a polished chrome mirror full-face mask '
                         'reflecting green light, featureless, hood up'),
    ('integral_mesh',    'a person wearing a black hoodie and a dark perforated metal mesh full-face '
                         'mask, faint light through the mesh, hood up'),
    ('integral_visiere', 'a person wearing a black technical hoodie, a wraparound reflective visor '
                         'covering the eyes and a smooth black panel covering the lower face, hood up'),
    ('partiel_cagoule',  'a person wearing a black hoodie and a black balaclava covering nose and '
                         'mouth, eyes clearly visible and well lit, expressive human eyes, hood up'),
    ('partiel_rouge',    'a person wearing a dark hoodie and a black cloth face covering over the '
                         'lower face, eyes clearly visible, red rim lighting from the side, hood up'),
]

print('chargement SDXL...', flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to('cuda')
pipe.set_progress_bar_config(disable=True)

W, H = 1344, 768        # 16:9 natif SDXL ; l avatar actuel n est qu en 848x464
SEEDS = [1, 2]

for nom, sujet in VARIANTES:
    for s in SEEDS:
        dst = '%s/%s_s%d.png' % (OUT, nom, s)
        if os.path.exists(dst):
            print('  saute %s' % os.path.basename(dst)); continue
        img = pipe(prompt='%s, %s' % (sujet, COMMUN), negative_prompt=NEG,
                   width=W, height=H, num_inference_steps=32, guidance_scale=6.5,
                   generator=torch.Generator('cuda').manual_seed(s)).images[0]
        img.save(dst)
        print('  ecrit %s' % os.path.basename(dst), flush=True)

print('\n%d images dans %s' % (len(os.listdir(OUT)), OUT))
