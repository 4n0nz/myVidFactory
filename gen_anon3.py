#!/usr/bin/env python3
# gen_anon3.py — serie corrigee.
#
# CE QU ON A APPRIS DES SERIES 1 ET 2 (mesure, pas suppose) :
#   - decrire les TRAITS PEINTS d un masque ("painted moustache, pointed beard") fait
#     comprendre au modele qu il s agit des traits DU VISAGE : il rend un homme moustachu
#     a visage nu. Echec total de la serie 1 sur la variante "theatral".
#   - decrire un OBJET lisse sans traits peints ("plain smooth pale mask, narrow dark eye
#     slits, no painted features") rend un VRAI masque rigide, aucune peau visible.
#   - le cadrage tres serre ne montre que les yeux : le buste est le bon cadre.
# D ou la regle appliquee ici : materiau et rigidite D ABORD, traits SCULPTES jamais
# peints, et un negatif ferme sur la peau et la pilosite.
import os, torch
from diffusers import StableDiffusionXLPipeline

OUT = os.path.expanduser('~/avatar_gen/out/anon3')
os.makedirs(OUT, exist_ok=True)
MODEL = os.path.expanduser('~/avatar_gen/models/sdxl')

BASE = ('a person wearing a BLACK hood pulled up, and %s. '
        'The mask is a rigid hard shell covering the ENTIRE face, mask edges clearly '
        'visible against the hood, no skin visible anywhere')
STYLE = ('chest-up portrait, centered, facing camera, dark background with a soft radial glow, '
         'dramatic side lighting, low key, monochrome black and white, high contrast, '
         'dark vignette, photorealistic product-like detail, sharp focus, 85mm')
NEG = ('bare face, human skin, skin texture on face, moustache, beard, facial hair, eyebrows, '
       'visible mouth, teeth, smiling man, portrait of a bearded man, '
       'white hood, light clothing, colorful, saturated, cartoon, anime, 3d render, cgi, '
       'deformed, watermark, text, blurry, low quality')

MASQUES = [
    ('lisse',      'a plain smooth matte white mask with only two narrow dark eye openings and '
                   'no painted features'),
    ('sculpte',    'a smooth pale mask with gently SCULPTED relief - a straight nose ridge, '
                   'raised brow ridges and a closed still mouth - dark eye openings, no paint'),
    ('porcelaine', 'a glossy pale porcelain mask with fine hairline cracks in the glaze, '
                   'calm neutral sculpted features, dark eye openings'),
    ('anguleux',   'a pale angular mask with faceted planes and sharp cheek edges, matte finish, '
                   'two dark eye openings, no paint'),
]

print('chargement SDXL...', flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, variant='fp16', use_safetensors=True).to('cuda')
pipe.set_progress_bar_config(disable=True)

W, H = 1344, 768
for nom, masque in MASQUES:
    for s in (1, 2, 3):
        dst = '%s/anon3_%s_s%d.png' % (OUT, nom, s)
        if os.path.exists(dst):
            print('  saute %s' % os.path.basename(dst)); continue
        img = pipe(prompt='%s, %s' % (BASE % masque, STYLE), negative_prompt=NEG,
                   width=W, height=H, num_inference_steps=34, guidance_scale=7.5,
                   generator=torch.Generator('cuda').manual_seed(s)).images[0]
        img.save(dst)
        print('  ecrit %s' % os.path.basename(dst), flush=True)

print('\n%d images dans %s' % (len(os.listdir(OUT)), OUT))
