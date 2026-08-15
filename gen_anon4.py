#!/usr/bin/env python3
# gen_anon4.py — on repart du SEUL prompt qui a produit a la fois un vrai masque ET un
# cadrage correct (serie 1, variante "lisse", cadrage buste), et on ne fait varier QU UNE
# chose : le design du masque.
#
# Historique des echecs, pour ne pas les refaire :
#   serie 1 "theatral" : decrire des traits PEINTS -> visage humain moustachu, pas de masque.
#   serie 2 (cadrage tres serre) -> ne montre que les yeux.
#   serie 3 (prompt long + "product-like detail") -> vrais masques mais composition perdue :
#           macro sur les yeux, et meme un diptyque facon planche de catalogue.
# Leçon : SDXL tient la composition sur un prompt COURT. Chaque contrainte ajoutee lui en
# fait lacher une autre.
import os, torch
from diffusers import StableDiffusionXLPipeline

OUT = os.path.expanduser('~/avatar_gen/out/anon4')
os.makedirs(OUT, exist_ok=True)
MODEL = os.path.expanduser('~/avatar_gen/models/sdxl')

# --- la recette qui a marche, mot pour mot ---
QUEUE = ('chest-up portrait, plain dark clothing, facing camera, black and white photograph, '
         'monochrome, high contrast dramatic studio lighting, grey gradient backdrop, '
         'dark vignette, cinematic, photorealistic, sharp focus, 50mm')
NEG = ('diptych, split image, two people, multiple views, collage, macro, extreme close-up, '
       'bare face, human skin on face, moustache, beard, facial hair, '
       'color, cartoon, 3d render, watermark, text, blurry, low quality')

MASQUES = [
    ('lisse',      'a plain smooth pale white featureless mask with only narrow dark eye slits, '
                   'no painted features'),
    ('ovale',      'a plain smooth pale oval mask with two dark almond eye openings and a faint '
                   'nose ridge, no painted features'),
    ('porcelaine', 'a pale porcelain mask with fine hairline cracks and two dark eye openings, '
                   'no painted features'),
    ('anguleux',   'a pale angular faceted mask with sharp cheek planes and two dark eye '
                   'openings, no painted features'),
]

print('chargement SDXL...', flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, variant='fp16', use_safetensors=True).to('cuda')
pipe.set_progress_bar_config(disable=True)

W, H = 1344, 768
for nom, masque in MASQUES:
    for s in (1, 2, 3):
        dst = '%s/anon4_%s_s%d.png' % (OUT, nom, s)
        if os.path.exists(dst):
            print('  saute %s' % os.path.basename(dst)); continue
        img = pipe(prompt='a person wearing a black hood pulled up and %s, %s' % (masque, QUEUE),
                   negative_prompt=NEG, width=W, height=H,
                   num_inference_steps=34, guidance_scale=7.0,
                   generator=torch.Generator('cuda').manual_seed(s)).images[0]
        img.save(dst)
        print('  ecrit %s' % os.path.basename(dst), flush=True)

print('\n%d images dans %s' % (len(os.listdir(OUT)), OUT))
