#!/usr/bin/env python3
# gen_anon2.py — serie calee sur la 2e reference : portrait SERRE tete-epaules, capuche,
# masque pale, fond sombre avec halo radial doux, ton desature chaud, lumiere laterale
# douce. Aucun objet au premier plan.
#
# C est le cadrage le plus utile pour le pipeline : l avatar est pose en cover dans les
# masques hero (paysage) ET pip (souvent portrait). Un sujet serre et centre, sur fond
# uni, se recadre dans les deux sans rien perdre — contrairement au plan avec laptop.
#
# Masques integraux : animation geometrique (anim_rigid), pas LivePortrait.
import os, torch
from diffusers import StableDiffusionXLPipeline

OUT = os.path.expanduser('~/avatar_gen/out/anon2')
os.makedirs(OUT, exist_ok=True)
MODEL = os.path.expanduser('~/avatar_gen/models/sdxl')

STYLE = ('close-up head and shoulders portrait, centered, dark background with a soft radial '
         'glow behind the subject, gentle side lighting, low key, desaturated warm sepia tone, '
         'shallow depth of field, photorealistic, fine detail, cinematic, 85mm')
NEG = ('bright, colorful, saturated, cartoon, anime, illustration, painting, 3d render, cgi, '
       'deformed, watermark, text, logo, blurry, low quality, full body, wide shot, '
       'laptop, computer, hands, two people')

MASQUES = [
    ('theatral',  'a black hood pulled up over a pale white theatrical mask with a painted thin '
                  'curled moustache, small pointed beard and a fixed calm smile, dark eye openings'),
    ('lisse',     'a black hood pulled up over a plain smooth pale mask with only narrow dark eye '
                  'slits and no painted features'),
    ('porcelaine','a black hood pulled up over a pale cracked porcelain mask with fine hairline '
                  'cracks and a calm neutral expression, dark eye openings'),
    ('sculpte',   'a black hood pulled up over a pale sculpted mask with sharp cheekbones, a '
                  'straight nose and a closed serene mouth, dark eye openings'),
]

print('chargement SDXL...', flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, variant='fp16', use_safetensors=True).to('cuda')
pipe.set_progress_bar_config(disable=True)

W, H = 1344, 768
for nom, masque in MASQUES:
    for s in (1, 2, 3):
        dst = '%s/anon2_%s_s%d.png' % (OUT, nom, s)
        if os.path.exists(dst):
            print('  saute %s' % os.path.basename(dst)); continue
        img = pipe(prompt='a person wearing %s, %s' % (masque, STYLE),
                   negative_prompt=NEG, width=W, height=H,
                   num_inference_steps=34, guidance_scale=7.0,
                   generator=torch.Generator('cuda').manual_seed(s)).images[0]
        img.save(dst)
        print('  ecrit %s' % os.path.basename(dst), flush=True)

print('\n%d images dans %s' % (len(os.listdir(OUT)), OUT))
