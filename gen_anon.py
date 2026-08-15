#!/usr/bin/env python3
# gen_anon.py — serie inspiree de la reference envoyee par Boss :
# noir et blanc contraste, capuche, masque pale theatral, eclairage dur, fond gris degrade.
#
# Les masques sont INTEGRAUX (ils couvrent tout le visage) : la voie d animation qui
# convient est la geometrique (anim_rigid). LivePortrait prendrait les traits PEINTS du
# masque pour un vrai visage et les deformerait comme de la chair — un masque rigide ne
# fait jamais ca. Une variante est passee dans LivePortrait pour montrer la difference.
#
# Deux cadrages par design : AVEC le laptop (fidele a la reference) et SANS (utilisable
# tel quel dans le pipeline, ou l avatar est pose en cover dans les masques hero/pip et
# ou un objet fixe au premier plan encombre).
import os, torch
from diffusers import StableDiffusionXLPipeline

OUT = os.path.expanduser('~/avatar_gen/out/anon')
os.makedirs(OUT, exist_ok=True)
MODEL = os.path.expanduser('~/avatar_gen/models/sdxl')

STYLE = ('black and white photograph, monochrome, high contrast dramatic studio lighting, '
         'grey gradient backdrop, dark vignette, cinematic, photorealistic, sharp focus, '
         'detailed fabric texture, 50mm')
LAPTOP = ('sitting behind a black laptop seen from the front, the closed laptop lid facing '
          'the camera fills the lower third of the frame, hands out of view')
BUSTE = 'chest-up portrait, plain dark clothing, facing camera'
NEG = ('color, colorful, saturated, cartoon, anime, illustration, painting, 3d render, cgi, '
       'deformed face, extra fingers, watermark, text, logo, blurry, low quality, '
       'two people, cropped head')

MASQUES = [
    ('theatral', 'wearing a black hood pulled up and a pale white theatrical mask with a '
                 'painted thin curled moustache, small pointed beard and a fixed calm smile, '
                 'dark eye openings'),
    ('lisse',    'wearing a black hood pulled up and a plain smooth pale white featureless '
                 'mask with only narrow dark eye slits, no painted features'),
    ('porcelaine','wearing a black hood pulled up and a pale cracked porcelain mask with fine '
                 'hairline cracks, calm neutral expression, dark eye openings'),
    ('carnaval', 'wearing a black hood pulled up and a pale venetian carnival mask with a '
                 'long straight nose and arched brow ridges, dark eye openings'),
]

print('chargement SDXL...', flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, variant='fp16', use_safetensors=True).to('cuda')
pipe.set_progress_bar_config(disable=True)

W, H = 1344, 768
for nom, masque in MASQUES:
    for cadre, txt in (('laptop', LAPTOP), ('buste', BUSTE)):
        for s in (1, 2):
            dst = '%s/anon_%s_%s_s%d.png' % (OUT, nom, cadre, s)
            if os.path.exists(dst):
                print('  saute %s' % os.path.basename(dst)); continue
            img = pipe(prompt='a person %s, %s, %s' % (masque, txt, STYLE),
                       negative_prompt=NEG, width=W, height=H,
                       num_inference_steps=34, guidance_scale=7.0,
                       generator=torch.Generator('cuda').manual_seed(s)).images[0]
            img.save(dst)
            print('  ecrit %s' % os.path.basename(dst), flush=True)

print('\n%d images dans %s' % (len(os.listdir(OUT)), OUT))
