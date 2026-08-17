#!/usr/bin/env python3
# gen_bg.py — decors de poste de travail, generes donc libres de droits, vires au vert.
#
# Remplace l image de magazine (filigranee) par un fond a nous : meme ambiance — plusieurs
# ecrans de code, eclairage LED, piece sombre — mais en 1344x768 natif, sans filigrane et
# sans question de licence.
#
# La teinte est tournee APRES generation, comme pour l avatar : on demande un decor bleu
# (le modele rend mieux cette ambiance, tres representee) puis on bascule vers le vert.
import os, subprocess, torch
from diffusers import StableDiffusionXLPipeline

OUT = os.path.expanduser('~/avatar_gen/out/bg_gen')
os.makedirs(OUT, exist_ok=True)
MODEL = os.path.expanduser('~/avatar_gen/models/sdxl')

COMMUN = ('dark room, moody LED ambient lighting, cinematic, photorealistic, highly detailed, '
          'sharp focus, wide angle, depth of field, no people')
NEG = ('people, person, hands, face, text overlay, watermark, logo, signature, caption, '
       'blurry, low quality, cartoon, illustration, distorted screens')

VARIANTES = [
    ('poste_3ecrans', 'a desk with three large monitors showing code editors, terminals and '
                      'data dashboards, mechanical keyboard, mouse, LED strip glow behind the '
                      'desk, dark home office'),
    ('poste_courbe',  'a curved ultrawide monitor on a wooden desk displaying terminal windows '
                      'and world map dashboards, laptop beside it, LED backlight, dark room'),
    ('mur_serveurs',  'a wall of server racks with status lights, a desk in front with monitors '
                      'showing terminal output, dark data center room'),
    ('bureau_laptop', 'a laptop open on a desk showing a terminal full of code, second monitor '
                      'behind, plant, headphones, LED strip, dark room at night'),
]

print('chargement SDXL...', flush=True)
pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, variant='fp16', use_safetensors=True).to('cuda')
pipe.set_progress_bar_config(disable=True)

W, H = 1344, 768
for nom, sujet in VARIANTES:
    for s in (1, 2):
        base = '%s/%s_s%d' % (OUT, nom, s)
        if os.path.exists(base + '_vert.png'):
            print('  saute %s' % os.path.basename(base)); continue
        img = pipe(prompt='%s, %s' % (sujet, COMMUN), negative_prompt=NEG,
                   width=W, height=H, num_inference_steps=34, guidance_scale=6.5,
                   generator=torch.Generator('cuda').manual_seed(s)).images[0]
        img.save(base + '.png')
        # meme bascule que pour l avatar : on mesure, puis on tourne vers 120 deg
        import cv2, numpy as np
        im = cv2.imread(base + '.png')
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        m = (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60)
        teinte = int(np.median(hsv[:, :, 0][m])) * 2 if m.sum() > 500 else 214
        rot = 120 - teinte
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', base + '.png',
                        '-vf', 'hue=h=%d,scale=1920:1080:flags=lanczos' % rot,
                        base + '_vert.png'], check=True)
        im2 = cv2.imread(base + '_vert.png')
        hsv2 = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
        m2 = (hsv2[:, :, 1] > 80) & (hsv2[:, :, 2] > 60)
        t2 = int(np.median(hsv2[:, :, 0][m2])) * 2 if m2.sum() > 500 else -1
        os.remove(base + '.png')
        print('  %-22s teinte %3d -> %3d deg (vert=120)' % (nom + '_s%d' % s, teinte, t2),
              flush=True)

print('\n%d decors dans %s' % (len([f for f in os.listdir(OUT) if f.endswith('_vert.png')]), OUT))
