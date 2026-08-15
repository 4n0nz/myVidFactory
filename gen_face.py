#!/usr/bin/env python3
# gen_face.py — genere les candidats de narrateur masque (Flux.1-schnell, Apache 2.0).
#
# Deux familles, parce qu elles n ont PAS la meme voie d animation ensuite :
#   INTEGRAL  — masque rigide couvrant tout le visage. Aucun repere facial exploitable :
#               LivePortrait est inadapte (il deformerait le masque comme de la chair,
#               ce qu un masque rigide ne fait jamais). Animation = geometrique.
#   PARTIEL   — bas du visage couvert, YEUX VISIBLES. Reperes detectables, donc
#               LivePortrait anime regard et clignements. Bouche cachee = zero lip sync.
#
# Masques ORIGINAUX volontairement : le Guy Fawkes de l avatar actuel est un design
# sous droits (Warner Bros), inconfortable des qu il y a monetisation.
import os, sys, torch
from diffusers import FluxPipeline

OUT = os.path.expanduser('~/avatar_gen/out/faces')
os.makedirs(OUT, exist_ok=True)
MODEL = os.path.expanduser('~/avatar_gen/models/flux-schnell')

COMMUN = ('cinematic portrait, dark studio background with faint green code-like glow, '
          'moody low-key lighting, shallow depth of field, chest-up framing, facing camera, '
          'photorealistic, sharp focus, 35mm')

VARIANTES = [
    ('integral_lisse',  'a person in a black hoodie wearing a smooth matte black full-face mask '
                        'with thin glowing green line accents, no facial features visible, hood up'),
    ('integral_chrome', 'a person in a dark hoodie wearing a polished chrome mirror full-face mask '
                        'reflecting green light, featureless, hood up'),
    ('integral_mesh',   'a person in a black hoodie wearing a dark perforated mesh full-face mask, '
                        'faint light passing through the mesh, hood up'),
    ('integral_visiere','a person in a black technical hoodie wearing a wraparound reflective visor '
                        'covering the eyes and a smooth panel covering the lower face, hood up'),
    ('partiel_cagoule', 'a person in a black hoodie wearing a black balaclava covering the nose and '
                        'mouth, EYES CLEARLY VISIBLE and well lit, expressive eyes, hood up'),
    ('partiel_rouge',   'a person in a dark hoodie wearing a black face covering over the lower face, '
                        'EYES CLEARLY VISIBLE, red rim lighting from the side, dark room, hood up'),
]

NEG_NOTE = ('Flux schnell ignore le negative prompt (guidance 0) — on cadre par le prompt seul.')

print('chargement Flux.1-schnell...', flush=True)
pipe = FluxPipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()   # 16 Go de VRAM : on garde de la marge

W, H = 1344, 768        # 16:9, cadre hero ; l avatar actuel est en 848x464
SEEDS = [1, 2]

for nom, sujet in VARIANTES:
    for s in SEEDS:
        dst = '%s/%s_s%d.png' % (OUT, nom, s)
        if os.path.exists(dst):
            print('  saute %s' % os.path.basename(dst)); continue
        img = pipe(
            prompt='%s, %s' % (sujet, COMMUN),
            width=W, height=H,
            num_inference_steps=4,
            guidance_scale=0.0,
            generator=torch.Generator('cpu').manual_seed(s),
        ).images[0]
        img.save(dst)
        print('  ecrit %s' % os.path.basename(dst), flush=True)

print('\n%d images dans %s' % (len(os.listdir(OUT)), OUT))
print(NEG_NOTE)
