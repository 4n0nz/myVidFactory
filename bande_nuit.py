#!/usr/bin/env python3
# bande_nuit.py [--maillons 6] [--graines 100,200,300,400] — banque de gestes LONGS.
#
# Constat Boss : avec des clips de 2 s, les scenes se repetent. On genere donc, pour chaque
# photo et chaque graine, une CHAINE longue (la derniere image d un maillon devient l entree
# du suivant : le mouvement continue vraiment, au lieu d un aller-retour qui se voit).
#
# 3 photos x 4 graines x 6 maillons = 12 clips de ~12 s = ~144 s de geste distinct,
# contre 18 s aujourd hui. Environ 7 h de calcul : c est un run de nuit.
#
# REPRISE : chaque clip fini est saute au redemarrage. Un run interrompu se relance avec la
# meme commande sans rien reperdre — indispensable sur 7 h.
#
# Tout est fait ici, plus d etape manuelle : chainage -> chroma key -> teinte verte -> alpha.
import glob, os, subprocess, sys
import numpy as np, torch, cv2
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image


def arg(n, d):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


MAILLONS = int(arg('--maillons', '6'))
GRAINES = [int(g) for g in arg('--graines', '100,200,300,400').split(',')]
BASE = os.path.expanduser('~/avatar_gen')
DIR = arg('--images', BASE + '/out/perso')
OUT = arg('--out', BASE + '/out/bande_nuit')
os.makedirs(OUT, exist_ok=True)
W, H, NF = 1280, 720, 49

BASE_TETE = ('natural head movements of someone talking: small nods, slight head tilts and '
             'small turns to emphasize words')
PROMPT = ('a hooded person wearing a rigid mask with glowing outlines is speaking to the '
          'camera, ' + BASE_TETE + ', chin moving gently, shoulders almost still, arms stay '
          'down and still, the mask stays a solid rigid object keeping its exact shape, '
          'static camera, flat uniform green screen background, the green background stays '
          'perfectly still and evenly lit')
NEG = ('raising arms, hand gestures, waving, pointing, crossing arms, big body movement, '
       'leaning, walking, dancing, turning around, camera movement, zoom, panning, '
       'the mask deforming melting or warping, distorted features, morphing, flickering, '
       'changing background color, green spill, shadows on the background, blurry, low quality')
SPECIAL = {
    '2021-07-20_01-15-39': (
        'a hooded person wearing a rigid mask with glowing outlines is speaking to the camera, '
        'hands clasped together in front of the chest, fingers shifting and adjusting slightly, '
        'subtle hand movement while talking, ' + BASE_TETE + ', the mask stays a solid rigid '
        'object keeping its exact shape, static camera, flat uniform green screen background',
        'raising arms above shoulders, waving, pointing, big body movement, leaning, walking, '
        'camera movement, zoom, the mask deforming melting or warping, distorted features, '
        'morphing, flickering, changing background color, green spill, blurry, low quality'),
}

images = sorted(glob.glob(os.path.join(DIR, '*.[jJ][pP][gG]')) +
                glob.glob(os.path.join(DIR, '*.[jJ][pP][eE][gG]')))
if not images:
    sys.exit('aucune image dans %s' % DIR)
total = len(images) * len(GRAINES)
print('%d photos x %d graines = %d clips de %d maillons (~%.0f s chacun)'
      % (len(images), len(GRAINES), total, MAILLONS, MAILLONS * NF / 24), flush=True)

print('chargement Wan 2.2...', flush=True)
vae = AutoencoderKLWan.from_pretrained(BASE + '/models/wan22', subfolder='vae',
                                       torch_dtype=torch.float32)
pipe = WanImageToVideoPipeline.from_pretrained(BASE + '/models/wan22', vae=vae,
                                               torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()


def cadre(im):
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


fait = 0
for src in images:
    nom = os.path.splitext(os.path.basename(src))[0].replace(' ', '_')
    p, ng = SPECIAL.get(nom, (PROMPT, NEG))
    for g in GRAINES:
        fait += 1
        webm = '%s/%s_g%d.webm' % (OUT, nom, g)
        if os.path.exists(webm):
            print('[%d/%d] saute %s' % (fait, total, os.path.basename(webm)), flush=True)
            continue
        print('[%d/%d] %s graine %d ...' % (fait, total, nom, g), flush=True)
        img = cadre(Image.open(src).convert('RGB'))
        ref = np.asarray(img.convert('L'), dtype=np.float32).mean()
        morceaux = []
        for m in range(MAILLONS):
            fr = pipe(image=img, prompt=p, negative_prompt=ng, height=H, width=W,
                      num_frames=NF, guidance_scale=5.0, num_inference_steps=40,
                      generator=torch.Generator('cuda').manual_seed(g + m)).frames[0]
            morceaux.extend(fr if m == 0 else fr[1:])
            img = cadre(fr[-1].convert('RGB') if isinstance(fr[-1], Image.Image)
                        else Image.fromarray((np.asarray(fr[-1]) * 255).astype(np.uint8)))
        derive = np.asarray(img.convert('L'), dtype=np.float32).mean() - ref

        brut = '%s/%s_g%d_brut.mp4' % (OUT, nom, g)
        export_to_video(morceaux, brut, fps=24)
        # chroma key (vert PUR mesure) + despill, puis rotation de teinte -90 deg
        # (contours a 210 deg -> 120 = vert). -auto-alt-ref 0 sinon l alpha est perdu.
        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', brut, '-vf',
                        'chromakey=0x00FF00:0.18:0.06,despill=type=green:mix=0.5,'
                        'hue=h=-90,format=yuva420p',
                        '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', '-b:v', '3M',
                        '-auto-alt-ref', '0', '-lag-in-frames', '0', webm], check=True)
        os.remove(brut)

        subprocess.run(['ffmpeg', '-y', '-v', 'error', '-c:v', 'libvpx-vp9', '-i', webm,
                        '-frames:v', '1', '/tmp/n.png'], check=True)
        im2 = cv2.imread('/tmp/n.png')
        hsv = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
        msk = (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 90)
        teinte = int(np.median(hsv[:, :, 0][msk])) * 2 if msk.sum() > 50 else -1
        alpha_ok = subprocess.run(['ffmpeg', '-v', 'error', '-c:v', 'libvpx-vp9', '-i', webm,
                                   '-vf', 'alphaextract', '-frames:v', '1', '-f', 'null', '-'],
                                  capture_output=True).returncode == 0
        print('        %d images | teinte %d deg | alpha %s | derive luminance %+.1f'
              % (len(morceaux), teinte, 'OK' if alpha_ok else 'PERDU', derive), flush=True)

clips = sorted(glob.glob(OUT + '/*.webm'))
print('\nBANQUE NUIT : %d clips, %.0f s de geste distinct'
      % (len(clips), sum(1 for _ in clips) * MAILLONS * NF / 24))
