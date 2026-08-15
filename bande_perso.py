#!/usr/bin/env python3
# bande_perso.py [--images d] [--max 9] — bande de mouvement depuis les photos fond vert.
#
# POURQUOI CES IMAGES CHANGENT TOUT :
#   - fond vert = detourage EXACT (chroma key), au lieu d une estimation par rembg qui
#     grelotte sur les contours ;
#   - 1920x1080 = pas d agrandissement, contrairement aux essais precedents (418x626) ;
#   - plusieurs poses = une banque de gestes des le depart, sans multiplier les graines.
#
# Chaine par image : Wan i2v (2 s) -> chroma key -> clip RGBA. Les clips sont ensuite
# assembles par sync_alpha selon la voix.
#
# Le prompt insiste sur un fond vert UNI ET FIXE : si Wan fait varier la teinte, le
# chroma key laisse des trous. On mesure la proprete du detourage a la fin pour le savoir
# au lieu de le decouvrir a l ecran.
import glob, os, subprocess, sys
import numpy as np, torch, cv2
from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image


def arg(n, d):
    return sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d


DIR = arg('--images', os.path.expanduser('~/avatar_gen/out/perso'))
MAX = int(arg('--max', '9'))
# graine : meme image + graine differente = MEME personnage, GESTE different. C est le
# moyen d etoffer la banque sans nouvelles photos.
SEED = int(arg('--seed', '1'))
SUFFIXE = arg('--suffixe', '')
BASE = os.path.expanduser('~/avatar_gen')
OUT = arg('--out', BASE + '/out/bande')
os.makedirs(OUT, exist_ok=True)
W, H, NF = 1280, 720, 49

# Consigne Boss : viser les mouvements de tete de quelqu un qui PARLE — pas des poses ni
# des gestes de bras. On decrit donc le registre de la narration (hochements, inclinaisons,
# petits tours pour appuyer un propos) et on interdit explicitement les gestes amples dans
# le negatif, sinon Wan part sur du langage corporel de mise en scene.
PROMPT = ('a hooded person wearing a rigid mask with glowing outlines is speaking to the '
          'camera, natural head movements of someone talking: small nods, slight head tilts '
          'and small turns to emphasize words, chin moving gently, shoulders almost still, '
          'arms stay down and still, the mask stays a solid rigid object keeping its exact '
          'shape, static camera, flat uniform green screen background, '
          'the green background stays perfectly still and evenly lit')
NEG = ('raising arms, hand gestures, waving, pointing, crossing arms, big body movement, '
       'leaning, walking, dancing, turning around, '
       'camera movement, zoom, panning, the mask deforming melting or warping, distorted '
       'features, morphing, flickering, changing background color, green spill, shadows on '
       'the background, blurry, low quality')

# Deux des sources montrent les MAINS : leur imposer "bras immobiles" fige une partie
# visible du cadre et ca se voit. On leur donne donc un mouvement propre a ce qu elles
# montrent, et on retire l interdiction de bouger les mains dans leur negatif.
BASE_TETE = ('natural head movements of someone talking: small nods, slight head tilts and '
             'small turns to emphasize words')
SPECIAL = {
    # cle avec tiret bas : le nom est normalise plus bas par replace(' ', '_')
    '2021-07-20_01-15-39': (
        'a hooded person wearing a rigid mask with glowing outlines is speaking to the camera, '
        'hands clasped together in front of the chest, fingers shifting and adjusting slightly, '
        'subtle hand movement while talking, ' + BASE_TETE + ', the mask stays a solid rigid '
        'object keeping its exact shape, static camera, flat uniform green screen background',
        'raising arms above shoulders, waving, pointing, big body movement, leaning, walking, '
        'camera movement, zoom, the mask deforming melting or warping, distorted features, '
        'morphing, flickering, changing background color, green spill, blurry, low quality'),
    'IMG_3611': (
        'a hooded person wearing a rigid mask with glowing outlines sits at a laptop, '
        'hands resting on the keyboard, fingers typing gently, occasional small glance down at '
        'the screen then back to camera, ' + BASE_TETE + ', the mask stays a solid rigid object '
        'keeping its exact shape, static camera, flat uniform green screen background',
        'raising arms, waving, standing up, leaving the frame, big body movement, '
        'camera movement, zoom, the mask deforming melting or warping, distorted features, '
        'morphing, flickering, changing background color, green spill, blurry, low quality'),
}

images = sorted(glob.glob(os.path.join(DIR, '*.[jJ][pP][gG]')) +
                glob.glob(os.path.join(DIR, '*.[jJ][pP][eE][gG]')))[:MAX]
if not images:
    sys.exit('aucune image dans %s' % DIR)
print('%d images' % len(images))

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


for i, src in enumerate(images):
    nom = os.path.splitext(os.path.basename(src))[0].replace(' ', '_')
    brut = '%s/%s%s_brut.mp4' % (OUT, nom, SUFFIXE)
    webm = '%s/%s%s.webm' % (OUT, nom, SUFFIXE)
    if os.path.exists(webm):
        print('  saute %s' % nom); continue
    img = cadre(Image.open(src).convert('RGB'))
    p, ng = SPECIAL.get(nom, (PROMPT, NEG))
    print('  %-30s %s' % (nom, 'mains + tete' if nom in SPECIAL else 'tete seule'), flush=True)
    frames = pipe(image=img, prompt=p, negative_prompt=ng, height=H, width=W,
                  num_frames=NF, guidance_scale=5.0, num_inference_steps=40,
                  generator=torch.Generator('cuda').manual_seed(SEED)).frames[0]
    export_to_video(frames, brut, fps=24)

    # chroma key + despill, puis VP9 alpha. -auto-alt-ref 0 sinon l alpha est perdu
    # silencieusement (piege deja paye).
    #
    # COULEUR MESUREE sur les sources, pas supposee : le fond est du vert PUR (0x00FF01),
    # pas le vert chroma normalise 0x00B140. Avec la mauvaise cible, le detourage effacait
    # TOUT, sujet compris (sujet 0,0 %). Similarite 0,18 verifiee sur image fixe : sujet
    # 21,5 %, 100 % de pixels francs ; au-dela de 0,30 la nettete s effondre.
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', brut, '-vf',
                    'chromakey=0x00FF00:0.18:0.06,despill=type=green:mix=0.5,format=yuva420p',
                    '-c:v', 'libvpx-vp9', '-pix_fmt', 'yuva420p', '-b:v', '3M',
                    '-auto-alt-ref', '0', '-lag-in-frames', '0', webm], check=True)

    # propretE du detourage : on lit l alpha de la 1re image
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-c:v', 'libvpx-vp9', '-i', webm,
                    '-vf', 'alphaextract,format=gray', '-frames:v', '1',
                    '/tmp/a_%s.png' % nom], check=True)
    a = cv2.imread('/tmp/a_%s.png' % nom, cv2.IMREAD_GRAYSCALE)
    net = 100 * ((a > 220) | (a < 35)).mean()      # pixels francs : opaque ou transparent
    sujet = 100 * (a > 128).mean()
    print('  %-30s sujet %4.1f%% | detourage net a %4.1f%%%s'
          % (nom, sujet, net, '   ⚠ SUSPECT' if sujet < 5 or net < 90 else ''), flush=True)
    # on GARDE le brut : si le detourage doit etre rejoue, on ne repaie pas la generation
    # (6 min par clip). C est ce qui a coute le premier clip de la serie precedente.

clips = sorted(glob.glob(OUT + '/*.webm'))
print('\nbande : %d clips, %.1f s de geste distinct' % (len(clips), len(clips) * NF / 24))
print(','.join(clips))
