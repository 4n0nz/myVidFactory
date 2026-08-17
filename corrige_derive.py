#!/usr/bin/env python3
# corrige_derive.py <dossier_clips> <dossier_sortie>
#
# Compense la derive du CHAINAGE a l interieur de chaque clip.
#
# Chaque clip de 12 s est fait de 6 maillons de 2 s : l image finale d un maillon sert
# d entree au suivant, donc elle repasse par l encodeur a chaque fois. Mesure sur un clip :
# saturation 119 -> 60 (moitie perdue), luminance 6,3 -> 4,6, nettete 64 -> 41. Le masque
# se delave, puis redevient net quand la lecture repart : c est la "coupure" vue toutes
# les 12 s.
#
# On ramene chaque image aux valeurs de la PREMIERE (saturation et luminance), en ne
# mesurant que sur le SUJET (alpha > 128) — le fond transparent fausserait la moyenne.
# Le gain est borne : au-dela, on amplifierait le bruit du VAE au lieu de restaurer.
import os, subprocess, sys
import cv2, numpy as np

SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)
GAIN_MAX = 3.2

for nom in sorted(os.listdir(SRC)):
    if not nom.endswith('.webm'):
        continue
    out = os.path.join(DST, nom)
    if os.path.exists(out):
        print('  saute %s' % nom); continue
    tmp = '/tmp/cd_%d' % os.getpid()
    subprocess.run(['rm', '-rf', tmp]); os.makedirs(tmp)
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-c:v', 'libvpx-vp9',
                    '-i', os.path.join(SRC, nom), '-vf', 'format=rgba',
                    '%s/f%%04d.png' % tmp], check=True)
    fichiers = sorted(os.listdir(tmp))

    ref_s = ref_v = None
    avant = apres = None
    for i, f in enumerate(fichiers):
        p = os.path.join(tmp, f)
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        alpha = im[:, :, 3]
        m = alpha > 128
        if m.sum() < 100:
            continue
        hsv = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
        s_moy = hsv[:, :, 1][m].mean()
        v_moy = hsv[:, :, 2][m].mean()
        if ref_s is None:                      # la 1re image donne la cible
            ref_s, ref_v = s_moy, v_moy
            avant = (s_moy, v_moy)
        else:
            gs = min(GAIN_MAX, ref_s / max(1.0, s_moy))
            gv = min(GAIN_MAX, ref_v / max(1.0, v_moy))
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * gs, 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * gv, 0, 255)
            im[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            cv2.imwrite(p, im)
            apres = (hsv[:, :, 1][m].mean(), hsv[:, :, 2][m].mean())

    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-framerate', '24',
                    '-i', '%s/f%%04d.png' % tmp, '-c:v', 'libvpx-vp9',
                    '-pix_fmt', 'yuva420p', '-b:v', '3M',
                    '-auto-alt-ref', '0', '-lag-in-frames', '0', out], check=True)
    subprocess.run(['rm', '-rf', tmp])
    print('  %-34s saturation %.0f -> %.0f en fin de clip (cible %.0f)'
          % (nom, avant[0] if avant else 0, apres[0] if apres else 0, ref_s or 0), flush=True)

print('\n%d clips corriges dans %s' % (len([f for f in os.listdir(DST) if f.endswith('.webm')]), DST))
