#!/usr/bin/env python3
# head_turn.py <image> <sortie.mp4> [--dur 8] [--amp 1.0] [--debug]
#
# Anime UNE TETE MASQUEE sans jamais deformer le masque.
#
# POURQUOI PAS UN MODELE GENERATIF : SVD (comme LivePortrait) n a aucune notion de
# rigidite. Il voit un visage et invente du mouvement de CHAIR : la coque ondule, les
# traits glissent. C est structurel, pas un reglage — verdict Boss sur mask2.
#
# CE QU ON FAIT A LA PLACE : un masque rigide qui tourne, physiquement, c est un SOLIDE
# en rotation. On decoupe la tete, on lui applique une homographie (rotation 3D simulee
# autour d un pivot au niveau du cou), on la recolle sur un fond FIXE. Le masque ne peut
# pas se deformer : ses pixels ne sont jamais regeneres, seulement deplaces ensemble.
#
# Boucle parfaite : toutes les frequences sont des multiples entiers de 1/duree.
import math, os, subprocess, sys
import cv2, numpy as np

SRC, DST = sys.argv[1], sys.argv[2]


def arg(n, d):
    return float(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d


DUR, FPS, AMP = arg('--dur', 8.0), int(arg('--fps', 30)), arg('--amp', 1.0)
DEBUG = '--debug' in sys.argv

img = cv2.imread(SRC)
if img is None:
    sys.exit('image illisible')
H, W = img.shape[:2]

# ---------------------------------------------------------------- 1. trouver la tete
# Le masque est la zone CLAIRE de l image (pale sur capuche et fond sombres) : un seuil
# d Otsu sur la luminance l isole sans modele de segmentation.
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binaire = cv2.threshold(cv2.GaussianBlur(gris, (5, 5), 0), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
n, lab, stats, _ = cv2.connectedComponentsWithStats(binaire, 8)
if n < 2:
    sys.exit('aucune zone claire trouvee — cette image ne se prete pas au seuillage')
i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))       # plus grande zone claire
x, y, w, h, aire = stats[i]
print('masque detecte : %dx%d en (%d,%d), %.1f%% de l image' % (w, h, x, y, 100 * aire / (W * H)))

# on etend la boite pour emporter la capuche et le contour du visage
mx, my = int(0.55 * w), int(0.75 * h)
x0, y0 = max(0, x - mx), max(0, y - my)
x1, y1 = min(W, x + w + mx), min(H, y + h + int(0.35 * h))
tete = img[y0:y1, x0:x1].copy()
th, tw = tete.shape[:2]

# masque de fondu : ellipse adoucie -> pas de bord franc a la recomposition
alpha = np.zeros((th, tw), np.float32)
cv2.ellipse(alpha, (tw // 2, th // 2), (int(tw * 0.46), int(th * 0.46)), 0, 0, 360, 1.0, -1)
alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=tw * 0.045, sigmaY=th * 0.045)
alpha = np.clip(alpha, 0, 1)[..., None]

# ---------------------------------------------------------------- 2. fond sans la tete
# La tete va bouger : ce qu elle decouvre doit exister. Le fond est sombre et quasi uni,
# donc un inpaint de Telea suffit largement.
trou = np.zeros((H, W), np.uint8)
cv2.rectangle(trou, (x0, y0), (x1, y1), 255, -1)
fond = cv2.inpaint(img, trou, 7, cv2.INPAINT_TELEA)

if DEBUG:
    cv2.imwrite('/tmp/dbg_tete.png', tete)
    cv2.imwrite('/tmp/dbg_fond.png', fond)
    cv2.imwrite('/tmp/dbg_alpha.png', (alpha[..., 0] * 255).astype(np.uint8))

# ---------------------------------------------------------------- 3. rotation solide
# Homographie = rotation 3D d un plan rigide. Le pivot est SOUS la tete (le cou) : c est
# ce qui rend le mouvement credible plutot qu une tete qui flotte.
N = int(round(DUR * FPS))
pivot_y = th * 1.15                                        # sous le bas de la vignette
src_pts = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
tmp = DST + '.raw.mp4'
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

for k in range(N):
    p = 2 * math.pi * k / N
    yaw = math.radians(2.6 * AMP * (math.sin(2 * p) + 0.4 * math.sin(3 * p + 0.7)))
    pitch = math.radians(1.5 * AMP * math.sin(3 * p + 1.9))
    roll = 1.1 * AMP * math.sin(2 * p + 2.4)
    dy = 0.006 * th * AMP * math.sin(2 * p)                # respiration verticale

    # yaw : les deux cotes du plan s eloignent/rapprochent (perspective)
    ky = math.tan(yaw) * tw * 0.5
    kp = math.tan(pitch) * th * 0.5
    dst_pts = np.float32([
        [0 - ky, 0 - kp * 0.5], [tw - ky, 0 + kp * 0.5],
        [tw + ky, th + kp * 0.5], [0 + ky, th - kp * 0.5]])
    Mh = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # roll + translation autour du pivot-cou, applique APRES la perspective
    Mr = cv2.getRotationMatrix2D((tw / 2, pivot_y), roll, 1.0)
    Mr = np.vstack([Mr, [0, 0, 1]])
    Mr[1, 2] += dy
    M = Mr @ Mh

    tete_w = cv2.warpPerspective(tete, M, (tw, th), flags=cv2.INTER_LANCZOS4,
                                 borderMode=cv2.BORDER_REPLICATE)
    alpha_w = cv2.warpPerspective(alpha[..., 0], M, (tw, th), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]

    frame = fond.copy()
    zone = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (tete_w.astype(np.float32) * alpha_w +
                           zone * (1 - alpha_w)).astype(np.uint8)
    vw.write(frame)
vw.release()

enc = (['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '19', '-b:v', '0']
       if subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                          'color=c=red:s=320x180:r=30', '-t', '1', '-c:v', 'h264_nvenc',
                          '/tmp/nvp2.mp4'], capture_output=True).returncode == 0
       else ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18'])
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-an'] + enc +
               ['-pix_fmt', 'yuv420p', DST], check=True)
os.remove(tmp)

# ---------------------------------------------------------------- 4. mesures
cap = cv2.VideoCapture(DST); ok, prev = cap.read()
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY); d, tete_m, fond_m = [], [], []
while len(d) < 200:
    ok, fr = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g, prev).astype(np.float32)
    tete_m.append(diff[y0:y1, x0:x1].mean())
    m = diff.copy(); m[y0:y1, x0:x1] = 0
    fond_m.append(m.sum() / max(1, (m > 0).sum()))
    d.append(float(diff.mean())); prev = g
cap.release()
a = np.array(d)
print('%s  %.0fs @%dim/s  %dx%d' % (DST, DUR, FPS, W, H))
print('tete %.2f | hors-tete %.3f (doit etre ~0 : le fond ne bouge pas)'
      % (float(np.mean(tete_m)), float(np.mean(fond_m))))
print('variation %.2f | images figees %d' % (a.std() / a.mean() if a.mean() else 0,
                                             int((a < 0.05).sum())))
