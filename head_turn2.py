#!/usr/bin/env python3
# head_turn2.py <image> <sortie.mp4> [--dur 8] [--amp 1.0] [--fps 30]
#
# Anime une tete masquee SANS AUCUNE deformation possible.
#
# CE QUI RATAIT AVANT (verdict Boss : "un drapeau qui bat au vent") : head_turn.py
# utilisait une HOMOGRAPHIE, en deplacant les 4 coins separement pour simuler un yaw.
# Une homographie deforme le quadrilatere — et surtout une tete n est pas un plan : la
# projeter comme une feuille qui pivote donne une carte qui gondole.
#
# ICI on n autorise qu une SIMILITUDE : rotation dans le plan + translation + echelle
# UNIFORME. Ce sont les seules transformations qui preservent les angles et les rapports
# de longueur. La deformation n est pas improbable, elle est impossible : cv2 ne peut
# pas gondoler une image avec une matrice de similitude.
#
# Le prix a payer est honnete : pas de vrai "tour de tete" (yaw), qui demanderait un
# modele 3D. On a une tete qui s incline, se deplace et respire — le registre d un
# narrateur attentif, pas d un presentateur qui se retourne.
import math, os, subprocess, sys
import cv2, numpy as np

SRC, DST = sys.argv[1], sys.argv[2]


def arg(n, d):
    return float(sys.argv[sys.argv.index(n) + 1]) if n in sys.argv else d


DUR, FPS, AMP = arg('--dur', 8.0), int(arg('--fps', 30)), arg('--amp', 1.0)

img = cv2.imread(SRC)
if img is None:
    sys.exit('image illisible')
H, W = img.shape[:2]

# ---------------------------------------------------------------- 1. localiser la tete
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binaire = cv2.threshold(cv2.GaussianBlur(gris, (5, 5), 0), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
n, lab, stats, _ = cv2.connectedComponentsWithStats(binaire, 8)
if n < 2:
    sys.exit('aucune zone claire trouvee')
i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
x, y, w, h, _ = stats[i]
mx, my = int(0.60 * w), int(0.80 * h)
x0, y0 = max(0, x - mx), max(0, y - my)
x1, y1 = min(W, x + w + mx), min(H, y + h + int(0.40 * h))
tete = img[y0:y1, x0:x1].copy()
th, tw = tete.shape[:2]
print('tete : %dx%d en (%d,%d)' % (tw, th, x0, y0))

alpha = np.zeros((th, tw), np.float32)
cv2.ellipse(alpha, (tw // 2, th // 2), (int(tw * 0.44), int(th * 0.44)), 0, 0, 360, 1.0, -1)
alpha = np.clip(cv2.GaussianBlur(alpha, (0, 0), tw * 0.05, th * 0.05), 0, 1)[..., None]

trou = np.zeros((H, W), np.uint8)
cv2.rectangle(trou, (x0, y0), (x1, y1), 255, -1)
fond = cv2.inpaint(img, trou, 7, cv2.INPAINT_TELEA)

# ---------------------------------------------------------------- 2. similitude seule
# pivot au niveau du cou : une tete pivote a sa base, pas en son centre
N = int(round(DUR * FPS))
pivot = (tw / 2.0, th * 1.12)
tmp = DST + '.raw.mp4'
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))

for k in range(N):
    p = 2 * math.pi * k / N                       # boucle exacte : k=N identique a k=0
    roll = 1.9 * AMP * (math.sin(2 * p) + 0.35 * math.sin(3 * p + 0.9))
    tx = 0.011 * tw * AMP * (math.sin(3 * p + 0.4) + 0.45 * math.sin(2 * p))
    ty = 0.008 * th * AMP * (math.sin(2 * p + 1.7) + 0.40 * math.sin(5 * p))
    sc = 1.0 + 0.004 * AMP * math.sin(2 * p + 0.3)          # respiration, echelle uniforme

    M = cv2.getRotationMatrix2D(pivot, roll, sc)  # similitude : angles preserves
    M[0, 2] += tx
    M[1, 2] += ty

    tw_img = cv2.warpAffine(tete, M, (tw, th), flags=cv2.INTER_LANCZOS4,
                            borderMode=cv2.BORDER_REPLICATE)
    aw = cv2.warpAffine(alpha[..., 0], M, (tw, th), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)[..., None]
    frame = fond.copy()
    zone = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (tw_img.astype(np.float32) * aw + zone * (1 - aw)).astype(np.uint8)
    vw.write(frame)
vw.release()

enc = (['-c:v', 'h264_nvenc', '-preset', 'p5', '-rc', 'vbr', '-cq', '19', '-b:v', '0']
       if subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                          'color=c=red:s=320x180:r=30', '-t', '1', '-c:v', 'h264_nvenc',
                          '/tmp/nvp3.mp4'], capture_output=True).returncode == 0
       else ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18'])
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', tmp, '-an'] + enc +
               ['-pix_fmt', 'yuv420p', DST], check=True)
os.remove(tmp)

# ---------------------------------------------------------------- 3. preuve de rigidite
# On verifie que la tete a bien subi une similitude : on suit des points par flot optique
# et on mesure l ecart RESIDUEL apres avoir ajuste la meilleure similitude. Un residu
# proche de 0 = aucune deformation. C est la mesure qui manquait a la version "drapeau".
cap = cv2.VideoCapture(DST)
ok, f0 = cap.read()
g0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
roi = np.zeros_like(g0); roi[y0:y1, x0:x1] = 255
pts = cv2.goodFeaturesToTrack(g0, maxCorners=180, qualityLevel=0.01, minDistance=12, mask=roi)
res, dep, k = [], [], 0
prev_g, prev_p = g0, pts
while pts is not None and k < 120:
    ok, fr = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_g, g, prev_p, None)
    if p1 is None:
        break
    a = prev_p[st == 1].reshape(-1, 2); b = p1[st == 1].reshape(-1, 2)
    if len(a) >= 12:
        M2, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC)  # similitude seule
        if M2 is not None:
            proj = (M2[:, :2] @ a.T).T + M2[:, 2]
            res.append(float(np.median(np.linalg.norm(proj - b, axis=1))))
            dep.append(float(np.median(np.linalg.norm(b - a, axis=1))))
    prev_g, prev_p = g, p1
    k += 1
cap.release()
print('%s  %.0fs @%dim/s  %dx%d' % (DST, DUR, FPS, W, H))
if res:
    print('deplacement median %.2f px/image | residu hors-similitude %.3f px'
          % (float(np.mean(dep)), float(np.mean(res))))
    print('(residu ~0 = mouvement strictement rigide, aucun gondolement)')
