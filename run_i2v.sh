#!/bin/bash
# run_i2v.sh — installe SVD, genere de VRAIES animations, et met cote a cote avec
# l ancienne version "mouvement de camera" pour que la difference soit visible d un coup.
set -e
BASE=$HOME/avatar_gen
PYSD=$BASE/.venv-sd/bin/python
LIV=$BASE/out/livraison_i2v
rm -rf "$LIV"; mkdir -p "$LIV"

echo "===== 1/4 telechargement SVD (~9,5 Go, aucun encodeur de texte) ====="
$PYSD - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download('stabilityai/stable-video-diffusion-img2vid-xt',
    local_dir='/home/boss/avatar_gen/models/svd',
    allow_patterns=['**/*.fp16.safetensors', '**/*.json', '**/*.txt', '*.json'],
    max_workers=8)
print('ok', p)
EOF
du -sh $BASE/models/svd

echo "===== 2/4 generation image-to-video ====="
$PYSD $BASE/gen_i2v.py 2>&1 | grep -v "^\[transformers\]" | tail -12

echo "===== 3/4 mesure : est-ce que le SUJET bouge, ou juste le cadre ? ====="
# On compare le mouvement au CENTRE (le personnage) et sur les BORDS (le fond).
# anim_rigid deplace tout : bords et centre bougent autant. Une vraie animation fait
# bouger le centre BEAUCOUP plus que les bords.
for f in $BASE/out/i2v/*.mp4 $BASE/out/livraison_anon4/anon4_porcelaine_s1_anim.mp4; do
  [ -f "$f" ] || continue
  $PYSD - "$f" <<'EOF'
import sys, cv2, numpy as np
cap = cv2.VideoCapture(sys.argv[1]); ok, prev = cap.read()
if not ok: sys.exit()
prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY); H, W = prev.shape
c, b, n = [], [], 0
while True:
    ok, fr = cap.read()
    if not ok or n > 40: break
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    d = cv2.absdiff(g, prev).astype(np.float32)
    c.append(d[H//4:3*H//4, W//3:2*W//3].mean())          # centre = le personnage
    b.append(np.concatenate([d[:, :W//8].ravel(), d[:, -W//8:].ravel()]).mean())  # bords = fond
    prev = g; n += 1
cap.release()
cm, bm = float(np.mean(c)), float(np.mean(b))
print('%-52s centre %5.2f | bords %5.2f | ratio %4.1f' %
      (sys.argv[1].split('/')[-1], cm, bm, cm/bm if bm > 0.01 else 99))
EOF
done

echo "===== 4/4 livraison ====="
cp $BASE/out/i2v/*.mp4 "$LIV/" 2>/dev/null
cp $BASE/out/livraison_anon4/anon4_porcelaine_s1_anim.mp4 "$LIV/ANCIEN_mouvement_de_camera.mp4" 2>/dev/null
cp $BASE/out/anon4/anon4_porcelaine_s1.png "$LIV/" 2>/dev/null
echo "===== TERMINE $(date '+%F %T') ====="
ls -lh "$LIV"
