#!/bin/bash
# smooth.sh — corrige le sacade des clips SVD (25 images a 7 im/s) en montant a 30 im/s.
#
# Trois sorties pour que la comparaison soit honnete :
#   _dup    : simple changement de cadence (images dupliquees) = ce qui sacade, la reference
#   _interp : minterpolate avec compensation de mouvement = le correctif, sans rien installer
#   _loop   : _interp en aller-retour (ping-pong) pour doubler la duree sans raccord visible
#             (les segments hero font 6,4 s en mediane, un clip de 3,6 s est trop court)
B=$HOME/avatar_gen
IN=$B/out/i2v
OUT=$B/out/smooth
mkdir -p "$OUT"

enc() { echo "-c:v h264_nvenc -preset p5 -rc vbr -cq 19 -b:v 0 -pix_fmt yuv420p"; }

for f in "$IN"/mask_m160.mp4 "$IN"/mask_m90.mp4; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .mp4)

  ffmpeg -y -v error -i "$f" -vf "fps=30" -an $(enc) "$OUT/${n}_dup.mp4"

  # mci + aobmc + vsbmc = le reglage le plus fin de minterpolate (CPU, mais 3,6 s a traiter)
  ffmpeg -y -v error -i "$f" \
    -vf "minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1" \
    -an $(enc) "$OUT/${n}_interp.mp4"

  # ping-pong : la sequence puis son miroir. Le raccord tombe sur une image identique,
  # donc il ne se voit pas ; ca ne cree aucun mouvement impossible (juste inverse).
  ffmpeg -y -v error -i "$OUT/${n}_interp.mp4" \
    -filter_complex "[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1[out]" \
    -map "[out]" -an $(enc) "$OUT/${n}_loop.mp4"
  echo "traite $n"
done

echo "--- mesure de fluidite ---"
# Une video fluide a des ecarts inter-images REGULIERS. Une video sacadee alterne grands
# ecarts (vraie image) et zeros (image dupliquee) : c est le coefficient de variation qui
# le revele, pas l amplitude moyenne.
$B/.venv-sd/bin/python - <<'EOF'
import cv2, glob, numpy as np
fichiers = sorted(glob.glob('/home/boss/avatar_gen/out/smooth/*.mp4')) + \
           sorted(glob.glob('/home/boss/avatar_gen/out/i2v/mask_m*.mp4'))
for p in fichiers:
    cap = cv2.VideoCapture(p); ok, prev = cap.read()
    if not ok: continue
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY); d = []
    while len(d) < 90:
        ok, fr = cap.read()
        if not ok: break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        d.append(float(cv2.absdiff(g, prev).mean())); prev = g
    cap.release()
    if not d: continue
    a = np.array(d); cv_ = a.std() / a.mean() if a.mean() > 0 else 0
    zeros = int((a < 0.05).sum())
    print('%-30s ecart moyen %5.2f | variation %4.2f | images figees %2d  (bas = fluide)'
          % (p.split('/')[-1], a.mean(), cv_, zeros))
EOF
echo "===== SMOOTH_TERMINE $(date '+%T') ====="
ls -lh "$OUT"
