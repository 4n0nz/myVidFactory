#!/bin/bash
# teinte_verte.sh — bascule les contours lumineux du masque du BLEU vers le VERT.
#
# Rotation MESUREE, pas devinee : la teinte mediane des pixels vifs est a 210 degres
# (bleu-cyan), le vert pur est a 120 -> rotation de -90 degres. On tourne la teinte plutot
# que de plaquer une dominante verte sur toute l image : le personnage est noir, seuls les
# contours LED portent de la couleur, et une teinte tournee garde leur luminosite intacte.
#
# L alpha doit survivre : VP9 exige -auto-alt-ref 0, sinon il est perdu en silence.
set -e
B=~/avatar_gen/out/bande_parle
V=~/avatar_gen/out/bande_verte
mkdir -p $V

# on attend que la generation en cours libere la banque
while pgrep -f "[b]ande_perso.py" >/dev/null 2>&1; do sleep 20; done
echo "banque complete : $(ls $B/*.webm | grep -v NARRATEUR | wc -l) clips"

for f in $B/*.webm; do
  case "$(basename $f)" in NARRATEUR*) continue;; esac
  n=$(basename "$f")
  ffmpeg -y -v error -c:v libvpx-vp9 -i "$f" \
    -vf "hue=h=-90,format=yuva420p" \
    -c:v libvpx-vp9 -pix_fmt yuva420p -b:v 3M \
    -auto-alt-ref 0 -lag-in-frames 0 "$V/$n"

  # verification : teinte obtenue + alpha toujours la
  ffmpeg -y -v error -c:v libvpx-vp9 -i "$V/$n" -frames:v 1 /tmp/tv.png
  A=$(ffmpeg -v error -c:v libvpx-vp9 -i "$V/$n" -vf alphaextract -frames:v 1 -f null - 2>&1 && echo OUI || echo NON)
  ~/avatar_gen/.venv-sd/bin/python - "$n" "$A" <<'EOF'
import sys, cv2, numpy as np
im = cv2.imread('/tmp/tv.png')
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
h, s, v = hsv[:, :, 0].astype(int), hsv[:, :, 1].astype(int), hsv[:, :, 2].astype(int)
m = (s > 90) & (v > 90)
med = int(np.median(h[m])) * 2 if m.sum() else -1
print('  %-34s teinte %3d deg (vert = 120) | alpha %s' % (sys.argv[1], med, sys.argv[2]))
EOF
done
echo "===== TEINTE VERTE OK $(date +%T) ====="
ls $V/*.webm | wc -l
