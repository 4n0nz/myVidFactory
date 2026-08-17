#!/bin/bash
# bg_anime.sh — fond anime d un tres leger va-et-vient.
# 2 cycles exactement sur 60 s (1800 images) : l etat a la derniere image rejoint celui de
# la premiere, donc la boucle se referme sans raccord visible.
# Ici le mouvement de camera est LEGITIME : c est un decor, pas un personnage.
cd ~/avatar_gen/out
ENC="-c:v h264_nvenc -preset p5 -rc vbr -cq 20 -b:v 0 -pix_fmt yuv420p"
Z='1.05+0.035*sin(2*PI*2*on/1800)'
ffmpeg -y -v error -loop 1 -i bg_vert.png -t 60 -r 30 \
  -vf "scale=2880:1620,zoompan=z='$Z':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080:fps=30" \
  $ENC bg_vert_anime.mp4
for f in bg_vert_fixe.mp4 bg_vert_anime.mp4; do
  printf "%-22s " "$f"
  ffprobe -v error -show_entries format=duration -show_entries stream=width,height -of csv=p=0 "$f" | tr '\n' ' '
  ls -lh "$f" | awk '{print $5}'
done
