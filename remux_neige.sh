#!/bin/bash
# remux_neige.sh — remet le son de la neige dans les montages de test.
#
# Le montage prenait l image de final_ et SEULEMENT la piste voix. Or le burst de neige
# (1,5 s au debut et a la fin, pose par vf_finalize) vit dans l AUDIO de final_ : en ne
# gardant que la voix, on le perd. Meme bug que celui deja corrige dans style_dub_srt
# (commit 2f12636), reintroduit ici par le mux fait a la main.
#
# On reinjecte donc la piste de final_ MUETTE partout sauf dans les deux fenetres de neige
# — le narrateur d origine ne doit jamais reapparaitre au milieu.
# La voix, elle, est decalee de 1,5 s : elle est calee sur le master vert, l image de final_
# commence par la neige.
set -e
VG=~/videogen; AG=~/avatar_gen; ID=MS7E5TXNviM
FIN=$VG/out/final_$ID.mp4
VOIX=$AG/out/voix_$ID.mp4
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$FIN")
FIN_NEIGE=$(python3 -c "print('%.3f' % ($D - 1.5))")
echo "final_ = ${D}s | fenetres de neige : 0-1.5 et ${FIN_NEIGE}-${D}"

for CIBLE in ORDRE NUIT; do
  SRC=$AG/out/${CIBLE}_$ID.mp4
  [ -f "$SRC" ] || { echo "  $CIBLE : absent, saute"; continue; }
  OUT=$AG/out/${CIBLE}_neige_$ID.mp4
  ffmpeg -y -v error -i "$FIN" -i "$VOIX" -filter_complex \
    "[1:a]adelay=1500|1500[voix];
     [0:a]volume=0:enable='between(t,1.5,$FIN_NEIGE)'[neige];
     [voix][neige]amix=inputs=2:normalize=0,alimiter=limit=0.95[out]" \
    -map 0:v -map "[out]" -c:v copy -c:a aac -b:a 160k -shortest "$OUT"

  # verification : du son au debut ET a la fin, la voix intacte au milieu
  deb=$(ffmpeg -v info -ss 0 -t 1.4 -i "$OUT" -af volumedetect -f null - 2>&1 | grep max_volume | grep -oE '[-0-9.]+ dB' | head -1)
  fin=$(ffmpeg -v info -sseof -1.4 -i "$OUT" -af volumedetect -f null - 2>&1 | grep max_volume | grep -oE '[-0-9.]+ dB' | head -1)
  mil=$(ffmpeg -v info -ss 300 -t 2 -i "$OUT" -af volumedetect -f null - 2>&1 | grep max_volume | grep -oE '[-0-9.]+ dB' | head -1)
  echo "  $CIBLE : debut $deb | fin $fin | milieu $mil"
done
echo "===== REMUX NEIGE OK $(date +%T) ====="
ls -lh $AG/out/*_neige_$ID.mp4 | awk '{print $5, $9}'
