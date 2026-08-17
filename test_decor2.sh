#!/bin/bash
# test_decor2.sh — montage complet avec le decor choisi par Boss.
#
# Deux corrections par rapport a la version precedente :
#  1. le filtre audio passe par -filter_complex_script (un FICHIER) : les apostrophes du
#     enable='between(...)' ne survivent pas au heredoc SSH et cassaient le montage ;
#  2. les sauvegardes sont VERIFIEES (taille) avant et apres — la restauration precedente
#     n a remis qu un des deux fichiers sans le signaler.
set -e
VG=~/videogen; AG=~/avatar_gen; ID=MS7E5TXNviM; S=$(date +%s)
PY=$VG/.venv/bin/python
BGV=$AG/out/bg_gen/poste_3ecrans_s2_60s.mp4

sauve() { cp -f "$1" "$1.orig.$S"; }
rends() {
  [ -f "$1.orig.$S" ] || { echo "  ATTENTION : pas de sauvegarde pour $(basename $1)"; return; }
  mv -f "$1.orig.$S" "$1"
  echo "  rendu $(basename $1) : $(stat -c %s "$1") octets"
}
restaure() { echo "--- restauration ---"; rends $VG/out/final_$ID.mp4; }
trap restaure EXIT

sauve $VG/out/final_$ID.mp4
echo "sauvegarde final_ : $(stat -c %s $VG/out/final_$ID.mp4.orig.$S) octets"

cd $VG
$PY $VG/vf_avatar.py "$ID" --avatar $AG/out/piste_nuit.mp4 2>&1 | tail -1
VF_BACKGROUND=$BGV bash $VG/vf_finalize.sh "$ID" 2>&1 | tail -1

D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $VG/out/final_$ID.mp4)
FN=$(python3 -c "print('%.3f' % ($D - 1.5))")
# filtre dans un fichier : aucun quoting a traverser
cat > /tmp/filtre_neige.txt <<FILTRE
[1:a]adelay=1500|1500[voix];
[0:a]volume=0:enable='between(t,1.5,$FN)'[neige];
[voix][neige]amix=inputs=2:normalize=0,alimiter=limit=0.95[out]
FILTRE
ffmpeg -y -v error -i $VG/out/final_$ID.mp4 -i $AG/out/voix_$ID.mp4 \
  -filter_complex_script /tmp/filtre_neige.txt \
  -map 0:v -map "[out]" -c:v copy -c:a aac -b:a 160k -shortest $AG/out/DECOR_$ID.mp4

deb=$(ffmpeg -v info -ss 0 -t 1.4 -i $AG/out/DECOR_$ID.mp4 -af volumedetect -f null - 2>&1 | grep max_volume | grep -oE '[-0-9.]+ dB' | head -1)
fin=$(ffmpeg -v info -sseof -1.4 -i $AG/out/DECOR_$ID.mp4 -af volumedetect -f null - 2>&1 | grep max_volume | grep -oE '[-0-9.]+ dB' | head -1)
echo "  neige : debut $deb | fin $fin"
echo "===== DECOR TERMINE $(date +%T) ====="
ls -lh $AG/out/DECOR_$ID.mp4 | awk '{print $5, $9}'
