#!/bin/bash
# test_pipdecor.sh — decor DERRIERE L AVATAR, fond Matrix inchange.
#
# Correction du malentendu precedent : le decor de Boss va dans la FENETRE de l avatar
# (compose derriere le personnage detoure, par piste_avatar --fond), pas en arriere-plan
# du montage. vf_finalize garde donc son background.mp4 habituel — aucun VF_BACKGROUND ici.
set -e
VG=~/videogen; AG=~/avatar_gen; ID=MS7E5TXNviM; S=$(date +%s)
PY=$VG/.venv/bin/python

restaure() {
  echo "--- restauration ---"
  if [ -f $VG/out/final_$ID.mp4.orig.$S ]; then
    mv -f $VG/out/final_$ID.mp4.orig.$S $VG/out/final_$ID.mp4
    echo "  rendu final_ : $(stat -c %s $VG/out/final_$ID.mp4) octets"
  else
    echo "  ATTENTION : pas de sauvegarde de final_"
  fi
  if [ -f $VG/out/av_$ID.mp4.orig.$S ]; then
    mv -f $VG/out/av_$ID.mp4.orig.$S $VG/out/av_$ID.mp4
    echo "  rendu av_    : $(stat -c %s $VG/out/av_$ID.mp4) octets"
  else
    echo "  ATTENTION : pas de sauvegarde de av_"
  fi
}
trap restaure EXIT

cp -f $VG/out/final_$ID.mp4 $VG/out/final_$ID.mp4.orig.$S
cp -f $VG/out/av_$ID.mp4    $VG/out/av_$ID.mp4.orig.$S
echo "sauvegardes : final_ $(stat -c %s $VG/out/final_$ID.mp4.orig.$S) | av_ $(stat -c %s $VG/out/av_$ID.mp4.orig.$S)"

cd $VG
$PY $VG/vf_avatar.py "$ID" --avatar $AG/out/piste_decor.mp4 2>&1 | tail -1
VF_AVATAR_POPOUT=$AG/out/piste_decor.mp4 bash $VG/vf_finalize.sh "$ID" 2>&1 | tail -1        # background.mp4 par defaut = Matrix

D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $VG/out/final_$ID.mp4)
FN=$(python3 -c "print('%.3f' % ($D - 1.5))")
cat > /tmp/filtre_pd.txt <<FILTRE
[1:a]adelay=1500|1500[voix];
[0:a]volume=0:enable='between(t,1.5,$FN)'[neige];
[voix][neige]amix=inputs=2:normalize=0,alimiter=limit=0.95[out]
FILTRE
ffmpeg -y -v error -i $VG/out/final_$ID.mp4 -i $AG/out/voix_$ID.mp4 \
  -filter_complex_script /tmp/filtre_pd.txt \
  -map 0:v -map "[out]" -c:v copy -c:a aac -b:a 160k -shortest $AG/out/PIPDECOR_$ID.mp4

deb=$(ffmpeg -v info -ss 0 -t 1.4 -i $AG/out/PIPDECOR_$ID.mp4 -af volumedetect -f null - 2>&1 | grep max_volume | grep -oE '[-0-9.]+ dB' | head -1)
fin=$(ffmpeg -v info -sseof -1.4 -i $AG/out/PIPDECOR_$ID.mp4 -af volumedetect -f null - 2>&1 | grep max_volume | grep -oE '[-0-9.]+ dB' | head -1)
echo "  neige : debut $deb | fin $fin"
echo "===== PIPDECOR TERMINE $(date +%T) ====="
ls -lh $AG/out/PIPDECOR_$ID.mp4 | awk '{print $5, $9}'
