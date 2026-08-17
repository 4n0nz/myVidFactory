#!/bin/bash
# rendu_final.sh — la chaine complete, corrigee de bout en bout.
#
# Les 4 defauts releves par Boss sur la version precedente :
#  1. Matrix rain absente  -> on ne touche PLUS a VF_BACKGROUND (defaut = background.mp4)
#  2. vieil avatar dans le pip -> VF_AVATAR_POPOUT, l avatar du pop-out etait en dur
#  3. pipintro sans son    -> plus de fondu audio a la jointure : il tombait pile sur le
#                             burst de neige (1,5 s a -16 dB), qu il rendait inaudible
#  4. intro coupee au milieu -> 18,4 s et non 12 : le generique se termine par un silence
#                             a 17,3 s. intro.mp4 contient d autres sequences apres, separees
#                             par du noir — ce n est pas un generique de 4 min.
set -e
VG=~/videogen; AG=~/avatar_gen; ID=MS7E5TXNviM; S=$(date +%s)
PY=$VG/.venv/bin/python

restaure() {
  echo "--- restauration ---"
  for f in final_$ID av_$ID; do
    if [ -f $VG/out/$f.mp4.orig.$S ]; then
      mv -f $VG/out/$f.mp4.orig.$S $VG/out/$f.mp4
      echo "  rendu $f : $(stat -c %s $VG/out/$f.mp4) octets"
    else
      echo "  ATTENTION : pas de sauvegarde pour $f"
    fi
  done
}
trap restaure EXIT
cp -f $VG/out/final_$ID.mp4 $VG/out/final_$ID.mp4.orig.$S
cp -f $VG/out/av_$ID.mp4    $VG/out/av_$ID.mp4.orig.$S
echo "sauvegardes : final_ $(stat -c %s $VG/out/final_$ID.mp4.orig.$S) | av_ $(stat -c %s $VG/out/av_$ID.mp4.orig.$S)"

cd $VG
echo "== 1/3 avatar dans le vert + pop-out (meme piste) =="
$PY $VG/vf_avatar.py "$ID" --avatar $AG/out/piste_decor.mp4 2>&1 | tail -1
VF_AVATAR_POPOUT=$AG/out/piste_decor.mp4 bash $VG/vf_finalize.sh "$ID" 2>&1 | tail -1

echo "== 2/3 voix + neige preservee =="
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $VG/out/final_$ID.mp4)
FN=$(python3 -c "print('%.3f' % ($D - 1.5))")
cat > /tmp/filtre_rf.txt <<FILTRE
[1:a]adelay=1500|1500[voix];
[0:a]volume=0:enable='between(t,1.5,$FN)'[neige];
[voix][neige]amix=inputs=2:normalize=0,alimiter=limit=0.95[out]
FILTRE
ffmpeg -y -v error -i $VG/out/final_$ID.mp4 -i $AG/out/voix_$ID.mp4 \
  -filter_complex_script /tmp/filtre_rf.txt \
  -map 0:v -map "[out]" -c:v copy -c:a aac -b:a 160k -shortest $AG/out/corps_$ID.mp4
deb=$(ffmpeg -v info -ss 0 -t 1.4 -i $AG/out/corps_$ID.mp4 -af volumedetect -f null - 2>&1 | grep max_volume | grep -oE '[-0-9.]+ dB' | head -1)
echo "  neige au debut du corps : $deb"

echo "== 3/3 generique debut + fin (sans fondu : il tuait la neige) =="
$PY $VG/vf_intro_outro.py $AG/out/corps_$ID.mp4 $AG/out/FINAL_$ID.mp4 2>&1 | tail -6
# la sonde se cale sur la DUREE REELLE de l intro : en dur a 18,5 s elle tombait dans le
# generique des qu on a pris l intro complete (241 s), et affichait le volume de la musique
IDUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $VG/assets/intro.mp4)
SONDE=$(python3 -c "print('%.2f' % ($IDUR + 0.1))")
deb2=$(ffmpeg -v info -ss $SONDE -t 1.4 -i $AG/out/FINAL_$ID.mp4 -af volumedetect -f null - 2>&1 | grep max_volume | grep -oE '[-0-9.]+ dB' | head -1)
echo "  neige juste apres l intro (t=${SONDE}s) : $deb2"
echo "===== RENDU FINAL TERMINE $(date +%T) ====="
ls -lh $AG/out/FINAL_$ID.mp4 | awk '{print $5, $9}'
