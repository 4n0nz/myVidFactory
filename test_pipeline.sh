#!/bin/bash
# test_pipeline.sh — passe les 2 pistes avatar dans le VRAI pipeline et compare.
#
# vf_finalize.sh travaille sur des noms fixes (av_<id>.mp4 -> final_<id>.mp4). On doit donc
# occuper temporairement ces noms. TOUT est sauvegarde et restaure, y compris si le script
# echoue en route (trap) : public/avatar.mp4, av_<id>.mp4 et final_<id>.mp4 de Boss doivent
# se retrouver intacts.
set -e
VG=~/videogen
AG=~/avatar_gen
ID=MS7E5TXNviM
STAMP=$(date +%s)

restaure() {
  echo "--- restauration ---"
  [ -f $VG/public/avatar.orig.$STAMP.mp4 ] && mv -f $VG/public/avatar.orig.$STAMP.mp4 $VG/public/avatar.mp4
  [ -f $VG/out/av_$ID.orig.$STAMP.mp4 ]    && mv -f $VG/out/av_$ID.orig.$STAMP.mp4 $VG/out/av_$ID.mp4
  [ -f $VG/out/final_$ID.orig.$STAMP.mp4 ] && mv -f $VG/out/final_$ID.orig.$STAMP.mp4 $VG/out/final_$ID.mp4
  ls -l $VG/public/avatar.mp4 $VG/out/av_$ID.mp4 $VG/out/final_$ID.mp4 | awk '{print $5, $9}'
}
trap restaure EXIT

cp -f $VG/public/avatar.mp4 $VG/public/avatar.orig.$STAMP.mp4
cp -f $VG/out/av_$ID.mp4    $VG/out/av_$ID.orig.$STAMP.mp4
cp -f $VG/out/final_$ID.mp4 $VG/out/final_$ID.orig.$STAMP.mp4
echo "sauvegardes faites"

for MODE in banque boucle; do
  echo "===== $MODE ====="
  cp -f $AG/out/piste_$MODE.mp4 $VG/public/avatar.mp4
  cd $VG
  $VG/.venv/bin/python vf_avatar.py $ID 2>&1 | tail -2
  bash vf_finalize.sh $ID 2>&1 | tail -2
  # la voix animateur remplace l audio d origine (le pipeline la pose plus tard d habitude)
  ffmpeg -y -v error -i $VG/out/final_$ID.mp4 -i $VG/out/ANIM_$ID.mp4 \
    -map 0:v -map 1:a -c:v copy -c:a aac -b:a 160k -shortest \
    $AG/out/TEST_${MODE}_$ID.mp4
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $AG/out/TEST_${MODE}_$ID.mp4)
  echo "  -> TEST_${MODE}_$ID.mp4 : ${d}s"
done

echo "===== TEST PIPELINE TERMINE $(date +%T) ====="
ls -lh $AG/out/TEST_*.mp4 | awk '{print $5, $9}'
