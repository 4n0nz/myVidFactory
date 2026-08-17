#!/bin/bash
# test_nouvel_ordre.sh — la chaine dans le BON ordre, sur MS7E5TXNviM.
#
# ORDRE ACTUEL (production)   : b_ -> avatar -> finalize -> re-voix    (--shift 1.5)
# ORDRE TESTE ICI             : b_ -> re-voix -> avatar synchronise -> finalize
#
# Pourquoi : dans l ordre actuel, l avatar est pose AVANT que la voix animateur existe —
# impossible de synchroniser les mouvements de tete dessus. En re-voixant sur le master
# vert, la voix est disponible pour piloter l avatar, et le --shift 1.5 disparait (il ne
# servait qu a compenser la neige que finalize insere avant l image).
#
# Rien n est ecrase : --avatar donne la piste sans toucher a public/avatar.mp4, et av_/
# final_ de Boss sont sauvegardes puis restaures (trap).
set -e
VG=~/videogen; AG=~/avatar_gen; MV=~/mvoice; ID=MS7E5TXNviM; S=$(date +%s)
PY=$VG/.venv/bin/python

restaure() {
  echo "--- restauration ---"
  [ -f $VG/out/av_$ID.orig.$S.mp4 ]    && mv -f $VG/out/av_$ID.orig.$S.mp4 $VG/out/av_$ID.mp4
  [ -f $VG/out/final_$ID.orig.$S.mp4 ] && mv -f $VG/out/final_$ID.orig.$S.mp4 $VG/out/final_$ID.mp4
  ls -l $VG/public/avatar.mp4 | awk '{print "public/avatar.mp4 :", $5, "octets (intact)"}'
}
trap restaure EXIT
cp -f $VG/out/av_$ID.mp4    $VG/out/av_$ID.orig.$S.mp4
cp -f $VG/out/final_$ID.mp4 $VG/out/final_$ID.orig.$S.mp4

# --- 1. la voix, sur le MASTER VERT (pas sur le montage) : aucun decalage a compenser
echo "== 1/4 re-voix sur le master vert (sans shift) =="
SRT=/tmp/${ID}.fr.srt
[ -s "$SRT" ] || { echo "srt absent : $SRT"; exit 1; }
if [ ! -s $AG/out/voix_$ID.mp4 ]; then
  $PY $MV/scripts/style_dub_srt.py "$VG/out/b_$ID.mp4" "$SRT" "$AG/out/voix_$ID.mp4" \
    --work "/tmp/rf_$ID" 2>&1 | tail -3
fi
ffmpeg -y -v error -i $AG/out/voix_$ID.mp4 -vn -ac 1 -ar 16000 $AG/out/voix_$ID.wav
echo "  voix : $(ffprobe -v error -show_entries format=duration -of csv=p=0 $AG/out/voix_$ID.wav) s"

# --- 2. piste avatar synchronisee sur CETTE voix, en temps master (offset 0)
echo "== 2/4 piste avatar synchronisee (segments, teinte verte) =="
CL=$(ls $AG/out/bande_verte/*.webm | tr '\n' ',' | sed 's/,$//')
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $VG/out/b_$ID.mp4)
$AG/.venv-sd/bin/python $AG/piste_avatar.py $AG/out/piste_ordre.mp4 $AG/out/voix_$ID.wav \
  --clips "$CL" --dur $DUR --mode segments --offset 0 \
  --segments $VG/wk_b_$ID/host_map.json 2>&1 | tail -3

# --- 3. avatar + finalize, avec la piste fournie explicitement
echo "== 3/4 pose de l avatar puis montage =="
cd $VG
$PY $VG/vf_avatar.py "$ID" --avatar $AG/out/piste_ordre.mp4 2>&1 | tail -1
bash $VG/vf_finalize.sh "$ID" 2>&1 | tail -1

# --- 4. la voix rejoint l image ; +1,5 s car finalize a insere la neige AVANT l image
echo "== 4/4 montage final =="
ffmpeg -y -v error -i $VG/out/final_$ID.mp4 -itsoffset 1.5 -i $AG/out/voix_$ID.mp4 \
  -map 0:v -map 1:a -c:v copy -c:a aac -b:a 160k -shortest \
  $AG/out/ORDRE_$ID.mp4
echo "===== NOUVEL ORDRE TERMINE $(date +%T) ====="
ls -lh $AG/out/ORDRE_$ID.mp4 | awk '{print $5, $9}'
