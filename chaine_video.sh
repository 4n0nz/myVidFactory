#!/bin/bash
# chaine_video.sh <ID> [espacement] — de b_<ID>.mp4 au livrable, tout compris.
#
# Reprend le master vert existant : la detection et le QC (des heures) ne sont pas refaits.
# Ordre corrige : la VOIX est produite AVANT l avatar, donc la piste se synchronise sur la
# vraie voix animateur et le --shift 1.5 n a plus lieu d etre.
#
# 1. voix animateur sur le master vert (sous-titres FR deja presents)
# 2. coupures du montage (detection de scenes) — c est la QUE l avatar changera de pose
# 3. piste avatar : decor derriere le personnage, mains reservees au pip
# 4. avatar dans le vert + pop-out (la MEME piste aux deux endroits)
# 5. voix + neige preservee
# 6. generique debut/fin
set -e
ID=${1:?usage: chaine_video.sh <ID> [espacement]}
ESP=${2:-45}
VG=~/videogen; AG=~/avatar_gen; MV=~/mvoice; S=$(date +%s)
PY=$VG/.venv/bin/python
SRT=/tmp/${ID}.fr.srt
WD=$VG/wk_b_$ID

for f in $VG/out/b_$ID.mp4 $SRT $WD/host_map.json; do
  [ -s "$f" ] || { echo "manquant : $f"; exit 1; }
done

restaure() {
  echo "--- restauration ---"
  for n in final_$ID av_$ID; do
    [ -f $VG/out/$n.mp4.orig.$S ] && { mv -f $VG/out/$n.mp4.orig.$S $VG/out/$n.mp4;
      echo "  rendu $n : $(stat -c %s $VG/out/$n.mp4) octets"; }
  done
}
trap restaure EXIT
for n in final_$ID av_$ID; do
  [ -f $VG/out/$n.mp4 ] && cp -f $VG/out/$n.mp4 $VG/out/$n.mp4.orig.$S
done

echo "== 1/6 voix animateur sur le master vert =="
if [ ! -s $AG/out/voix_$ID.mp4 ]; then
  $PY $MV/scripts/style_dub_srt.py $VG/out/b_$ID.mp4 $SRT $AG/out/voix_$ID.mp4 \
    --work /tmp/rf_$ID 2>&1 | tail -3
fi
ffmpeg -y -v error -i $AG/out/voix_$ID.mp4 -vn -ac 1 -ar 16000 $AG/out/voix_$ID.wav
echo "  voix : $(ffprobe -v error -show_entries format=duration -of csv=p=0 $AG/out/voix_$ID.wav) s"

echo "== 2/6 coupures du montage =="
# -v info OBLIGATOIRE : avec -v error, showinfo ne sort rien et on croit qu il n y a
# aucune coupure.
[ -s $AG/out/coupures_$ID.txt ] || ffmpeg -v info -i $VG/out/b_$ID.mp4 \
  -vf "select=gt(scene\,0.30),showinfo" -an -f null - 2>&1 \
  | grep -oE "pts_time:[0-9.]+" | cut -d: -f2 > $AG/out/coupures_$ID.txt
echo "  $(wc -l < $AG/out/coupures_$ID.txt) coupures detectees"

echo "== 3/6 piste avatar =="
CL=$(ls $AG/out/bande_courte/*.webm | tr '\n' ',' | sed 's/,$//')
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $VG/out/b_$ID.mp4)
$AG/.venv-sd/bin/python $AG/piste_avatar.py $AG/out/piste_$ID.mp4 $AG/out/voix_$ID.wav \
  --clips "$CL" --dur $DUR --mode segments --offset 0 --segments $WD/host_map.json \
  --coupures $AG/out/coupures_$ID.txt --espacement $ESP \
  --fond $AG/out/bg_gen/poste_3ecrans_s2_vert.png --fondu 0 --mains 2021-07-20 2>&1 | tail -4

echo "== 4/6 avatar (vert + pop-out) =="
cd $VG
$PY $VG/vf_avatar.py "$ID" --avatar $AG/out/piste_$ID.mp4 2>&1 | tail -1
VF_AVATAR_POPOUT=$AG/out/piste_$ID.mp4 bash $VG/vf_finalize.sh "$ID" 2>&1 | tail -1

echo "== 5/6 voix + neige =="
D=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $VG/out/final_$ID.mp4)
FN=$(python3 -c "print('%.3f' % ($D - 1.5))")
cat > /tmp/filtre_$ID.txt <<FILTRE
[1:a]adelay=1500|1500[voix];
[0:a]volume=0:enable='between(t,1.5,$FN)'[neige];
[voix][neige]amix=inputs=2:normalize=0,alimiter=limit=0.95[out]
FILTRE
ffmpeg -y -v error -i $VG/out/final_$ID.mp4 -i $AG/out/voix_$ID.mp4 \
  -filter_complex_script /tmp/filtre_$ID.txt \
  -map 0:v -map "[out]" -c:v copy -c:a aac -b:a 160k -shortest $AG/out/corps_$ID.mp4
echo "  neige : $(ffmpeg -v info -ss 0 -t 1.4 -i $AG/out/corps_$ID.mp4 -af volumedetect -f null - 2>&1 | grep max_volume | grep -oE '[-0-9.]+ dB' | head -1)"

echo "== 6/6 generique =="
$PY $VG/vf_intro_outro.py $AG/out/corps_$ID.mp4 $AG/out/LIVRABLE_$ID.mp4 2>&1 | tail -5
echo "===== CHAINE TERMINEE $ID $(date +%T) ====="
ls -lh $AG/out/LIVRABLE_$ID.mp4 | awk '{print $5, $9}'
