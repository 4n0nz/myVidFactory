#!/bin/bash
# vf_one.sh <video_id> — chaîne complète pour UNE vidéo, sous flock.
# Conçu pour tourner détaché : setsid nohup ./vf_one.sh <id> > /tmp/vf_one_<id>.log 2>&1 &
# Progression : /tmp/vf_one_<id>.progress — dernière ligne DONE ... = passe finie.
id=$1
[ -n "$id" ] || { echo "usage: vf_one.sh <video_id>"; exit 2; }
PY=/home/boss/videogen/.venv/bin/python
VG=/home/boss/videogen
WD=$VG/wk_b_$id
OUT=b_${id}.mp4
PROG=/tmp/vf_one_${id}.progress
export GREEN_PIP=1
[ -f "$WD/source.mp4" ] || { echo "NO_SOURCE $WD" | tee -a "$PROG"; exit 1; }

# encodeur : NVENC si vivant, sinon CPU
if ffmpeg -y -v error -f lavfi -i color=c=red:s=320x180:r=30 -t 1 -c:v h264_nvenc /tmp/nvenc_probe.mp4 2>/dev/null; then
  export VF_ENC=gpu
else
  export VF_ENC=cpu
fi

exec 9>/tmp/vf_render.flock; flock -n 9 || { echo "FLOCK_BUSY — abort" | tee -a "$PROG"; exit 1; }
echo "START $(date '+%F %T') enc=$VF_ENC" >> "$PROG"

echo "pinpoint3..." >> "$PROG"
$PY $VG/pinpoint3.py "$WD" > /tmp/vf_one_pp.log 2>&1
$PY $VG/box_consensus.py "$WD" > /tmp/vf_one_bc.log 2>&1
[ -s "$WD/narrator_feat.npy" ] || { echo "DONE NO_NARRATOR $(date '+%F %T')" >> "$PROG"; exit 1; }

echo "pin_render..." >> "$PROG"
$PY $VG/pin_render.py "$WD" > /tmp/vf_one_pr.log 2>&1

echo "render..." >> "$PROG"
$PY $VG/build_seg.py "$WD" "$OUT" > /tmp/vf_one_bs.log 2>&1 && "$WD/run_seg.sh" > /dev/null 2>&1
if [ ! -s "$VG/out/$OUT" ]; then
  if [ "$VF_ENC" != "cpu" ] && ! ffmpeg -y -v error -f lavfi -i color=c=red:s=320x180:r=30 -t 1 -c:v h264_nvenc /tmp/nvenc_probe.mp4 2>/dev/null; then
    echo "NVENC_DOWN — bascule CPU" >> "$PROG"; export VF_ENC=cpu
    $PY $VG/build_seg.py "$WD" "$OUT" > /dev/null 2>&1 && "$WD/run_seg.sh" > /dev/null 2>&1
  fi
  [ -s "$VG/out/$OUT" ] || { echo "DONE RENDER_FAIL $(date '+%F %T')" >> "$PROG"; exit 1; }
fi

qc="LEAK"; tours=0
for i in 1 2 3 4 5; do
  tours=$i
  echo "qc_ident tour $i..." >> "$PROG"
  if $PY $VG/qc_ident.py "$WD" "$VG/out/$OUT" > /tmp/vf_one_qc.log 2>&1; then qc="CLEAN"; break; fi
  $PY $VG/qc_fix.py "$WD" > /dev/null 2>&1
  $PY $VG/pin_render.py "$WD" > /dev/null 2>&1
  $PY $VG/build_seg.py "$WD" "$OUT" > /dev/null 2>&1 && "$WD/run_seg.sh" > /dev/null 2>&1
done
[ "$qc" = "LEAK" ] && $PY $VG/qc_ident.py "$WD" "$VG/out/$OUT" > /tmp/vf_one_qc.log 2>&1 && qc="CLEAN"
leaks=$(grep -oE 'QC FUITES : [0-9]+' /tmp/vf_one_qc.log | grep -oE '[0-9]+' | tail -1)
[ "$qc" = "LEAK" ] && qc="LEAK_${leaks}"

geom="FAIL"
for j in 1 2 3; do
  echo "qc_geom tour $j..." >> "$PROG"
  if $PY $VG/qc_geom.py "$WD" > /tmp/vf_one_qg.log 2>&1; then geom="OK"; break; fi
  [ $j -eq 3 ] && break
  $PY $VG/qc_geom_fix.py "$WD" > /dev/null 2>&1
  $PY $VG/pin_render.py "$WD" > /dev/null 2>&1
  $PY $VG/build_seg.py "$WD" "$OUT" > /dev/null 2>&1 && "$WD/run_seg.sh" > /dev/null 2>&1
done
if [ "$geom" != "OK" ]; then
  geom="GEOM_$(grep -oE 'ECHECS : [0-9]+' /tmp/vf_one_qg.log | grep -oE '[0-9]+')"
  $PY $VG/qc_ident.py "$WD" "$VG/out/$OUT" > /tmp/vf_one_qc.log 2>&1 && qc="CLEAN" || qc="LEAK_$(grep -oE 'QC FUITES : [0-9]+' /tmp/vf_one_qc.log | grep -oE '[0-9]+' | tail -1)"
fi

# QC FIDELITE : le vert coincide-t-il avec la CARTE mesuree dans la SOURCE ?
# (juge ajoute le 2026-07-28 — ni qc_ident ni qc_geom ne voyaient la SUR-couverture
# ni les faux pips ; ils remontaient donc a Boss passe apres passe.)
fid="FAIL"
for k in 1 2; do
  echo "qc_fid tour $k..." >> "$PROG"
  if $PY $VG/qc_fid.py "$WD" "$VG/out/$OUT" > /tmp/vf_one_qf.log 2>&1; then fid="OK"; break; fi
  [ $k -eq 2 ] && break
  $PY $VG/qc_fid_fix.py "$WD" >> /tmp/vf_one_qf.log 2>&1
  $PY $VG/pin_render.py "$WD" > /dev/null 2>&1
  $PY $VG/build_seg.py "$WD" "$OUT" > /dev/null 2>&1 && "$WD/run_seg.sh" > /dev/null 2>&1
done
if [ "$fid" != "OK" ]; then
  _nf=$(grep -oE 'QC-FID ECHECS : [0-9]+' /tmp/vf_one_qf.log | grep -oE '[0-9]+' | tail -1)
  _nc=$(grep -oE 'PAS-DE-CARTE : [0-9]+' /tmp/vf_one_qf.log | grep -oE '[0-9]+' | tail -1)
  fid="FID_${_nf:-0}"
  [ "${_nc:-0}" != "0" ] && fid="${fid}_NOCARD${_nc}"
fi

sdur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WD/source.mp4")
vdur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WD/segs/videoonly.mp4" 2>/dev/null)
durok=$($PY -c "print('OK' if abs($sdur-($vdur or 0))<0.5 else 'DESYNC(%.1f)'%($vdur or 0))" 2>/dev/null)
echo "DONE qc=$qc geom=$geom fid=$fid tours=$tours dur=$durok $(date '+%F %T')" >> "$PROG"
