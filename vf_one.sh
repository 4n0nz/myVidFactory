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

# INVARIANTS : controle TOTAL du host_map final — zero pixel, zero echantillonnage,
# donc zero angle mort par construction. Complement des juges perceptuels, qui eux
# echantillonnent et ont donc des trous (qc_ident : 1 frame/s + tolerance +-0.6 s dans
# covered_by_avatar = aucun trou de frontiere sous 1.2 s ne peut etre flagge, 71 trous
# passes CLEAN le 2026-08-01). Ne relance AUCUNE passe : c est un rapport, pas un fixer.
echo "invariants..." >> "$PROG"
inv="OK"
$PY $VG/vf_invariants.py "$WD" > /tmp/vf_one_inv.log 2>&1 || inv="INV_$(grep -oE "INVARIANTS : [0-9]+" /tmp/vf_one_inv.log | grep -oE "[0-9]+" | tail -1)"
_ns=$(grep -oE "SUSPECTS : [0-9]+" /tmp/vf_one_inv.log | grep -oE "[0-9]+" | tail -1)
[ "${_ns:-0}" != "0" ] && inv="${inv}+SUSP${_ns}"

# JUGES ANCRES SOURCE (rapport seul, plan Efforts/VF-Plan-Robustesse.md #1) :
#   hc = la tete du narrateur (trouvee dans la SOURCE) est-elle verte dans le rendu la
#        ou le plan dit pip/hero ? Chercher dans le rendu est vain : le vert cache pile
#        la box du detecteur (o9x8 t=428, qc_ident disait "0 fuite" sur une demi-tete).
#   op = narrateur LIVE dans les scenes off (le trou de rognage itWI 331-351 etait
#        inv=OK : une scene off est legale, mais elle ne peint rien) + marge tete
#        pre-rendu sur les pip.
# AUCUN fixer ne consomme ces verdicts — ils pointent, la loop confirme aux frames.
echo "headcover..." >> "$PROG"
hc="OK"
$PY $VG/vf_headcover.py "$WD" "$VG/out/$OUT" --probes 3 --thr 0.80 > /tmp/vf_one_hc.log 2>&1 || hc="HC_$(grep -cE "^  scene\[" /tmp/vf_one_hc.log)"
echo "offprobe..." >> "$PROG"
op="OK"
$PY $VG/vf_offprobe.py "$WD" > /tmp/vf_one_op.log 2>&1 || op="OP_$(grep -cE "OFF-LIVE|MARGE-TETE" /tmp/vf_one_op.log)"
echo "overcover..." >> "$PROG"
oc="OK"
$PY $VG/vf_overcover.py "$WD" > /tmp/vf_one_oc.log 2>&1 || oc="OC_$(grep -cE "SUR_struct|SUR_cont " /tmp/vf_one_oc.log)"

# FINALIZE (presentation) : composite du master vert devant le background anime,
# 1,5 s de fond seul avant et apres, cote auto selon la position des pips.
# Le master b_<id>.mp4 reste la reference pour l avatar.
echo "finalize..." >> "$PROG"
fin="OK"
bash $VG/vf_finalize.sh "$id" > /tmp/vf_one_fin.log 2>&1 || fin="FIN_FAIL"

sdur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WD/source.mp4")
vdur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WD/segs/videoonly.mp4" 2>/dev/null)
durok=$($PY -c "print('OK' if abs($sdur-($vdur or 0))<0.5 else 'DESYNC(%.1f)'%($vdur or 0))" 2>/dev/null)
echo "DONE qc=$qc geom=$geom fid=$fid inv=$inv hc=$hc op=$op oc=$oc fin=$fin tours=$tours dur=$durok $(date '+%F %T')" >> "$PROG"
