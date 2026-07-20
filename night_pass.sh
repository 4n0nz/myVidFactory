#!/bin/bash
# night_pass.sh — fournee post-batch : re-runs avec tous les fixes + le special gnfH (full).
# Append au MEME rapport ; NIGHT_DONE a la fin. Attend le flock du batch.
PY=/home/boss/videogen/.venv/bin/python
VG=/home/boss/videogen
REPORT=/tmp/batch20_report.tsv
PROG=/tmp/batch20.progress
LIST="4D7YDhsV-jM AAmdB1bvmYw jZpOI5petho kDM2rBcvWh0 Iup815Xz_ZU gnfHlIoh34Q"
exec 9>/tmp/vf_render.flock; flock 9   # bloque jusqu'a liberation par le batch

for id in $LIST; do
  WD=$VG/wk_b_$id
  [ -f "$WD/source.mp4" ] || { printf "%s\tNO_SOURCE\t-\t-\t-\t-\t-\t-\n" "R_$id" >> "$REPORT"; continue; }
  OUT="b_${id}.mp4"
  echo "[NUIT] $id — pinpoint3..." >> "$PROG"
  $PY $VG/pinpoint3.py "$WD" > /tmp/np_pin.log 2>&1
  if [ ! -s "$WD/narrator_feat.npy" ]; then
    printf "%s\tNO_NARRATOR\t-\t-\t-\t-\t-\t-\n" "R_$id" >> "$REPORT"; continue
  fi
  $PY $VG/pin_render.py "$WD" > /tmp/np_pr.log 2>&1
  heros=$(grep -oE 'heros: [0-9]+' /tmp/np_pr.log | grep -oE '[0-9]+' | tail -1)
  masks=$(grep -oE 'masques uniques: [0-9]+' /tmp/np_pr.log | grep -oE '[0-9]+' | tail -1)
  scenes=$($PY -c "import json;print(len(json.load(open('$WD/host_map.json'))))" 2>/dev/null)
  $PY $VG/build_seg.py "$WD" "$OUT" > /dev/null 2>&1 && "$WD/run_seg.sh" > /dev/null 2>&1
  if [ ! -s "$VG/out/$OUT" ]; then
    printf "%s\tRENDER_FAIL\t%s\t%s\t%s\t-\t-\t-\n" "R_$id" "$scenes" "$heros" "$masks" >> "$REPORT"; continue
  fi
  qc="LEAK"; tours=0
  for i in 1 2 3; do
    tours=$i
    if $PY $VG/qc_ident.py "$WD" "$VG/out/$OUT" > /tmp/np_qc.log 2>&1; then qc="CLEAN"; break; fi
    $PY $VG/qc_fix.py "$WD" > /dev/null 2>&1
    $PY $VG/pin_render.py "$WD" > /dev/null 2>&1
    $PY $VG/build_seg.py "$WD" "$OUT" > /dev/null 2>&1 && "$WD/run_seg.sh" > /dev/null 2>&1
  done
  [ "$qc" = "LEAK" ] && $PY $VG/qc_ident.py "$WD" "$VG/out/$OUT" > /tmp/np_qc.log 2>&1 && qc="CLEAN"
  leaks=$(grep -oE 'QC FUITES : [0-9]+' /tmp/np_qc.log | grep -oE '[0-9]+' | tail -1)
  [ "$qc" = "LEAK" ] && qc="LEAK_${leaks}"
  sdur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WD/source.mp4")
  vdur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WD/segs/videoonly.mp4" 2>/dev/null)
  durok=$($PY -c "print('OK' if abs($sdur-($vdur or 0))<0.5 else 'DESYNC')" 2>/dev/null)
  sz=$(du -h "$VG/out/$OUT" | cut -f1)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "R_$id" "$qc" "$scenes" "$heros" "$masks" "$durok" "$sz" "$tours" >> "$REPORT"
  echo "[NUIT] $id — $qc" >> "$PROG"
done
echo "NIGHT_DONE $(date)" >> "$PROG"
