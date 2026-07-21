#!/bin/bash
# batch20.sh — passe tout le corpus en pinpoint3 + boucle QC. Serie (GPU unique) sous flock.
# Rapport TSV : id | qc | scenes | heros | masques | dur_ok | taille | tours
PY=/home/boss/videogen/.venv/bin/python
VG=/home/boss/videogen
export GREEN_PIP=1   # passe verte : pip source -> vert chroma (Boss 2026-07-20)
REPORT=/tmp/batch20_report.tsv
PROG=/tmp/batch20.progress
exec 9>/tmp/vf_render.flock; flock -n 9 || { echo "FLOCK_BUSY — un autre render tourne, abort"; exit 1; }

printf "id\tqc\tgeom\tscenes\theros\tmasques\tdur_ok\ttaille\ttours\n" > "$REPORT"
: > "$PROG"
n=0; tot=$(ls -d $VG/wk_b_*/ 2>/dev/null | wc -l)
for WD in $VG/wk_b_*/; do
  WD=${WD%/}; id=$(basename "$WD" | sed 's/^wk_b_//')
  [ -f "$WD/source.mp4" ] || { printf "%s\tNO_SOURCE\t-\t-\t-\t-\t-\t-\n" "$id" >> "$REPORT"; continue; }
  n=$((n+1)); OUT="b_${id}.mp4"
  echo "[$n/$tot] $id — pinpoint3..." >> "$PROG"

  plog=$($PY $VG/pinpoint3.py "$WD" 2>&1)
  $PY $VG/box_consensus.py "$WD" > /tmp/bc.log 2>&1
  narr=$(echo "$plog" | grep -o 'narrateur: [0-9]*/[0-9]*' | head -1)
  if [ ! -s "$WD/narrator_feat.npy" ]; then
    printf "%s\tNO_NARRATOR\t-\t-\t-\t-\t-\t-\n" "$id" >> "$REPORT"; continue
  fi

  $PY $VG/pin_render.py "$WD" > /tmp/pr.log 2>&1
  rl=$(grep -oE 'heros: [0-9]+ / masques uniques: [0-9]+ / groupes position: [0-9]+' /tmp/pr.log | tail -1)
  heros=$(echo "$rl" | grep -oE 'heros: [0-9]+' | grep -oE '[0-9]+')
  masks=$(echo "$rl" | grep -oE 'masques uniques: [0-9]+' | grep -oE '[0-9]+')
  scenes=$($PY -c "import json;print(len(json.load(open('$WD/host_map.json'))))" 2>/dev/null)

  $PY $VG/build_seg.py "$WD" "$OUT" > /tmp/bs.log 2>&1 && "$WD/run_seg.sh" > /dev/null 2>&1
  if [ ! -s "$VG/out/$OUT" ]; then
    printf "%s\tRENDER_FAIL\t%s\t%s\t%s\t-\t-\t-\n" "$id" "$scenes" "$heros" "$masks" >> "$REPORT"; continue
  fi

  qc="LEAK"; tours=0
  for i in 1 2 3 4 5; do
    tours=$i
    if $PY $VG/qc_ident.py "$WD" "$VG/out/$OUT" > /tmp/qc.log 2>&1; then qc="CLEAN"; break; fi
    $PY $VG/qc_fix.py "$WD" > /dev/null 2>&1
    $PY $VG/pin_render.py "$WD" > /dev/null 2>&1
    $PY $VG/build_seg.py "$WD" "$OUT" > /dev/null 2>&1 && "$WD/run_seg.sh" > /dev/null 2>&1
  done
  [ "$qc" = "LEAK" ] && $PY $VG/qc_ident.py "$WD" "$VG/out/$OUT" > /tmp/qc.log 2>&1 && qc="CLEAN"
  leaks=$(grep -oE 'QC FUITES : [0-9]+' /tmp/qc.log | grep -oE '[0-9]+' | tail -1)
  [ "$qc" = "LEAK" ] && qc="LEAK_${leaks}"

  # QC geometrique CORRECTIF : sous-couverture/trop-grand -> box ajustee a la carte reelle,
  # re-render, max 2 tours (les fuites SANS visage sont invisibles au QC identite - TzJC torse)
  geom="FAIL"
  for j in 1 2 3; do
    if $PY $VG/qc_geom.py "$WD" > /tmp/qg.log 2>&1; then geom="OK"; break; fi
    [ $j -eq 3 ] && break
    $PY $VG/qc_geom_fix.py "$WD" > /dev/null 2>&1
    $PY $VG/pin_render.py "$WD" > /dev/null 2>&1
    $PY $VG/build_seg.py "$WD" "$OUT" > /dev/null 2>&1 && "$WD/run_seg.sh" > /dev/null 2>&1
  done
  if [ "$geom" != "OK" ]; then
    geom="GEOM_$(grep -oE 'ECHECS : [0-9]+' /tmp/qg.log | grep -oE '[0-9]+')"
    # re-verif identite apres les re-renders geometriques
    $PY $VG/qc_ident.py "$WD" "$VG/out/$OUT" > /tmp/qc.log 2>&1 && qc="CLEAN" || qc="LEAK_$(grep -oE 'QC FUITES : [0-9]+' /tmp/qc.log | grep -oE '[0-9]+' | tail -1)"
  fi
  sdur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WD/source.mp4")
  vdur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$WD/segs/videoonly.mp4" 2>/dev/null)
  durok=$($PY -c "print('OK' if abs($sdur-($vdur or 0))<0.5 else 'DESYNC(%.1f)'%($vdur or 0))" 2>/dev/null)
  sz=$(du -h "$VG/out/$OUT" | cut -f1)

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$id" "$qc" "$geom" "$scenes" "$heros" "$masks" "$durok" "$sz" "$tours" >> "$REPORT"
  echo "[$n/$tot] $id — $qc ($narr, scenes=$scenes)" >> "$PROG"
done
echo "BATCH_DONE $(date)" >> "$PROG"
