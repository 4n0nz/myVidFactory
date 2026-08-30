#!/bin/bash
# Regenere la baseline des deux etalons avec le code courant (les host_map ont ete
# effaces au reset du 2026-08-01). Sortie hors des workdirs, pour A/B avant tout fix.
VG=/home/boss/videogen
PY=$VG/.venv/bin/python
export GREEN_PIP=1
B=$VG/baseline_$(git -C $VG rev-parse --short HEAD)
mkdir -p $B
echo "BASELINE_START $(date '+%F %T') HEAD=$(git -C $VG rev-parse --short HEAD)" > $B/progress.txt
for id in 4D7YDhsV-jM eglVxLaWRUU; do
  WD=$VG/wk_b_$id
  echo "pinpoint3 $id $(date '+%T')" >> $B/progress.txt
  nice -n 15 $PY $VG/pinpoint3.py "$WD" > $B/pp_$id.log 2>&1
  echo "box_consensus $id $(date '+%T')" >> $B/progress.txt
  nice -n 15 $PY $VG/box_consensus.py "$WD" > $B/bc_$id.log 2>&1
  echo "pin_render $id $(date '+%T')" >> $B/progress.txt
  nice -n 15 $PY $VG/pin_render.py "$WD" > $B/pr_$id.log 2>&1
  for f in host_map.json box_consensus.json host_map_pin.json; do
    cp "$WD/$f" "$B/${f%.json}_$id.json" 2>/dev/null || echo "MANQUE $f $id" >> $B/progress.txt
  done
  echo "OK $id $(date '+%T')" >> $B/progress.txt
done
echo "BASELINE_DONE $(date '+%F %T')" >> $B/progress.txt
