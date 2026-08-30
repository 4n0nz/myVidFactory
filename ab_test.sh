#!/bin/bash
VG=/home/boss/videogen
PY=$VG/.venv/bin/python
export GREEN_PIP=1
B=$VG/baseline_86cc362
A=$VG/ab_extlat
mkdir -p $A
echo "AB_START $(date '+%F %T')" > $A/progress.txt
for id in 4D7YDhsV-jM eglVxLaWRUU; do
  WD=$VG/wk_b_$id
  nice -n 15 $PY $VG/box_consensus.py "$WD" > $A/bc_$id.log 2>&1
  nice -n 15 $PY $VG/pin_render.py "$WD" > $A/pr_$id.log 2>&1
  cp "$WD/host_map.json" $A/host_map_$id.json
  cp "$WD/box_consensus.json" $A/box_consensus_$id.json
  echo "--- $id" >> $A/progress.txt
  if diff -q $B/host_map_$id.json $A/host_map_$id.json > /dev/null; then
    echo "host_map.json IDENTIQUE" >> $A/progress.txt
  else
    echo "host_map.json DIFFERENT :" >> $A/progress.txt
    diff $B/host_map_$id.json $A/host_map_$id.json >> $A/progress.txt
  fi
done
echo "AB_DONE $(date '+%F %T')" >> $A/progress.txt
