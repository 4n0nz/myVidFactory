#!/bin/bash
# vf_multi.sh <id> [<id> ...] — enchaine vf_one.sh sur plusieurs videos, en serie.
# Detache : setsid nohup ./vf_multi.sh a b c > /tmp/vf_multi.log 2>&1 &
# Progression globale : /tmp/vf_multi.progress (une ligne START/END par video).
# Chaque video garde son propre /tmp/vf_one_<id>.progress.
VG=/home/boss/videogen
PROG=/tmp/vf_multi.progress
: > "$PROG"
echo "MULTI_START $(date '+%F %T') : $*" >> "$PROG"
for id in "$@"; do
  if [ ! -f "$VG/wk_b_$id/source.mp4" ]; then
    echo "SKIP $id (pas de source)" >> "$PROG"; continue
  fi
  echo "START $id $(date '+%F %T')" >> "$PROG"
  "$VG/vf_one.sh" "$id"
  echo "END $id $(date '+%F %T') : $(tail -1 /tmp/vf_one_${id}.progress 2>/dev/null)" >> "$PROG"
done
echo "MULTI_DONE $(date '+%F %T')" >> "$PROG"
