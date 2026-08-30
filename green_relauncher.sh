#!/bin/bash
# Attend la fin du batch courant (flock libre) puis lance la passe VERTE consensus a zero.
# Ne re-tente QUE sur FLOCK_BUSY — un vrai crash ne doit pas boucler-relancer.
while true; do
  rm -f /tmp/batch20_report.tsv.green.pending
  out=$(cd ~/videogen && ./batch20.sh 2>&1)
  case "$out" in
    *FLOCK_BUSY*) sleep 300 ;;
    *) echo "$out"; echo "GREEN_BATCH_EXIT $(date)"; break ;;
  esac
done
