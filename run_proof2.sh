#!/bin/bash
# run_proof2.sh — attend le fix, genere les visages (SDXL), anime les 2 voies, livre.
BASE=$HOME/avatar_gen
PYSD=$BASE/.venv-sd/bin/python      # diffusers
PYLP=$BASE/.venv/bin/python         # LivePortrait
LIV=$BASE/out/livraison
rm -rf "$LIV"; mkdir -p "$LIV"

echo "===== attente du fix d install ====="
for i in $(seq 1 120); do
  grep -q "FIX OK" /tmp/fix_install.log 2>/dev/null && break
  if ! pgrep -f fix_install >/dev/null && ! grep -q "FIX OK" /tmp/fix_install.log 2>/dev/null; then
    echo "FIX MORT sans fin propre :"; tail -20 /tmp/fix_install.log; exit 1
  fi
  sleep 30
done
echo "fix OK $(date '+%T')"

echo "===== 1/3 generation des visages (SDXL) ====="
$PYSD $BASE/gen_face_sdxl.py 2>&1 | tail -20 || { echo "GEN FAIL"; tail -20 /tmp/run_proof2.log; exit 1; }

echo "===== 2/3 voie A : animation geometrique (masque integral) ====="
for f in $BASE/out/faces/integral_*.png; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .png)
  $PYSD $BASE/anim_rigid.py "$f" "$LIV/A_${n}.mp4" --dur 12 2>&1 | tail -2
done

echo "===== 3/3 voie B : LivePortrait (masque partiel, yeux visibles) ====="
cd $BASE/LivePortrait || exit 1
DRV=$(ls assets/examples/driving/*.mp4 2>/dev/null | head -1)
if [ -z "$DRV" ]; then
  echo "  aucune video de conduite d exemple — voie B sautee"
else
  echo "  video de conduite : $DRV"
  for f in $BASE/out/faces/partiel_*.png; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .png)
    if $PYLP inference.py -s "$f" -d "$DRV" -o "$BASE/out/lp_$n" > "/tmp/lp_$n.log" 2>&1; then
      r=$(ls "$BASE/out/lp_$n"/*.mp4 2>/dev/null | grep -v concat | head -1)
      if [ -n "$r" ]; then cp "$r" "$LIV/B_${n}.mp4"; echo "  OK $n"; else echo "  $n : pas de mp4 en sortie"; fi
    else
      echo "  ECHEC LivePortrait sur $n :"; tail -3 "/tmp/lp_$n.log"
    fi
  done
fi

cp ~/videogen/public/avatar.mp4 "$LIV/REFERENCE_avatar_actuel_848x464.mp4" 2>/dev/null
cp $BASE/out/faces/*.png "$LIV/" 2>/dev/null
echo "===== TERMINE $(date '+%F %T') ====="
ls -lh "$LIV"
