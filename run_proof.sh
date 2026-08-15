#!/bin/bash
# run_proof.sh — attend la fin de l install, genere les visages, anime les deux voies,
# et depose tout dans ~/avatar_gen/out/livraison.
BASE=$HOME/avatar_gen
PY=$BASE/.venv/bin/python
LIV=$BASE/out/livraison
mkdir -p "$LIV"

echo "===== attente fin d install ====="
for i in $(seq 1 120); do
  grep -q "INSTALL TERMINEE" /tmp/install_avatar_gen.log 2>/dev/null && break
  pgrep -f install_avatar_gen >/dev/null || { grep -q "INSTALL TERMINEE" /tmp/install_avatar_gen.log || { echo "install morte sans fin propre :"; tail -15 /tmp/install_avatar_gen.log; exit 1; }; }
  sleep 30
done
echo "install OK $(date '+%T')"

echo "===== 1/3 generation des visages ====="
$PY $BASE/gen_face.py 2>&1 | tail -20 || { echo "GEN FAIL"; exit 1; }

echo "===== 2/3 voie A : animation geometrique (masque integral) ====="
for f in $BASE/out/faces/integral_*_s1.png; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .png)
  $PY $BASE/anim_rigid.py "$f" "$LIV/A_${n}.mp4" --dur 12 2>&1 | tail -2
done

echo "===== 3/3 voie B : LivePortrait (masque partiel, yeux visibles) ====="
cd $BASE/LivePortrait || exit 1
DRV=$(ls assets/examples/driving/*.mp4 2>/dev/null | head -1)
if [ -z "$DRV" ]; then
  echo "  pas de driving d exemple dans le repo — voie B sautee"
else
  echo "  driving : $DRV"
  for f in $BASE/out/faces/partiel_*_s1.png; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .png)
    if $PY inference.py -s "$f" -d "$DRV" -o "$BASE/out/lp_$n" --flag_pasteback true > "/tmp/lp_$n.log" 2>&1; then
      r=$(ls "$BASE/out/lp_$n"/*.mp4 2>/dev/null | grep -v concat | head -1)
      [ -n "$r" ] && cp "$r" "$LIV/B_${n}.mp4" && echo "  OK $n"
    else
      echo "  ECHEC LivePortrait sur $n (attendu si les reperes du visage sont masques) :"
      tail -3 "/tmp/lp_$n.log"
    fi
  done
fi

# reference : l avatar actuel, pour comparer a taille reelle
cp ~/videogen/public/avatar.mp4 "$LIV/REFERENCE_avatar_actuel_848x464.mp4" 2>/dev/null
cp $BASE/out/faces/*.png "$LIV/" 2>/dev/null

echo "===== TERMINE $(date '+%F %T') ====="
ls -lh "$LIV" | tail -25
