#!/bin/bash
# run_anon3.sh — serie corrigee : genere, TRIE les masques reussis, anime, livre.
BASE=$HOME/avatar_gen
PYSD=$BASE/.venv-sd/bin/python
LIV=$BASE/out/livraison_anon3
rm -rf "$LIV"; mkdir -p "$LIV"

while pgrep -f "gen_anon2.py|anim_rigid.py" >/dev/null 2>&1; do sleep 10; done

echo "===== generation serie 3 (prompt corrige) ====="
$PYSD $BASE/gen_anon3.py 2>&1 | grep -v "^\[transformers\]" | tail -16
ls $BASE/out/anon3/*.png >/dev/null 2>&1 || { echo "AUCUNE IMAGE"; exit 1; }

echo "===== animation geometrique ====="
for f in $BASE/out/anon3/*_s1.png $BASE/out/anon3/*_s2.png; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .png)
  $PYSD $BASE/anim_rigid.py "$f" "$LIV/${n}_anim.mp4" --dur 12 2>&1 | tail -1
done

cp $BASE/out/anon3/*.png "$LIV/"
# la seule reussite des series precedentes, pour comparaison
cp $BASE/out/anon/anon_lisse_buste_s1.png "$LIV/SERIE1_seule_reussite_lisse.png" 2>/dev/null
echo "===== TERMINE $(date '+%F %T') ====="
ls "$LIV" | wc -l
