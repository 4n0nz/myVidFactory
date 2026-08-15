#!/bin/bash
BASE=$HOME/avatar_gen
PYSD=$BASE/.venv-sd/bin/python
LIV=$BASE/out/livraison_anon4
rm -rf "$LIV"; mkdir -p "$LIV"
echo "===== generation serie 4 (prompt court, recette validee) ====="
$PYSD $BASE/gen_anon4.py 2>&1 | grep -v "^\[transformers\]" | tail -16
ls $BASE/out/anon4/*.png >/dev/null 2>&1 || { echo "AUCUNE IMAGE"; exit 1; }
echo "===== animation geometrique ====="
for f in $BASE/out/anon4/*_s1.png $BASE/out/anon4/*_s2.png; do
  [ -f "$f" ] || continue
  n=$(basename "$f" .png)
  $PYSD $BASE/anim_rigid.py "$f" "$LIV/${n}_anim.mp4" --dur 12 2>&1 | tail -1
done
cp $BASE/out/anon4/*.png "$LIV/"
cp $BASE/out/anon/anon_lisse_buste_s1.png "$LIV/REFERENCE_recette_validee.png" 2>/dev/null
echo "===== TERMINE $(date "+%F %T") ====="
ls "$LIV" | wc -l
