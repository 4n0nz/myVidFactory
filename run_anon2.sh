#!/bin/bash
# run_anon2.sh — attend que la serie 1 libere le GPU, genere la serie 2 (portrait serre),
# anime, et rassemble TOUT dans une seule livraison.
BASE=$HOME/avatar_gen
PYSD=$BASE/.venv-sd/bin/python
LIV=$BASE/out/livraison_anon
mkdir -p "$LIV"

echo "===== attente de la serie 1 (un seul GPU) ====="
while pgrep -f "gen_anon.py|run_anon.sh|inference.py" >/dev/null 2>&1; do sleep 20; done
echo "GPU libre $(date '+%T')"

echo "===== generation serie 2 (portrait serre) ====="
$PYSD $BASE/gen_anon2.py 2>&1 | grep -v "^\[transformers\]" | tail -16
ls $BASE/out/anon2/*.png >/dev/null 2>&1 || { echo "AUCUNE IMAGE SERIE 2"; exit 1; }

echo "===== animation (geometrique — masques rigides) ====="
for f in $BASE/out/anon2/*_s1.png; do
  n=$(basename "$f" .png)
  $PYSD $BASE/anim_rigid.py "$f" "$LIV/${n}_anim.mp4" --dur 12 2>&1 | tail -1
done

cp $BASE/out/anon2/*.png "$LIV/" 2>/dev/null
echo "===== TERMINE $(date '+%F %T') ====="
ls "$LIV" | wc -l
