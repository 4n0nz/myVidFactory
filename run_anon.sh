#!/bin/bash
# run_anon.sh — genere la serie N&B inspiree de la reference, anime, livre.
BASE=$HOME/avatar_gen
PYSD=$BASE/.venv-sd/bin/python
PYLP=$BASE/.venv/bin/python
LIV=$BASE/out/livraison_anon
rm -rf "$LIV"; mkdir -p "$LIV"

echo "===== 1/3 generation (SDXL, N&B) ====="
$PYSD $BASE/gen_anon.py 2>&1 | grep -v "^\[transformers\]" | tail -20
ls $BASE/out/anon/*.png >/dev/null 2>&1 || { echo "AUCUNE IMAGE"; exit 1; }

echo "===== 2/3 animation geometrique (la voie juste pour un masque rigide) ====="
for f in $BASE/out/anon/*_s1.png; do
  n=$(basename "$f" .png)
  $PYSD $BASE/anim_rigid.py "$f" "$LIV/${n}_anim.mp4" --dur 12 2>&1 | tail -1
done

echo "===== 3/3 contre-epreuve LivePortrait sur un masque rigide ====="
# Prediction a verifier : LivePortrait prend les traits PEINTS du masque pour un visage
# et les deforme comme de la chair. On garde la sortie pour comparer aux boucles ci-dessus.
cd $BASE/LivePortrait || exit 1
DRV=$(ls assets/examples/driving/*.mp4 2>/dev/null | head -1)
SRC=$BASE/out/anon/anon_theatral_buste_s1.png
if [ -n "$DRV" ] && [ -f "$SRC" ]; then
  if timeout 900 $PYLP inference.py -s "$SRC" -d "$DRV" -o "$BASE/out/lp_anon" > /tmp/lp_anon.log 2>&1; then
    r=$(ls "$BASE/out/lp_anon"/*.mp4 2>/dev/null | grep -v concat | head -1)
    [ -n "$r" ] && cp "$r" "$LIV/CONTRE-EPREUVE_liveportrait_sur_masque_rigide.mp4" && \
      echo "  LivePortrait a produit une sortie — a comparer a l oeil"
  else
    echo "  LivePortrait a echoue (pas de reperes exploitables) :"; tail -3 /tmp/lp_anon.log
  fi
fi

cp $BASE/out/anon/*.png "$LIV/"
echo "===== TERMINE $(date '+%F %T') ====="
ls "$LIV"
