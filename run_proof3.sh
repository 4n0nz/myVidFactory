#!/bin/bash
# run_proof3.sh — genere les visages (SDXL), anime les 2 voies, livre.
# On ne guette plus un marqueur dans un log d install (source des 2 faux departs) :
# on verifie l ETAT REEL, c est-a-dire que le modele est la.
BASE=$HOME/avatar_gen
PYSD=$BASE/.venv-sd/bin/python      # diffusers
PYLP=$BASE/.venv/bin/python         # LivePortrait
LIV=$BASE/out/livraison
rm -rf "$LIV"; mkdir -p "$LIV"

[ -f "$BASE/models/sdxl/model_index.json" ] || { echo "MODELE SDXL ABSENT"; exit 1; }
echo "modele present : $(du -sh $BASE/models/sdxl | cut -f1)"

echo "===== 1/3 generation des visages (SDXL) ====="
$PYSD $BASE/gen_face_sdxl.py 2>&1 | grep -v "^\[transformers\]" | tail -20
ls $BASE/out/faces/*.png >/dev/null 2>&1 || { echo "AUCUNE IMAGE GENEREE"; exit 1; }

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
  echo "  video de conduite : $(basename $DRV)"
  for f in $BASE/out/faces/partiel_*.png; do
    [ -f "$f" ] || continue
    n=$(basename "$f" .png)
    if timeout 900 $PYLP inference.py -s "$f" -d "$DRV" -o "$BASE/out/lp_$n" > "/tmp/lp_$n.log" 2>&1; then
      r=$(ls "$BASE/out/lp_$n"/*.mp4 2>/dev/null | grep -v concat | head -1)
      if [ -n "$r" ]; then cp "$r" "$LIV/B_${n}.mp4"; echo "  OK $n"; else echo "  $n : aucun mp4 en sortie"; fi
    else
      echo "  ECHEC LivePortrait sur $n :"; tail -4 "/tmp/lp_$n.log"
    fi
  done
fi

cp ~/videogen/public/avatar.mp4 "$LIV/REFERENCE_avatar_actuel_848x464.mp4" 2>/dev/null
cp $BASE/out/faces/*.png "$LIV/" 2>/dev/null
echo "===== TERMINE $(date '+%F %T') ====="
ls -lh "$LIV"
