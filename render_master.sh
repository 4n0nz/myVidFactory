#!/usr/bin/env bash
# Master complet — host module (avatar + terminal) intégré DANS Remotion.
# Plus d'overlay alpha séparé : un seul rendu.
set -e
cd /home/anon/videogen
. .venv/bin/activate

echo "==== 1/4 voix off + manifest (toutes scènes) ===="
python build_audio.py script_qc.json

echo "==== 2/4 cues des scènes 'action' (whisper word-timing) ===="
python build_cues.py

echo "==== 3/4 sous-titres terminal (texte propre, pour le host) ===="
python gen_captions.py script_qc.json

echo "==== 4/4 rendu unique (scènes + host avatar/terminal + voix) ===="
npx remotion render src/index.ts Tutorial out/master_final.mp4 --concurrency=3 --log=error

echo "==== DONE ===="
ls -la out/master_final.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 out/master_final.mp4
