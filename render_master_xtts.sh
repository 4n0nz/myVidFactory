#!/usr/bin/env bash
# Master complet avec voix XTTS (GPU local, Damien Black). Tout dans .venv-xtts.
set -e
cd /home/anon/videogen
. .venv-xtts/bin/activate

echo "==== 1/4 voix off XTTS (GPU) + manifest ===="
python build_audio_xtts.py script_qc.json

echo "==== 2/4 cues des scènes 'action' (whisper) ===="
python build_cues.py

echo "==== 3/4 sous-titres terminal ===="
python gen_captions.py script_qc.json

echo "==== 4/4 rendu unique (scènes + host + voix XTTS) ===="
npx remotion render src/index.ts Tutorial out/master_xtts.mp4 --concurrency=3 --log=error

echo "==== DONE ===="
ls -la out/master_xtts.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 out/master_xtts.mp4
