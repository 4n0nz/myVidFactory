#!/usr/bin/env bash
set -e
VIDGEN=/home/anon/videogen
WORKDIR=$VIDGEN/agent_yt
cd $VIDGEN
. .venv-xtts/bin/activate

# Use the YT script
cp $WORKDIR/script_qc.json $VIDGEN/script_qc.json

echo '==== 1/4 broll ===='
python add_broll.py

echo '==== 2/4 voix XTTS ===='
python build_audio_xtts.py script_qc.json boss_voice_ref.wav

echo '==== 3/4 sous-titres ===='
python gen_captions.py script_qc.json

echo '==== 4/4 rendu Remotion ===='
npx remotion render src/index.ts Tutorial out/yt_botnet.mp4 --concurrency=3 --log=error

echo '==== DONE ===='
ls -lh out/yt_botnet.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 out/yt_botnet.mp4
