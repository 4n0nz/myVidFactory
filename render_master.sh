#!/usr/bin/env bash
# Layered master render: base (visuals+voice) + terminal-PiP overlay (clean subs).
set -e
cd /home/anon/videogen
. .venv/bin/activate

echo "==== 1/4 clean captions ===="
python gen_captions.py script_qc.json

echo "==== 2/4 base render (visuals + voix) ===="
npx remotion render src/index.ts Tutorial out/master.mp4 --concurrency=3 --log=error

echo "==== 3/4 terminal-PiP (alpha, full length) ===="
npx remotion render src/index.ts SubsTerminal out/term_full.mov --codec=prores --prores-profile=4444 --log=error

echo "==== 4/4 overlay PiP bottom-right ===="
ffmpeg -y -i out/master.mp4 -i out/term_full.mov \
  -filter_complex '[0][1]overlay=W-w-48:H-h-48' \
  -c:a copy -c:v libx264 -preset medium -crf 20 out/master_final.mp4

echo "==== DONE ===="
ls -la out/master_final.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 out/master_final.mp4
