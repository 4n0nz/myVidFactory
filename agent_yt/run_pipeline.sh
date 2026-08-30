#!/bin/bash
set -e
cd /home/anon/videogen/agent_yt
exec > pipeline.log 2>&1

source /home/anon/videogen/.venv-xtts/bin/activate

echo '=== 1/3 DOWNLOAD ==='
rm -f source.mp4 source.webm source.mkv 2>/dev/null || true

python3 -m yt_dlp \n  -f 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]' \n  --merge-output-format mp4 \n  -o 'source.%(ext)s' \n  'https://www.youtube.com/watch?v=dE-I5xKtmso'

# Ensure file is named source.mp4
for f in source.webm source.mkv; do [ -f $f ] && ffmpeg -i $f -c copy source.mp4 -y && rm $f; done
ls -lh source.* 2>/dev/null

echo ''
echo '=== 2/3 EXTRACT AUDIO ==='
ffmpeg -i source.mp4 -ar 16000 -ac 1 -c:a pcm_s16le source_audio.wav -y

echo ''
echo '=== 3/3 WHISPER ==='
python3 /home/anon/videogen/agent_yt/whisper_transcribe.py

echo ''
echo '==== TRANSCRIPT DONE ===='
