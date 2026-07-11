#!/bin/bash
set -e
cd /home/anon/videogen/agent_yt3
exec > pipeline.log 2>&1

source /home/anon/videogen/.venv-xtts/bin/activate

echo '=== 1/4 DOWNLOAD ==='
rm -f source.mp4 source.webm source.mkv 2>/dev/null || true
python3 -m yt_dlp \n  -f 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]' \n  --merge-output-format mp4 \n  -o 'source.%(ext)s' \n  'https://www.youtube.com/watch?v=_KzObeom88Y'
ls -lh source.*

echo '=== 2/4 EXTRACT AUDIO ==='
ffmpeg -i source.mp4 -ar 16000 -ac 1 -c:a pcm_s16le source_audio.wav -y

echo '=== 3/4 WHISPER ==='
python3 /home/anon/videogen/agent_yt3/whisper_transcribe.py

echo '=== 4/4 HOST ANALYSIS (transcode + face detect) ==='
ffmpeg -i source.mp4 -vf scale=854:480 -c:v libx264 -preset ultrafast -an source_h264.mp4 -y
python3 /home/anon/videogen/agent_yt/analyze_host.py /home/anon/videogen/agent_yt3/source_h264.mp4

echo '==== PREP DONE ===='
