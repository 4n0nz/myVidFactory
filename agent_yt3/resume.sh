#!/bin/bash
set -e
cd /home/anon/videogen/agent_yt3
exec >> pipeline.log 2>&1

source /home/anon/videogen/.venv-xtts/bin/activate

echo '=== 2/4 EXTRACT AUDIO ==='
ffmpeg -i source.mp4 -ar 16000 -ac 1 -c:a pcm_s16le source_audio.wav -y

echo '=== 3/4 WHISPER ==='
python3 /home/anon/videogen/agent_yt3/whisper_transcribe.py

echo '=== 4/4 HOST ANALYSIS ==='
ffmpeg -i source.mp4 -vf scale=854:480 -c:v libx264 -preset ultrafast -an source_h264.mp4 -y
python3 /home/anon/videogen/agent_yt/analyze_host.py /home/anon/videogen/agent_yt3/source_h264.mp4

echo '==== PREP DONE ===='
