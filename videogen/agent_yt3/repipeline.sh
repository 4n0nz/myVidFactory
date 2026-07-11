#!/bin/bash
set -e
exec > /home/anon/videogen/agent_yt3/repipeline.log 2>&1
source /home/anon/videogen/.venv-xtts/bin/activate
echo '=== 1/3 ANALYSIS ==='
python3 /home/anon/videogen/agent_yt/analyze_host.py /home/anon/videogen/agent_yt3/source_h264.mp4
echo '=== 2/3 BUILD ==='
python3 /home/anon/videogen/agent_yt/build_compositor.py /home/anon/videogen/agent_yt3 KzObeom88Y_remix.mp4
echo '=== 3/3 COMPOSITOR ==='
bash /home/anon/videogen/agent_yt3/run_compositor.sh
echo '==== PIPELINE DONE ===='
