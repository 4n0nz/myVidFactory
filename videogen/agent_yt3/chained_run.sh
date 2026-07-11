#!/bin/bash
echo 'Waiting for dE-I5xKtmso compositor...'
until grep -q 'DONE' /home/anon/videogen/agent_yt/compositor.log 2>/dev/null; do
  sleep 15
done

echo 'Waiting for _KzObeom88Y prep...'
until grep -q 'PREP DONE' /home/anon/videogen/agent_yt3/pipeline.log 2>/dev/null; do
  sleep 15
done

echo 'Building compositor script...'
python3 /home/anon/videogen/agent_yt3/build_compositor.py

echo 'Launching compositor...'
bash /home/anon/videogen/agent_yt3/run_compositor.sh

echo 'ALL DONE'
