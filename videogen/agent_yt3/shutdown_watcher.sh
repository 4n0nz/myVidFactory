# NOTE: pw sudo retire pour repo public. Sur la box: configurer 'sudo NOPASSWD' ou passer via env.
#!/bin/bash
until grep -q '==== DONE ====' /home/anon/videogen/agent_yt3/compositor.log 2>/dev/null; do sleep 20; done
echo "$(date) - compositor done, shutting down..." >> /home/anon/videogen/agent_yt3/watcher.log
sudo shutdown -h now
