#!/usr/bin/env bash
# Test end-to-end scène d'action ShadowBroker : narration -> TTS -> cues whisper -> rendu synchro.
set -e
cd /home/anon/videogen
. .venv/bin/activate

# --- narration écrite par "l'API Claude" (ici à la main pour le test) ---
TXT="Ok, checke le toolkit de reconnaissance de ShadowBroker. Tu rentres une cible dans le champ, pis tu cliques sur Analyser. La fenêtre des résultats OSINT s ouvre direct : géolocalisation, WHOIS, pis sanctions. Clique sur le premier résultat pour sortir le dossier complet de l entité."

echo "==== 1/5 voix off (Thierry -16Hz +15%) ===="
edge-tts --voice fr-CA-ThierryNeural --rate=+15% --pitch=-16Hz --text "$TXT" --write-media public/demo_action.mp3
ffmpeg -y -i public/demo_action.mp3 -ar 16000 -ac 1 out/sb_action.wav 2>/dev/null

echo "==== 2/5 timing mot-par-mot (whisper) ===="
/home/anon/whisper.cpp/build/bin/whisper-cli -m /home/anon/whisper.cpp/models/ggml-small.bin \
  -f out/sb_action.wav -l fr --max-len 1 -osrt -of out/demo_words 2>/dev/null

echo "==== 3/5 extraction des cues ===="
python gen_cues.py out/demo_words.srt

echo "==== 4/5 rendu de la scène synchro ===="
npx remotion render src/index.ts ActionScene out/action_sb.mp4 --concurrency=3 --log=error 2>&1 | tail -1

echo "==== 5/5 ok ===="
ls -la out/action_sb.mp4
ffprobe -v error -show_entries format=duration -of csv=p=0 out/action_sb.mp4
