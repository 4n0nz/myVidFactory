#!/usr/bin/env bash
# Setup bare-metal de VideoFactory. Prerequis DEJA installes : GPU NVIDIA + drivers + CUDA.
# (l'install des drivers NVIDIA n'est PAS automatisee ici - trop risque/variable selon la distro.)
set -e
REPO="$(cd "$(dirname "$0")" && pwd)"

echo "== check GPU NVIDIA =="
command -v nvidia-smi >/dev/null && nvidia-smi -L || { echo "!! pas de nvidia-smi : installe les drivers NVIDIA + CUDA d'abord"; exit 1; }

echo "== deps systeme (apt) =="
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv ffmpeg
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q h264_nvenc && echo "ffmpeg NVENC OK" || echo "!! ffmpeg sans h264_nvenc - le render NVENC echouera"

echo "== chemins /home/boss -> \$HOME + layout ~/yolo ~/videogen =="
sed -i "s|/home/boss|$HOME|g" "$REPO/videogen/build_seg.py" "$REPO/videogen/run_one.sh" "$REPO/yolo/scripts/yolo_extent.py"
[ -e "$HOME/yolo" ]     || ln -s "$REPO/yolo"     "$HOME/yolo"
[ -e "$HOME/videogen" ] || ln -s "$REPO/videogen" "$HOME/videogen"

echo "== venv YOLO (detection, torch CUDA) =="
python3.10 -m venv "$HOME/yolo/.venv"
"$HOME/yolo/.venv/bin/pip" install -U pip
"$HOME/yolo/.venv/bin/pip" install --extra-index-url https://download.pytorch.org/whl/cu128 -r "$HOME/yolo/requirements.txt"

echo "== venv videogen (compositor) =="
python3.10 -m venv "$HOME/videogen/.venv"
"$HOME/videogen/.venv/bin/pip" install -U pip
"$HOME/videogen/.venv/bin/pip" install --extra-index-url https://download.pytorch.org/whl/cu128 -r "$HOME/videogen/requirements.txt"

echo ""
echo "OK. Lancer :  cd ~/videogen && bash run_one.sh wk_demo \"<url_youtube>\" demo.mp4"
