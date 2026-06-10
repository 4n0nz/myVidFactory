#!/usr/bin/env bash
# myVidFactory — installation (Ubuntu/Debian x86_64).
# Usage: ./install.sh [--whisper]
set -e

WHISPER=0
[ "$1" = "--whisper" ] && WHISPER=1

echo "==> Dépendances système (Chrome headless + ffmpeg)"
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update -qq
$SUDO apt-get install -y -q \
  ffmpeg \
  libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 \
  libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2t64 \
  libpango-1.0-0 libcairo2 fonts-liberation \
  python3 python3-venv

echo "==> Environnement Python + edge-tts (voix off)"
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q edge-tts

echo "==> Dépendances Node + navigateur Remotion"
npm install --no-audit --no-fund
npx remotion browser ensure

if [ "$WHISPER" -eq 1 ]; then
  echo "==> whisper.cpp (sous-titres par transcription, optionnel)"
  $SUDO apt-get install -y -q build-essential cmake
  if [ ! -d whisper.cpp ]; then
    git clone --depth 1 https://github.com/ggerganov/whisper.cpp
  fi
  cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build whisper.cpp/build -j"$(nproc)"
  bash whisper.cpp/models/download-ggml-model.sh small
fi

echo ""
echo "✓ Installation terminée."
echo "  Génère une vidéo :  bash render_master.sh"
echo "  (édite d'abord script_qc.json, ou crée ton propre script.json)"
