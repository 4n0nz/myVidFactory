#!/usr/bin/env python3
"""Voix off via XTTS (local GPU) — speaker 'Damien Black'.
Charge le modèle UNE fois, génère toutes les scènes, écrit le manifest.
Tourne dans .venv-xtts. Remplace build_audio.py (edge-tts)."""
import json, os, subprocess, math, sys
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "script_qc.json")
SPEAKER = sys.argv[2] if len(sys.argv) > 2 else "Zacharie Aimilios"
AUDIO_DIR = os.path.join(ROOT, "public", "audio")
MANIFEST = os.path.join(ROOT, "render-manifest.json")
TAIL, MIN_SEC = 0.7, 3.0
os.makedirs(AUDIO_DIR, exist_ok=True)

data = json.load(open(SCRIPT, encoding="utf-8"))
meta = data["meta"]
fps = meta["fps"]

def duration(path):
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "csv=p=0", path])
    return float(out.strip())

print("[xtts] chargement du modèle sur GPU...", flush=True)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

out_scenes = []
for sc in data["scenes"]:
    sid = sc["id"]
    # scènes B-roll : pas de voix, durée fixe, host masqué
    if sc["type"] == "broll" or not sc.get("narration", "").strip():
        secs = sc.get("props", {}).get("durationSec", 3)
        out_scenes.append({
            "id": sid, "type": sc["type"], "host": sc.get("host", "off"),
            "props": sc["props"], "audioFile": None,
            "durationInFrames": math.ceil(secs * fps),
        })
        print(f"[broll] {sid} -> {math.ceil(secs*fps)} frames", flush=True)
        continue
    wav = os.path.join(AUDIO_DIR, f"{sid}.wav")
    mp3 = os.path.join(AUDIO_DIR, f"{sid}.mp3")
    print(f"[tts] {sid} ...", flush=True)
    tts.tts_to_file(text=sc["narration"], speaker=SPEAKER, language="fr", file_path=wav)
    subprocess.run(["ffmpeg", "-y", "-i", wav, "-b:a", "192k", mp3], capture_output=True)
    os.remove(wav)
    dur = max(duration(mp3) + TAIL, MIN_SEC)
    frames = math.ceil(dur * fps)
    out_scenes.append({
        "id": sid, "type": sc["type"],
        "host": sc.get("host", sc.get("props", {}).get("host", "pip")),
        "props": sc["props"], "audioFile": f"audio/{sid}.mp3",
        "durationInFrames": frames,
    })
    print(f"      {dur:.2f}s -> {frames} frames", flush=True)

meta["voice"] = f"xtts:{SPEAKER}"
json.dump({"meta": meta, "scenes": out_scenes}, open(MANIFEST, "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
total = sum(s["durationInFrames"] for s in out_scenes)
print(f"\n[done] {len(out_scenes)} scenes, {total} frames = {total/fps:.1f}s", flush=True)
