#!/usr/bin/env python3
"""Generate TTS voiceover per scene and build the Remotion render manifest."""
import json, os, subprocess, math, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(ROOT, sys.argv[1] if len(sys.argv) > 1 else "script.json")
AUDIO_DIR = os.path.join(ROOT, "public", "audio")
MANIFEST = os.path.join(ROOT, "render-manifest.json")
TAIL_SEC = 0.7          # silence padding after each narration
MIN_SEC = 3.0           # floor per scene

os.makedirs(AUDIO_DIR, exist_ok=True)

with open(SCRIPT, encoding="utf-8") as f:
    data = json.load(f)

meta = data["meta"]
fps = meta["fps"]
voice = sys.argv[2] if len(sys.argv) > 2 else meta["voice"]
meta["voice"] = voice  # reflect override in manifest
rate = meta.get("rate", "+0%")
pitch = meta.get("pitch", "+0Hz")

def duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", path
    ])
    return float(out.strip())

out_scenes = []
for sc in data["scenes"]:
    sid = sc["id"]
    mp3 = os.path.join(AUDIO_DIR, f"{sid}.mp3")
    print(f"[tts] {sid} ...", flush=True)
    subprocess.run([
        "edge-tts", "--voice", voice, f"--rate={rate}", f"--pitch={pitch}",
        "--text", sc["narration"], "--write-media", mp3
    ], check=True)
    dur = max(duration(mp3) + TAIL_SEC, MIN_SEC)
    frames = math.ceil(dur * fps)
    out_scenes.append({
        "id": sid,
        "type": sc["type"],
        "host": sc.get("host", sc.get("props", {}).get("host", "pip")),
        "props": sc["props"],
        "audioFile": f"audio/{sid}.mp3",
        "durationInFrames": frames,
    })
    print(f"      {dur:.2f}s -> {frames} frames", flush=True)

manifest = {"meta": meta, "scenes": out_scenes}
with open(MANIFEST, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

total = sum(s["durationInFrames"] for s in out_scenes)
print(f"\n[done] {len(out_scenes)} scenes, {total} frames = {total/fps:.1f}s", flush=True)
