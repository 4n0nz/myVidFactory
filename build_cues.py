#!/usr/bin/env python3
"""Pour chaque scène 'action' du manifest : whisper word-timing sur son audio,
trouve les mots-déclencheurs (props.triggers) et écrit props.cues."""
import json, os, re, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
WH = "/home/anon/whisper.cpp/build/bin/whisper-cli"
MODEL = "/home/anon/whisper.cpp/models/ggml-small.bin"
MAN = os.path.join(ROOT, "render-manifest.json")

def words_from_audio(mp3):
    wav = mp3.rsplit(".", 1)[0] + ".16k.wav"
    subprocess.run(["ffmpeg", "-y", "-i", mp3, "-ar", "16000", "-ac", "1", wav],
                   capture_output=True)
    srt = mp3.rsplit(".", 1)[0] + ".words"
    subprocess.run([WH, "-m", MODEL, "-f", wav, "-l", "fr", "--max-len", "1",
                    "-osrt", "-of", srt], capture_output=True)
    txt = open(srt + ".srt", encoding="utf-8").read()
    def to_sec(ts):
        h, m, rest = ts.split(":"); s, ms = rest.split(",")
        return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000
    words = []
    for blk in re.split(r"\n\s*\n", txt.strip()):
        lines = [l for l in blk.splitlines() if l.strip()]
        if len(lines) < 2: continue
        m = re.search(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->", lines[1])
        if not m: continue
        raw = " ".join(lines[2:]) if len(lines) > 2 else ""
        piece = raw.strip()
        if not piece or piece in ",.'": continue
        if raw.startswith(" ") or not words:
            words.append([to_sec(m.group(1)), piece])
        else:
            words[-1][1] += piece
    for w in words:
        w[1] = re.sub(r"[,.;:!?']", "", w[1]).lower()
    return words

def find(words, phrase, after=0.0):
    if not phrase: return None
    target = phrase.split()[0].lower()
    for start, w in words:
        if start >= after and target in w:
            return round(start, 2)
    return None

man = json.load(open(MAN, encoding="utf-8"))
n = 0
for sc in man["scenes"]:
    if sc["type"] != "action":
        continue
    mp3 = os.path.join(ROOT, "public", sc["audioFile"])
    words = words_from_audio(mp3)
    tr = sc["props"].get("triggers", {"start": "analyser", "open": "fenêtre", "result": "premier"})
    start = find(words, tr.get("start"))
    openw = find(words, tr.get("open"), after=(start or 0))
    res = find(words, tr.get("result"), after=(openw or 0))
    dur = sc["durationInFrames"] / man["meta"]["fps"]
    sc["props"]["cues"] = {
        "startClick": start if start is not None else dur * 0.2,
        "openWindow": openw if openw is not None else dur * 0.4,
        "resultClick": res if res is not None else dur * 0.75,
    }
    print(f"[cues] {sc['id']}: {sc['props']['cues']}")
    n += 1

json.dump(man, open(MAN, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[done] {n} scène(s) action traitée(s)")
