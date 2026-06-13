#!/usr/bin/env python3
"""Extrait le timing des mots-clés (cues) depuis un SRT whisper --max-len 1.
Fusionne les tokens en mots, puis cherche les phrases-déclencheuses.
Sortie: cues.json = {action_id: time_seconds}."""
import json, re, sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
srt = sys.argv[1] if len(sys.argv) > 1 else "out/demo_words.srt"

def to_sec(ts):
    h, m, rest = ts.split(":"); s, ms = rest.split(",")
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

# parse tokens
txt = open(os.path.join(ROOT, srt), encoding="utf-8").read()
toks = []
for blk in re.split(r"\n\s*\n", txt.strip()):
    lines = [l for l in blk.splitlines() if l.strip()]
    if len(lines) < 2: continue
    m = re.search(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->", lines[1])
    if not m: continue
    raw = " ".join(lines[2:]) if len(lines) > 2 else ""
    toks.append((to_sec(m.group(1)), raw))

# fusionne tokens -> mots (un token qui commence par espace = nouveau mot)
words = []  # (start, word)
for start, raw in toks:
    piece = raw.strip()
    if not piece or piece in ",.'":  # ponctuation seule
        continue
    if raw.startswith(" ") or not words:
        words.append([start, piece])
    else:
        words[-1][1] += piece

# nettoie ponctuation collée
for w in words:
    w[1] = re.sub(r"[,.;:!?']", "", w[1]).lower()

def find(phrase, after=0.0):
    """temps du 1er mot de `phrase` apparaissant après `after`."""
    target = phrase.split()[0].lower()
    for start, w in words:
        if start >= after and target in w:
            return round(start, 2)
    return None

# --- cues du démo (phrase déclencheuse -> action) ---
cues = {}
cues["startClick"] = find("analyser") or find("démarrer")
cues["openWindow"] = find("fenêtre")
t2 = cues["openWindow"] or 0
cues["resultClick"] = find("premier", after=t2)

# durée de l'audio -> pour caler la durée de la scène
import subprocess
mp3 = os.path.join(ROOT, "public", "demo_action.mp3")
try:
    out = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "csv=p=0", mp3])
    cues["audioDur"] = round(float(out.strip()), 2)
except Exception:
    cues["audioDur"] = None

json.dump(cues, open(os.path.join(ROOT, "cues.json"), "w"), indent=2)
print("mots fusionnés:", " ".join(w for _, w in words))
print("cues:", json.dumps(cues))
