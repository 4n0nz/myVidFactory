import os, sys
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS

text = sys.argv[1]
out = sys.argv[2]
ref = sys.argv[3] if len(sys.argv) > 3 else None   # clip de référence (clone) optionnel
spk = sys.argv[4] if len(sys.argv) > 4 else "Damien Black"

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
kw = dict(text=text, language="fr", file_path=out)
if ref:
    kw["speaker_wav"] = ref
else:
    kw["speaker"] = spk
tts.tts_to_file(**kw)
print("DONE", out)
