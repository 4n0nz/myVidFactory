import os, subprocess
os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "out", "voices")
os.makedirs(OUT, exist_ok=True)

MALE = [
 "Andrew Chipper","Badr Odhiambo","Dionisio Schuyler","Royston Min","Viktor Eka",
 "Abrahan Mack","Adde Michal","Baldur Sanjin","Craig Gutsy","Damien Black",
 "Gilberto Mathias","Ilkin Urbano","Kazuhiko Atallah","Ludvig Milivoj","Suad Qasim",
 "Torcull Diarmuid","Viktor Menelaos","Zacharie Aimilios","Ige Behringer","Filip Traverse",
 "Damjan Chapman","Wulf Carlevaro","Aaron Dreschner","Kumar Dahl","Eugenio Mataracı",
 "Ferran Simen","Xavier Hayasaka","Luis Moray","Marcos Rudaski",
]

TXT = ("ShadowBroker. Pis non, c'est pas le groupe de hackers. C'est une plateforme OSINT "
       "open-source qui met soixante sources de renseignements en temps réel sur une seule "
       "carte. Avions, navires, satellites, caméras de surveillance, tout en même temps. Checke ça.")

def mp3(src, dst):
    subprocess.run(["ffmpeg","-y","-i",src,"-ar","44100","-ac","1","-b:a","192k",dst], capture_output=True)

# silence 0.6s
sil = os.path.join(OUT, "_sil.mp3")
subprocess.run(["ffmpeg","-y","-f","lavfi","-i","anullsrc=r=44100:cl=mono","-t","0.6","-b:a","192k",sil], capture_output=True)

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")
parts = []
for i, sp in enumerate(MALE):
    wav = os.path.join(OUT, f"{i:02d}.wav")
    m = os.path.join(OUT, f"{i:02d}_{sp.replace(' ','_')}.mp3")
    if os.path.exists(m):
        print(f"[{i+1}/{len(MALE)}] {sp} (déjà)", flush=True); parts.append(m); continue
    print(f"[{i+1}/{len(MALE)}] {sp}", flush=True)
    try:
        tts.tts_to_file(text=f"Voix numéro {i+1}. {sp}. {TXT}", speaker=sp, language="fr", file_path=wav)
        mp3(wav, m); os.remove(wav)
        parts.append(m)
    except Exception as e:
        print(f"   ✗ {sp}: {e}", flush=True)

# concat avec silence entre chaque
lst = os.path.join(OUT, "list.txt")
with open(lst, "w") as f:
    for m in parts:
        f.write(f"file '{m}'\n"); f.write(f"file '{sil}'\n")
final = os.path.join(ROOT, "out", "voices_demo_male.mp3")
subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",lst,"-c","copy",final], capture_output=True)
print("FINAL", final, flush=True)
