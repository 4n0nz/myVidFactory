# VideoFactory

Pipeline qui **remplace le narrateur** d'une vidéo (screen-record + webcam) par un **avatar**,
en gardant le contenu écran et l'audio source. Détecte la webcam par segment (YOLO facecam +
fallbacks), remplace le visage par l'avatar avec coins arrondis, rend en NVENC.

## Structure (monorepo, 2 sous-projets)

Le pipeline = 2 briques qui tournent ensemble :

```
videogen/          # orchestration + compositor
  run_one.sh         # ENTRÉE : download → détection → extent YOLO → build → render
  build_seg.py       # compositor segmenté (overlay avatar dans la webcam pip)
  agent_yt/          # analyze_host2.py (classification hero/pip/off) + scripts pipeline
  agent_yt3/         # variantes/utilitaires pipeline
  vf_qc.py           # QC couverture
  public/avatar.mp4  # ← l'avatar de remplacement (inclus)
  face_detection_yunet_2023mar.onnx  # ← détecteur YuNet (inclus)
  requirements.txt

yolo/              # détection facecam
  scripts/yolo_extent.py   # box webcam précise/segment (YOLO → YuNet → coin)
                           # + split intro, filtre densité, raffinement frontière
  scripts/                 # dataset synthétique, train, validation, score_real.py
  runs/facecam5/weights/best_real.pt  # ← MODÈLE PROD epoch8 (inclus)
  runs/facecam/weights/best.pt        # ← modèle v3 rollback (inclus)
  requirements.txt
```

Modèles, avatar et onnx sont **dans le repo** (pas de download externe). Seuls les venvs et les
workdirs temporaires `wk_*` (source + rendus par vidéo) sont hors repo.

## Prérequis système

- **Linux** + **GPU NVIDIA** (testé RTX 5060 Ti 16 Go) avec drivers CUDA
- **ffmpeg** compilé avec **NVENC** (`h264_nvenc`) — le render l'utilise
- **Python 3.10**
- Le modèle YOLO tourne sur CUDA (torch cu12x)

## Installation

Trois façons, du plus simple au plus manuel.

### A. Docker (recommandé — machine neuve, zéro install manuelle)

Prérequis hôte : **Docker** + **nvidia-container-toolkit** (pour passer le GPU au container).
Rien d'autre à installer (CUDA/ffmpeg/python sont dans l'image).

```bash
git clone https://github.com/4n0nz/myVidFactory.git && cd myVidFactory
docker build -t videofactory .
docker run --gpus all -v "$PWD/out:/root/videogen/out" videofactory wk_demo "<url_youtube>" demo.mp4
# résultat -> ./out/demo.mp4
```

`--gpus all` expose le GPU **et NVENC** de l'hôte au container (via `NVIDIA_DRIVER_CAPABILITIES`).

### B. setup.sh (bare-metal — la machine a déjà GPU NVIDIA + drivers CUDA)

```bash
git clone https://github.com/4n0nz/myVidFactory.git && cd myVidFactory
bash setup.sh          # apt: ffmpeg/python + venvs + pip install + symlinks + adapte les chemins
cd ~/videogen && bash run_one.sh wk_demo "<url_youtube>" demo.mp4
```

`setup.sh` **n'installe PAS** les drivers NVIDIA/CUDA (trop variable selon la distro) — il vérifie
juste que `nvidia-smi` répond. Installe CUDA avant si besoin.

### C. Manuel

```bash
git clone https://github.com/4n0nz/myVidFactory.git vf
ln -s "$PWD/vf/yolo" ~/yolo ; ln -s "$PWD/vf/videogen" ~/videogen
# adapter les chemins en dur si le user n'est pas 'boss' :
sed -i "s|/home/boss|$HOME|g" ~/videogen/build_seg.py ~/videogen/run_one.sh ~/yolo/scripts/yolo_extent.py
python3.10 -m venv ~/yolo/.venv     && ~/yolo/.venv/bin/pip     install --extra-index-url https://download.pytorch.org/whl/cu128 -r ~/yolo/requirements.txt
python3.10 -m venv ~/videogen/.venv && ~/videogen/.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r ~/videogen/requirements.txt
```

## Lancer le pipeline

```bash
cd ~/videogen
bash run_one.sh <workdir> <url_youtube> <sortie.mp4>

# exemple
bash run_one.sh wk_demo "https://www.youtube.com/watch?v=XXXX" demo.mp4
```

- `<sortie.mp4>` : **l'extension `.mp4` est obligatoire** (sinon le muxer ffmpeg plante à la fin).
- Étapes : download (yt-dlp) → `analyze_host2.py` (hero/pip/off) → `yolo_extent.py` (box webcam
  précise, modèle epoch8 par défaut) → `build_seg.py` (overlay avatar) → render NVENC.
- Résultat dans `~/videogen/out/<sortie.mp4>`.

## Modèle

`yolo_extent.py` utilise **`runs/facecam5/weights/best_real.pt`** (epoch8) par défaut.
Rollback vers v3 :

```bash
YOLO_WEIGHTS=~/yolo/runs/facecam/weights/best.pt bash run_one.sh ...
```

## Notes

- Webcam mobile / multi-position : gérée par segment (détection + raffinement de frontière sur hard-cut).
- Chantier ouvert : webcam **grande sans bordure** (plein côté d'écran) encore sous-couverte (box = visage).
- Audio : piste source copiée telle quelle (tag `und`).
