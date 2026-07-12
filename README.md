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

```bash
# 1. cloner en 2 dossiers attendus par le code (chemins ~/yolo et ~/videogen)
git clone https://github.com/4n0nz/myVidFactory.git vf
ln -s "$PWD/vf/yolo"     ~/yolo
ln -s "$PWD/vf/videogen" ~/videogen

# 2. venv YOLO (détection)
python3.10 -m venv ~/yolo/.venv
~/yolo/.venv/bin/pip install -r ~/yolo/requirements.txt

# 3. venv videogen (compositor)
python3.10 -m venv ~/videogen/.venv
~/videogen/.venv/bin/pip install -r ~/videogen/requirements.txt
```

> ⚠️ **Chemins en dur** : le code référence `~/yolo`, `~/videogen` et quelques
> `/home/boss/videogen/...`. Layout attendu = un user dont `$HOME` contient `yolo/` et `videogen/`.
> Pour un autre user, adapter les `/home/boss/` (grep `/home/boss` dans `build_seg.py` et
> `yolo/scripts/yolo_extent.py`).

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
