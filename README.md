# VideoFactory

Pipeline qui remplace le narrateur d'une vidéo (screen-record + webcam) par un avatar,
en gardant le contenu écran et l'audio source.

## Structure

- **`videogen/`** — compositor & orchestration
  - `run_one.sh` — pipeline complet : download → détection host → extent YOLO → build → render (NVENC segmenté)
  - `build_seg.py` — compositor segmenté (overlay avatar dans la webcam pip, coins arrondis)
  - `agent_yt/analyze_host2.py` — classification hero/pip/off par segment (YuNet + live-motion)
- **`yolo/`** — détection facecam
  - `scripts/yolo_extent.py` — box webcam précise par segment (YOLO facecam → YuNet fallback → coin prudent),
    split intro, filtre densité anti-box-gonflée, raffinement de frontière sur hard-cut
  - `scripts/` — dataset synthétique, train, validation (score sur vraies frames)

Modèles, venvs et médias sont hors repo (voir `.gitignore` de chaque sous-dossier).
