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

## Assets inclus

Modèles YOLO (`yolo/runs/.../*.pt`), YuNet (`videogen/*.onnx`) et avatar (`videogen/public/avatar.mp4`)
sont dans le repo. Deps Python : `requirements.txt` dans chaque sous-dossier (recréer un venv + `pip install -r`).
Seuls les venvs et les workdirs `wk_*` (temp par vidéo) restent hors repo.

## Modèle par défaut

`yolo/scripts/yolo_extent.py` utilise `runs/facecam5/weights/best_real.pt` (epoch8).
Rollback v3 : `YOLO_WEIGHTS=.../runs/facecam/weights/best.pt`.
