#!/bin/bash
# fix2.sh — telecharge SDXL par l API Python.
# huggingface_hub 1.x a RENOMME la CLI (`huggingface-cli` -> `hf`) : l ancienne commande
# se contente d afficher son aide, et le script d avant croyait avoir telecharge.
# snapshot_download ne depend d aucun nom de commande.
# On prend la variante fp16 (6,9 Go, tient entiere en VRAM) et pas le fp32 (13,9 Go).
set -e
BASE=$HOME/avatar_gen
$BASE/.venv-sd/bin/pip install -q torchvision --index-url https://download.pytorch.org/whl/cu128

$BASE/.venv-sd/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download(
    'stabilityai/stable-diffusion-xl-base-1.0',
    local_dir='/home/boss/avatar_gen/models/sdxl',
    allow_patterns=['**/*.fp16.safetensors', '**/*.json', '**/*.txt', '*.json'],
    max_workers=8,
)
print('telecharge dans', p)
EOF

echo "--- contenu ---"
du -sh $BASE/models/sdxl
find $BASE/models/sdxl -name "*.safetensors" -printf "%-58p %10s\n" | sed "s|$BASE/models/sdxl/||"

echo "--- chargement reel du pipeline (la vraie preuve) ---"
$BASE/.venv-sd/bin/python - <<'EOF'
import torch
from diffusers import StableDiffusionXLPipeline
p = StableDiffusionXLPipeline.from_pretrained(
    '/home/boss/avatar_gen/models/sdxl', torch_dtype=torch.float16,
    variant='fp16', use_safetensors=True).to('cuda')
print('pipeline charge sur', p.device, '| VRAM %.1f Go' % (torch.cuda.memory_allocated()/1e9))
EOF
echo "===== FIX2 OK $(date '+%F %T') ====="
