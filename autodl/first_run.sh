#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="tmm-chart"

if ! command -v conda >/dev/null 2>&1; then
  echo "[autodl] conda not found"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

export HF_HOME=/root/autodl-tmp/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface
export TORCH_HOME=/root/autodl-tmp/cache/torch
export TOKENIZERS_PARALLELISM=false

echo "[autodl] nvidia-smi"
nvidia-smi || true

echo "[autodl] python / torch check"
python - <<'PY'
import platform
import torch
print("python:", platform.python_version())
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

echo "[autodl] running smoke test"
cd "${ROOT_DIR}"
python scripts/run_smoke_test.py --config configs/base_experiment.yaml --limit 8

echo "[autodl] first run finished"
