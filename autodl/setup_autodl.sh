#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="tmm-chart"

echo "[autodl] root dir: ${ROOT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[autodl] conda not found. Please choose an AutoDL image with conda/miniconda preinstalled."
  exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[autodl] creating conda env ${ENV_NAME}"
  conda env create -f "${ROOT_DIR}/autodl/environment.yml"
else
  echo "[autodl] conda env ${ENV_NAME} already exists"
fi

conda activate "${ENV_NAME}"

mkdir -p /root/autodl-tmp/cache/huggingface
mkdir -p /root/autodl-tmp/cache/torch
mkdir -p "${ROOT_DIR}/outputs"

export HF_HOME=/root/autodl-tmp/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/cache/huggingface
export TORCH_HOME=/root/autodl-tmp/cache/torch
export TOKENIZERS_PARALLELISM=false

echo "[autodl] installing python dependencies"
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install -e "${ROOT_DIR}"

echo "[autodl] checking torch/cuda"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

echo "[autodl] setup complete"
echo "[autodl] activate later with:"
echo "  source /root/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
