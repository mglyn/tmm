#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/data"
LEGACY_ROOT="${REPO_ROOT}/legacy/chart_vqa_synthesis_1"
OUT_ROOT="${REPO_ROOT}/router_runs/e03_beststage_multiseed"
SEEDS=(42 3407 2025)
PORT_BASE=8100
BASE_MODEL="/root/autodl-tmp/data/models/Qwen2.5-VL-7B-Instruct"
LLAMA_FACTORY_SRC="${LEGACY_ROOT}/LLaMA-Factory-main/src"
LLAMA_FACTORY_DATA="${LEGACY_ROOT}/outputs/llamafactory_data"
TMP_DATA_DIR="${OUT_ROOT}/llamafactory_data_task2_jsonl"
STAGE1_CHECKPOINT="${LEGACY_ROOT}/outputs/models/stage1_description"

export PYTHONPATH="${LLAMA_FACTORY_SRC}:${PYTHONPATH:-}"

mkdir -p "${OUT_ROOT}"

python - <<'PY'
import json
from pathlib import Path

src_root = Path("/root/autodl-tmp/data/legacy/chart_vqa_synthesis_1/outputs/llamafactory_data")
dst_root = Path("/root/autodl-tmp/data/router_runs/e03_beststage_multiseed/llamafactory_data_task2_jsonl")
dst_root.mkdir(parents=True, exist_ok=True)

dataset_info = {
    "task2_basic_vqa_train": {
        "file_name": "task2_basic_vqa/train.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    },
    "task2_basic_vqa_val": {
        "file_name": "task2_basic_vqa/val.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    },
}

(dst_root / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

for split in ("train", "val"):
    src_file = src_root / "task2_basic_vqa" / f"{split}.json"
    dst_dir = dst_root / "task2_basic_vqa"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_file = dst_dir / f"{split}.jsonl"
    rows = json.loads(src_file.read_text(encoding="utf-8"))
    with dst_file.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
PY

wait_for_api() {
  local api_base="$1"
  python - "$api_base" <<'PY'
import sys
import time
import requests

api_base = sys.argv[1]
deadline = time.time() + 300
last_error = None

while time.time() < deadline:
    try:
        response = requests.get(f"{api_base}/v1/models", timeout=5)
        if response.status_code == 200:
            sys.exit(0)
        last_error = f"status={response.status_code}"
    except Exception as exc:  # pragma: no cover
        last_error = str(exc)
    time.sleep(5)

raise SystemExit(f"API did not become ready: {last_error}")
PY
}

stop_api() {
  local pid_file="$1"
  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi
  python - "$pid_file" <<'PY'
import json
import os
import signal
import sys
import time
from pathlib import Path

pid_file = Path(sys.argv[1])
payload = json.loads(pid_file.read_text(encoding="utf-8"))
pid = int(payload["pid"])

try:
    os.killpg(pid, signal.SIGTERM)
except ProcessLookupError:
    pid_file.unlink(missing_ok=True)
    raise SystemExit(0)

for _ in range(30):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        raise SystemExit(0)
    time.sleep(1)

try:
    os.killpg(pid, signal.SIGKILL)
except ProcessLookupError:
    pass

pid_file.unlink(missing_ok=True)
PY
}

echo "Running E03 best single-stage multi-seed with seeds: ${SEEDS[*]}"

for idx in "${!SEEDS[@]}"; do
  seed="${SEEDS[$idx]}"
  port="$((PORT_BASE + idx))"
  run_dir="${OUT_ROOT}/seed_${seed}"
  model_dir="${run_dir}/stage2_basic_vqa"
  eval_dir="${run_dir}/eval_chartqa"
  api_log="${run_dir}/stage2_api.log"
  pid_file="${run_dir}/stage2_api.pid.json"

  mkdir -p "${run_dir}"

  echo "==== Seed ${seed}: train stage2 ===="
  (
    cd "${LEGACY_ROOT}"
    llamafactory-cli train \
      --stage sft \
      --do_train True \
      --model_name_or_path "${BASE_MODEL}" \
      --preprocessing_num_workers 16 \
      --finetuning_type lora \
      --template qwen2_vl \
      --flash_attn auto \
      --dataset_dir "${TMP_DATA_DIR}" \
      --media_dir "${LEGACY_ROOT}" \
      --dataset task2_basic_vqa_train \
      --cutoff_len 2048 \
      --learning_rate 5e-05 \
      --num_train_epochs 10 \
      --max_samples 100000 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --lr_scheduler_type cosine \
      --max_grad_norm 1.0 \
      --logging_steps 5 \
      --save_steps 200 \
      --warmup_steps 100 \
      --packing False \
      --report_to none \
      --output_dir "${model_dir}" \
      --bf16 True \
      --plot_loss True \
      --trust_remote_code True \
      --ddp_timeout 180000000 \
      --optim adamw_torch \
      --lora_rank 8 \
      --lora_alpha 16 \
      --lora_dropout 0 \
      --lora_target all \
      --freeze_vision_tower True \
      --freeze_multi_modal_projector True \
      --image_max_pixels 589824 \
      --image_min_pixels 1024 \
      --resume_from_checkpoint "${STAGE1_CHECKPOINT}" \
      --eval_strategy epoch \
      --eval_dataset task2_basic_vqa_val \
      --per_device_eval_batch_size 4 \
      --save_total_limit 3 \
      --overwrite_output_dir True \
      --seed "${seed}"
  )

  echo "==== Seed ${seed}: start stage2 API on port ${port} ===="
  python "${REPO_ROOT}/scripts/start_stage_api_bg.py" \
    --port "${port}" \
    --adapter_path "${model_dir}" \
    --log_file "${api_log}" \
    --pid_file "${pid_file}" \
    --model_name_or_path "${BASE_MODEL}"

  wait_for_api "http://127.0.0.1:${port}"

  echo "==== Seed ${seed}: eval ChartQA ===="
  (
    cd "${LEGACY_ROOT}"
    python evaluate_chartqa.py \
      --api_base "http://127.0.0.1:${port}" \
      --model_name "default" \
      --split test \
      --output_dir "${eval_dir}"
  )

  echo "==== Seed ${seed}: stop API ===="
  stop_api "${pid_file}"
done

echo "==== Aggregate multi-seed summary ===="
python - <<'PY'
import csv
import json
import math
from pathlib import Path

out_root = Path("/root/autodl-tmp/data/router_runs/e03_beststage_multiseed")
seed_dirs = sorted([path for path in out_root.glob("seed_*") if path.is_dir()])

rows = []
accuracies = []


def extract_valid_train_summary(seed_dir: Path) -> dict:
    trainer_state = json.loads((seed_dir / "stage2_basic_vqa" / "trainer_state.json").read_text(encoding="utf-8"))
    candidates = []
    for entry in trainer_state.get("log_history", []):
        if "train_loss" not in entry:
            continue
        runtime = float(entry.get("train_runtime", 0.0) or 0.0)
        steps = int(entry.get("step", 0) or 0)
        candidates.append((runtime, steps, entry))

    if not candidates:
        raise ValueError(f"No valid train summary found in {seed_dir / 'stage2_basic_vqa' / 'trainer_state.json'}")

    # Prefer the real completed training summary instead of the spurious zero-runtime record.
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[-1][2]

for seed_dir in seed_dirs:
    seed = int(seed_dir.name.split("_")[-1])
    train_results = extract_valid_train_summary(seed_dir)
    eval_payload = json.loads((seed_dir / "eval_chartqa" / "chartqa_test_default.json").read_text(encoding="utf-8"))
    metrics = eval_payload["metrics"]
    row = {
        "seed": seed,
        "train_loss": train_results.get("train_loss"),
        "train_runtime_seconds": round(float(train_results.get("train_runtime", 0.0) or 0.0), 4),
        "chartqa_accuracy_percent": round(float(metrics["accuracy"]), 4),
        "chartqa_total": int(metrics["total"]),
        "chartqa_correct": int(metrics["correct"]),
    }
    rows.append(row)
    accuracies.append(row["chartqa_accuracy_percent"])

mean_acc = sum(accuracies) / max(len(accuracies), 1)
std_acc = math.sqrt(sum((value - mean_acc) ** 2 for value in accuracies) / max(len(accuracies), 1))

summary = {
    "seeds": [row["seed"] for row in rows],
    "num_seeds": len(rows),
    "chartqa_accuracy_percent_mean": round(mean_acc, 4),
    "chartqa_accuracy_percent_std": round(std_acc, 4),
    "chartqa_accuracy_mean_std": f"{mean_acc:.2f} +/- {std_acc:.2f}",
    "runs": rows,
    "config": {
        "model_name_or_path": "/root/autodl-tmp/data/models/Qwen2.5-VL-7B-Instruct",
        "stage": "stage2_basic_vqa",
        "resume_from_checkpoint": "/root/autodl-tmp/data/legacy/chart_vqa_synthesis_1/outputs/models/stage1_description",
        "eval_split": "test",
    },
}

(out_root / "multiseed_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

with (out_root / "multiseed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
