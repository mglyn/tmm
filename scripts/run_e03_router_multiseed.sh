#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/autodl-tmp/data"
TRAIN_FEATURES="${REPO_ROOT}/router_runs/formal_sar/features/train_full.pt"
VAL_FEATURES="${REPO_ROOT}/router_runs/formal_sar/features/val_full.pt"
OUT_ROOT="${REPO_ROOT}/router_runs/e03_router_multiseed"
SEEDS=(42 3407 2025)
TRAIN_SCRIPT="${REPO_ROOT}/scripts/train_router_multimodal.py"
EVAL_SCRIPT="${REPO_ROOT}/scripts/eval_router_chartqa.py"

mkdir -p "${OUT_ROOT}"

echo "Running E03 router multi-seed with seeds: ${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  run_dir="${OUT_ROOT}/seed_${seed}"
  ckpt_dir="${run_dir}/checkpoints"
  eval_dir="${run_dir}/eval_chartqa"

  echo "==== Seed ${seed}: train ===="
  python "${TRAIN_SCRIPT}" \
    --train_features "${TRAIN_FEATURES}" \
    --val_features "${VAL_FEATURES}" \
    --output_dir "${ckpt_dir}" \
    --epochs 40 \
    --batch_size 512 \
    --hidden_dim 1024 \
    --num_hidden_layers 1 \
    --dropout 0.1 \
    --lr 0.0005 \
    --weight_decay 0.02 \
    --label_smoothing 0.05 \
    --class_weight_mode none \
    --selection_metric accuracy \
    --seed "${seed}" \
    --device cuda

  echo "==== Seed ${seed}: eval ===="
  python "${EVAL_SCRIPT}" \
    --router_checkpoint "${ckpt_dir}/best.pt" \
    --output_dir "${eval_dir}" \
    --batch_size 8 \
    --log_every 100 \
    --routing_policy stage2_fallback \
    --stage2_abs_threshold 0.20
done

echo "==== Aggregate multi-seed summary ===="
python - <<'PY'
import csv
import json
import math
from pathlib import Path

out_root = Path("/root/autodl-tmp/data/router_runs/e03_router_multiseed")
seed_dirs = sorted([path for path in out_root.glob("seed_*") if path.is_dir()])

rows = []
accuracies = []
elapsed = []

for seed_dir in seed_dirs:
    seed = int(seed_dir.name.split("_")[-1])
    train_metrics = json.loads((seed_dir / "checkpoints" / "best_metrics.json").read_text(encoding="utf-8"))
    eval_summary = json.loads((seed_dir / "eval_chartqa" / "summary.json").read_text(encoding="utf-8"))
    row = {
        "seed": seed,
        "router_val_accuracy": train_metrics["best_val_metrics"]["accuracy"],
        "router_val_macro_f1": train_metrics["best_val_metrics"]["macro_f1"],
        "chartqa_accuracy": eval_summary["accuracy"],
        "chartqa_accuracy_percent": eval_summary["accuracy_percent"],
        "elapsed_seconds": eval_summary["elapsed_seconds"],
    }
    rows.append(row)
    accuracies.append(row["chartqa_accuracy_percent"])
    elapsed.append(row["elapsed_seconds"])

mean_acc = sum(accuracies) / max(len(accuracies), 1)
std_acc = math.sqrt(sum((value - mean_acc) ** 2 for value in accuracies) / max(len(accuracies), 1))
mean_elapsed = sum(elapsed) / max(len(elapsed), 1)

summary = {
    "seeds": [row["seed"] for row in rows],
    "num_seeds": len(rows),
    "chartqa_accuracy_percent_mean": round(mean_acc, 4),
    "chartqa_accuracy_percent_std": round(std_acc, 4),
    "chartqa_accuracy_mean_std": f"{mean_acc:.2f} +/- {std_acc:.2f}",
    "mean_elapsed_seconds": round(mean_elapsed, 2),
    "runs": rows,
    "config": {
        "train_features": "/root/autodl-tmp/data/router_runs/formal_sar/features/train_full.pt",
        "val_features": "/root/autodl-tmp/data/router_runs/formal_sar/features/val_full.pt",
        "routing_policy": "stage2_fallback",
        "stage2_abs_threshold": 0.20,
        "selection_metric": "accuracy",
        "class_weight_mode": "none",
        "label_smoothing": 0.05,
        "lr": 0.0005,
        "weight_decay": 0.02,
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
