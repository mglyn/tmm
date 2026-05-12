#!/usr/bin/env python3
"""Evaluate 2D-CL + SAR on ChartQA using cached stage predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from datasets import load_from_disk
from PIL import Image
from torch import nn

from build_router_dataset import DEFAULT_STAGE_ORDER, is_answer_correct


# Keep the environment aligned with the existing feature extraction script.
_original_find_spec = importlib.util.find_spec


def _patched_find_spec(name, package=None):
    blocked_prefixes = ("scipy", "sklearn")
    if any(str(name) == prefix or str(name).startswith(prefix + ".") for prefix in blocked_prefixes):
        return None
    return _original_find_spec(name, package)


importlib.util.find_spec = _patched_find_spec

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - fallback for older installs
    from transformers import AutoModelForImageTextToText, AutoProcessor

    Qwen2_5_VLForConditionalGeneration = None


class RouterBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, residual: bool):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.use_residual = residual and in_dim == out_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(features)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        if self.use_residual:
            hidden = hidden + features
        return hidden


class MultimodalRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float, num_labels: int, residual: bool = False):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self.input_norm = nn.LayerNorm(input_dim)
        self.blocks = nn.ModuleList(
            [
                RouterBlock(in_dim=dims[idx], out_dim=dims[idx + 1], dropout=dropout, residual=residual)
                for idx in range(len(dims) - 1)
            ]
        )
        final_dim = dims[-1] if hidden_dims else input_dim
        self.output = nn.Linear(final_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_norm(features)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


def resolve_hidden_dims(config: Dict) -> List[int]:
    if "hidden_dims_resolved" in config and config["hidden_dims_resolved"] is not None:
        return [int(value) for value in config["hidden_dims_resolved"]]
    if "hidden_dims" in config and config["hidden_dims"]:
        return [int(part.strip()) for part in str(config["hidden_dims"]).split(",") if part.strip()]
    if "hidden_dim" in config:
        num_hidden_layers = int(config.get("num_hidden_layers", 1))
        return [int(config["hidden_dim"])] * num_hidden_layers
    return []


def read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_model(model_path: str, dtype: torch.dtype, device_map: str | None):
    if Qwen2_5_VLForConditionalGeneration is not None:
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    return AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )


def build_inputs(processor, images: Sequence[Image.Image], questions: Sequence[str], device: torch.device) -> Dict[str, torch.Tensor]:
    texts: List[str] = []
    for question in questions:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    batch = processor(
        text=texts,
        images=list(images),
        padding=True,
        return_tensors=None,
    )

    converted: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value is None:
            continue
        tensor = torch.as_tensor(value)
        if key in {"pixel_values", "pixel_values_videos"}:
            tensor = tensor.to(device=device, dtype=torch.float32)
        else:
            tensor = tensor.to(device)
        converted[key] = tensor
    return converted


def pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def load_stage_predictions(stage_files: Dict[str, Path]) -> List[Dict]:
    per_stage = {stage: read_jsonl(path) for stage, path in stage_files.items()}
    lengths = {stage: len(rows) for stage, rows in per_stage.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent stage file lengths: {lengths}")

    total = next(iter(lengths.values()))
    merged_records: List[Dict] = []
    for idx in range(total):
        base = per_stage[DEFAULT_STAGE_ORDER[0]][idx]
        sample_id = int(base["sample_id"])
        question = base["question"]
        gold_answers = base["gold_answers"]
        stage_predictions = {}
        for stage in DEFAULT_STAGE_ORDER:
            row = per_stage[stage][idx]
            if int(row["sample_id"]) != sample_id:
                raise ValueError(
                    f"Sample id mismatch at row {idx}: {stage} has {row['sample_id']}, expected {sample_id}"
                )
            stage_predictions[stage] = row["prediction"]
        merged_records.append(
            {
                "sample_id": sample_id,
                "question": question,
                "gold_answers": gold_answers,
                "stage_predictions": stage_predictions,
            }
        )
    return merged_records


def parse_stage_files(items: Sequence[str]) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got: {item}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = Path(value.strip())
    missing = [stage for stage in DEFAULT_STAGE_ORDER if stage not in parsed]
    if missing:
        raise ValueError(f"Missing stage files for: {', '.join(missing)}")
    return parsed


def apply_routing_policy(
    predicted_stage: str,
    probabilities: Dict[str, float],
    policy_name: str,
    stage2_abs_threshold: float,
) -> str:
    if policy_name == "top1":
        return predicted_stage
    if policy_name == "stage2_fallback":
        if predicted_stage != "stage2" and probabilities["stage2"] >= stage2_abs_threshold:
            return "stage2"
        return predicted_stage
    raise ValueError(f"Unsupported routing_policy: {policy_name}")


def run_router_batch(
    records: Sequence[Dict],
    dataset,
    processor,
    backbone,
    router: nn.Module,
    device: torch.device,
) -> tuple[List[int], List[List[float]]]:
    images = []
    questions = []
    for record in records:
        sample = dataset[int(record["sample_id"])]
        images.append(sample["image"])
        questions.append(record["question"])

    try:
        with torch.no_grad():
            batch_inputs = build_inputs(processor, images=images, questions=questions, device=device)
            outputs = backbone(
                **batch_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states[-1]
            attention_mask = batch_inputs["attention_mask"]
            pooled = pool_hidden_states(hidden_states, attention_mask)
            pooled = pooled.to(next(router.parameters()).dtype)
            logits = router(pooled)
            pred_ids = torch.argmax(logits, dim=-1).detach().cpu().tolist()
            probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        del batch_inputs, outputs, hidden_states, attention_mask, pooled, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return pred_ids, probs
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(records) == 1:
            raise
        mid = len(records) // 2
        left_pred, left_prob = run_router_batch(
            records=records[:mid],
            dataset=dataset,
            processor=processor,
            backbone=backbone,
            router=router,
            device=device,
        )
        right_pred, right_prob = run_router_batch(
            records=records[mid:],
            dataset=dataset,
            processor=processor,
            backbone=backbone,
            router=router,
            device=device,
        )
        return left_pred + right_pred, left_prob + right_prob


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate 2D-CL + SAR on ChartQA.")
    parser.add_argument("--router_checkpoint", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/data/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/root/autodl-tmp/data/legacy/chart_vqa_synthesis_1/chartqa_dataset",
    )
    parser.add_argument("--dataset_split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument(
        "--routing_policy",
        type=str,
        default="stage2_fallback",
        choices=["top1", "stage2_fallback"],
        help="Default uses the current best reproducible stage2 fallback policy.",
    )
    parser.add_argument(
        "--stage2_abs_threshold",
        type=float,
        default=0.20,
        help="When routing_policy=stage2_fallback, reroute non-stage2 top1 predictions to stage2 if p(stage2) exceeds this threshold.",
    )
    parser.add_argument(
        "--stage-file",
        action="append",
        default=[],
        help="Repeatable key=value argument, e.g. stage2=/path/to/stage2.jsonl",
    )
    parser.add_argument(
        "--stage_cache_dir",
        type=str,
        default="/root/autodl-tmp/data/router_data/full_pipeline/cache",
        help="Used only when --stage-file is not provided.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else output_dir / "eval.log"

    if args.stage_file:
        stage_files = parse_stage_files(args.stage_file)
    else:
        stage_cache_dir = Path(args.stage_cache_dir)
        stage_files = {stage: stage_cache_dir / f"{stage}.jsonl" for stage in DEFAULT_STAGE_ORDER}

    for stage, path in stage_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Stage cache for {stage} not found: {path}")

    merged_records = load_stage_predictions(stage_files)
    if args.max_samples is not None:
        merged_records = merged_records[: args.max_samples]

    dataset = load_from_disk(args.dataset_path)[args.dataset_split]
    checkpoint = torch.load(args.router_checkpoint, map_location="cpu")
    config = checkpoint["config"]

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)
    backbone = load_model(args.model_path, dtype=dtype, device_map=args.device_map)
    backbone.eval()
    if args.device_map is None:
        backbone.to(device)

    router = MultimodalRouter(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=resolve_hidden_dims(config),
        dropout=float(config["dropout"]),
        num_labels=len(DEFAULT_STAGE_ORDER),
        residual=bool(config.get("residual", False)),
    )
    router.load_state_dict(checkpoint["model_state_dict"])
    router.to(device)
    router.eval()

    run_config = {
        "router_checkpoint": args.router_checkpoint,
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "dataset_split": args.dataset_split,
        "output_dir": str(output_dir),
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "device": args.device,
        "device_map": args.device_map,
        "max_samples": args.max_samples,
        "log_every": args.log_every,
        "routing_policy": args.routing_policy,
        "stage2_abs_threshold": args.stage2_abs_threshold,
        "stage_files": {stage: str(path) for stage, path in stage_files.items()},
    }
    save_json(output_dir / "config.json", run_config)

    results_path = output_dir / "predictions.jsonl"
    results_path.write_text("", encoding="utf-8")

    start_time = time.time()
    stage_usage = Counter()
    correct = 0
    total = 0
    latencies: List[float] = []

    for start_idx in range(0, len(merged_records), args.batch_size):
        batch_records = merged_records[start_idx : start_idx + args.batch_size]
        batch_start = time.time()
        pred_ids, probs = run_router_batch(
            records=batch_records,
            dataset=dataset,
            processor=processor,
            backbone=backbone,
            router=router,
            device=device,
        )
        batch_seconds = time.time() - batch_start
        latencies.append(batch_seconds)

        with results_path.open("a", encoding="utf-8") as handle:
            for record, pred_id, prob in zip(batch_records, pred_ids, probs):
                raw_predicted_stage = DEFAULT_STAGE_ORDER[pred_id]
                router_probabilities = {
                    stage: round(float(prob[idx]), 6) for idx, stage in enumerate(DEFAULT_STAGE_ORDER)
                }
                selected_stage = apply_routing_policy(
                    predicted_stage=raw_predicted_stage,
                    probabilities=router_probabilities,
                    policy_name=args.routing_policy,
                    stage2_abs_threshold=args.stage2_abs_threshold,
                )
                prediction = record["stage_predictions"][selected_stage]
                is_correct = is_answer_correct(prediction, record["gold_answers"])
                correct += int(is_correct)
                total += 1
                stage_usage[selected_stage] += 1

                row = {
                    "sample_id": record["sample_id"],
                    "question": record["question"],
                    "gold_answers": record["gold_answers"],
                    "predicted_stage_raw": raw_predicted_stage,
                    "predicted_stage_raw_id": pred_id,
                    "selected_stage": selected_stage,
                    "routing_policy": args.routing_policy,
                    "router_probabilities": router_probabilities,
                    "final_prediction": prediction,
                    "correct": is_correct,
                    "stage_predictions": record["stage_predictions"],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        done = total
        elapsed = time.time() - start_time
        if done == len(merged_records) or (args.log_every > 0 and done % args.log_every == 0):
            progress = {
                "done": done,
                "total": len(merged_records),
                "accuracy": round(correct / max(total, 1), 4),
                "elapsed_seconds": round(elapsed, 2),
                "avg_seconds_per_sample": round(elapsed / max(done, 1), 4),
                "last_batch_seconds": round(batch_seconds, 2),
            }
            print(json.dumps(progress, ensure_ascii=False), flush=True)
            append_jsonl(log_file, progress)
            save_json(
                output_dir / "progress.json",
                {
                    **progress,
                    "stage_usage": dict(stage_usage),
                },
            )

    elapsed_seconds = time.time() - start_time
    accuracy = correct / max(total, 1)
    summary = {
        "num_samples": total,
        "num_correct": correct,
        "accuracy": accuracy,
        "accuracy_percent": round(accuracy * 100, 4),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "avg_seconds_per_sample": round(elapsed_seconds / max(total, 1), 4),
        "avg_batch_seconds": round(sum(latencies) / max(len(latencies), 1), 4),
        "routing_policy": args.routing_policy,
        "stage2_abs_threshold": args.stage2_abs_threshold,
        "stage_usage": {
            stage: {
                "count": stage_usage.get(stage, 0),
                "ratio": round(stage_usage.get(stage, 0) / max(total, 1), 6),
            }
            for stage in DEFAULT_STAGE_ORDER
        },
    }
    save_json(output_dir / "summary.json", summary)
    append_jsonl(log_file, {"event": "completed", **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
