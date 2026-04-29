from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..utils.common import read_json, read_jsonl, write_json, write_jsonl


@dataclass
class BenchmarkSample:
    sample_id: str
    image_path: str
    question: str
    answer: str
    metadata: dict[str, Any]


def normalize_answer(answer: str, *, lowercase: bool, strip_punctuation: bool) -> str:
    value = answer.strip()
    if lowercase:
        value = value.lower()
    if strip_punctuation:
        value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def numeric_equal(prediction: str, target: str, tolerance: float) -> bool:
    try:
        return abs(float(prediction) - float(target)) <= tolerance
    except ValueError:
        return False


def score_predictions(
    predictions: Iterable[dict[str, Any]],
    normalization_cfg: dict[str, Any],
) -> dict[str, Any]:
    total = 0
    correct = 0
    for item in predictions:
        total += 1
        pred = normalize_answer(
            item["prediction"],
            lowercase=normalization_cfg["lowercase"],
            strip_punctuation=normalization_cfg["strip_punctuation"],
        )
        gold = normalize_answer(
            item["answer"],
            lowercase=normalization_cfg["lowercase"],
            strip_punctuation=normalization_cfg["strip_punctuation"],
        )
        if pred == gold or numeric_equal(pred, gold, normalization_cfg["numeric_tolerance"]):
            correct += 1
    accuracy = 0.0 if total == 0 else correct / total * 100.0
    return {"accuracy": round(accuracy, 2), "count": total, "correct": correct}


def load_benchmark_samples(root_dir: Path, benchmark_name: str) -> list[BenchmarkSample]:
    bench_dir = root_dir / "data" / "benchmarks" / benchmark_name.lower()
    jsonl_path = bench_dir / "test.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {jsonl_path}")
    items = read_jsonl(jsonl_path)
    return [BenchmarkSample(**item) for item in items]


def save_prediction_bundle(
    output_dir: Path,
    benchmark_name: str,
    strategy_name: str,
    predictions: list[dict[str, Any]],
    normalization_cfg: dict[str, Any],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"{benchmark_name.lower()}_{strategy_name}_predictions.jsonl"
    metrics_path = output_dir / f"{benchmark_name.lower()}_{strategy_name}_metrics.json"
    write_jsonl(predictions_path, predictions)
    metrics = score_predictions(predictions, normalization_cfg)
    metrics.update({"benchmark": benchmark_name, "strategy": strategy_name})
    write_json(metrics_path, metrics)
    return metrics
