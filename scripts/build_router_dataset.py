#!/usr/bin/env python3
"""Build oracle routing data for stage-aware adapter selection.

This script evaluates multiple stage experts on the same ChartQA split via
OpenAI-compatible APIs, then produces per-sample oracle routing labels.

Example:
python /data/scripts/build_router_dataset.py \
  --dataset_path /data/legacy/chart_vqa_synthesis_1/chartqa_dataset \
  --split test \
  --stage-endpoint stage2=http://localhost:8002 \
  --stage-endpoint stage3=http://localhost:8003 \
  --stage-endpoint stage4=http://localhost:8004 \
  --stage-endpoint stage5=http://localhost:8005 \
  --output_dir /data/router_data/chartqa_router
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import requests
from datasets import load_from_disk


DEFAULT_STAGE_ORDER = ["stage2", "stage3", "stage4", "stage5"]


@dataclass
class StageClient:
    name: str
    api_base: str
    model_name: str


def parse_key_value(items: Iterable[str], default_value: str = "") -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected key=value format, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Empty key in argument: {item}")
        parsed[key] = value if value else default_value
    return parsed


def encode_image_to_base64(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s\.]", "", text)
    return " ".join(text.split())


def extract_first_number(text: str):
    if text is None:
        return None
    text = str(text).replace(",", "")
    match = re.search(r"-?\d+\.?\d*", text)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def is_answer_correct(prediction: str, gold_answers: List[str]) -> bool:
    pred_norm = normalize_text(prediction)
    for gold in gold_answers:
        gold_norm = normalize_text(gold)
        if pred_norm == gold_norm:
            return True
        if gold_norm and (gold_norm in pred_norm or pred_norm in gold_norm):
            return True

        pred_num = extract_first_number(prediction)
        gold_num = extract_first_number(gold)
        if pred_num is not None and gold_num is not None:
            abs_err = abs(pred_num - gold_num)
            rel_err = abs_err / (abs(gold_num) + 1e-8)
            if abs_err < 0.01 or rel_err < 0.01:
                return True

    return False


def answer_error_score(prediction: str, gold_answers: List[str]) -> float:
    """Smaller is better.

    0 means exact/correct under the same rule as `is_answer_correct`.
    """
    if is_answer_correct(prediction, gold_answers):
        return 0.0

    pred_num = extract_first_number(prediction)
    numeric_scores: List[float] = []
    text_scores: List[float] = []

    for gold in gold_answers:
        gold_num = extract_first_number(gold)
        if pred_num is not None and gold_num is not None:
            numeric_scores.append(abs(pred_num - gold_num) / (abs(gold_num) + 1e-8))

        pred_norm = normalize_text(prediction)
        gold_norm = normalize_text(gold)
        text_scores.append(1.0 - SequenceMatcher(None, pred_norm, gold_norm).ratio())

    if numeric_scores:
        return min(numeric_scores)
    if text_scores:
        return min(text_scores)
    return 1.0


def heuristic_stage_priority(question: str, stage_order: List[str]) -> List[str]:
    q = question.lower()
    if any(token in q for token in ["sum", "difference", "total", "average", "ratio", "more than", "less than"]):
        preferred = ["stage3", "stage2", "stage4", "stage5"]
    elif any(token in q for token in ["legend", "axis", "bar", "line", "slice", "color", "chart type", "how many bars"]):
        preferred = ["stage4", "stage2", "stage3", "stage5"]
    elif any(token in q for token in ["code", "generate", "python", "script"]):
        preferred = ["stage5", "stage3", "stage4", "stage2"]
    else:
        preferred = ["stage2", "stage3", "stage4", "stage5"]

    filtered = [stage for stage in preferred if stage in stage_order]
    remaining = [stage for stage in stage_order if stage not in filtered]
    return filtered + remaining


def pick_oracle_stage(
    question: str,
    gold_answers: List[str],
    stage_predictions: Dict[str, str],
    stage_order: List[str],
) -> Tuple[str, List[str], Dict[str, float], str]:
    correct_stages = [
        stage
        for stage in stage_order
        if is_answer_correct(stage_predictions[stage], gold_answers)
    ]
    error_scores = {
        stage: answer_error_score(stage_predictions[stage], gold_answers)
        for stage in stage_order
    }

    if correct_stages:
        priority = heuristic_stage_priority(question, correct_stages)
        oracle_stage = priority[0]
        label_source = "correct_stage"
    else:
        oracle_stage = min(
            stage_order,
            key=lambda stage: (error_scores[stage], stage_order.index(stage)),
        )
        label_source = "least_error_stage"

    return oracle_stage, correct_stages, error_scores, label_source


def build_request_payload(model_name: str, image_b64: str, question: str, max_tokens: int) -> Dict:
    return {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    {"type": "text", "text": question},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }


def query_stage(client: StageClient, image_b64: str, question: str, max_tokens: int, timeout: int) -> str:
    payload = build_request_payload(client.model_name, image_b64, question, max_tokens=max_tokens)
    response = requests.post(
        f"{client.api_base}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def ensure_stage_health(clients: List[StageClient], timeout: int) -> None:
    for client in clients:
        response = requests.get(f"{client.api_base}/v1/models", timeout=timeout)
        response.raise_for_status()


def train_val_split(records: List[Dict], train_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_size = int(len(shuffled) * train_ratio)
    return shuffled[:train_size], shuffled[train_size:]


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build oracle routing dataset from multi-stage API predictions.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to local load_from_disk ChartQA dataset.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep_between_requests", type=float, default=0.0)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stage-endpoint",
        action="append",
        required=True,
        help="Repeatable key=value argument, e.g. stage2=http://localhost:8002",
    )
    parser.add_argument(
        "--stage-model",
        action="append",
        default=[],
        help="Optional key=value argument, e.g. stage2=default. Defaults to 'default'.",
    )
    args = parser.parse_args()

    endpoint_map = parse_key_value(args.stage_endpoint)
    model_map = parse_key_value(args.stage_model, default_value="default")

    stage_order = [stage for stage in DEFAULT_STAGE_ORDER if stage in endpoint_map]
    missing = [stage for stage in DEFAULT_STAGE_ORDER if stage not in endpoint_map]
    if missing:
        raise ValueError(f"Missing stage endpoints for: {', '.join(missing)}")

    clients = [
        StageClient(name=stage, api_base=endpoint_map[stage].rstrip("/"), model_name=model_map.get(stage, "default"))
        for stage in stage_order
    ]

    ensure_stage_health(clients, timeout=args.timeout)

    dataset = load_from_disk(args.dataset_path)[args.split]
    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.sample_limit is not None:
        dataset = dataset.select(range(min(args.sample_limit, len(dataset))))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    stage_hit_count = {stage: 0 for stage in stage_order}
    correct_any_count = 0

    for relative_idx, sample in enumerate(dataset):
        sample_id = args.start_index + relative_idx
        question = sample["query"]
        gold_answers = sample["label"] if isinstance(sample["label"], list) else [sample["label"]]
        image_b64 = encode_image_to_base64(sample["image"])

        stage_predictions: Dict[str, str] = {}
        for client in clients:
            prediction = query_stage(
                client,
                image_b64=image_b64,
                question=question,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            stage_predictions[client.name] = prediction
            if args.sleep_between_requests > 0:
                time.sleep(args.sleep_between_requests)

        oracle_stage, correct_stages, error_scores, label_source = pick_oracle_stage(
            question=question,
            gold_answers=gold_answers,
            stage_predictions=stage_predictions,
            stage_order=stage_order,
        )

        if correct_stages:
            correct_any_count += 1
        stage_hit_count[oracle_stage] += 1

        record = {
            "dataset_path": args.dataset_path,
            "dataset_split": args.split,
            "sample_id": sample_id,
            "question": question,
            "gold_answers": gold_answers,
            "oracle_stage": oracle_stage,
            "oracle_stage_id": stage_order.index(oracle_stage),
            "label_source": label_source,
            "correct_stages": correct_stages,
            "stage_predictions": stage_predictions,
            "stage_correctness": {
                stage: stage in correct_stages for stage in stage_order
            },
            "stage_error_scores": error_scores,
        }
        records.append(record)

    train_records, val_records = train_val_split(records, train_ratio=args.train_ratio, seed=args.seed)

    write_jsonl(output_dir / "router_all.jsonl", records)
    write_jsonl(output_dir / "router_train.jsonl", train_records)
    write_jsonl(output_dir / "router_val.jsonl", val_records)

    summary = {
        "dataset_path": args.dataset_path,
        "split": args.split,
        "num_samples": len(records),
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "correct_any_rate": (correct_any_count / len(records)) if records else 0.0,
        "oracle_distribution": {
            stage: {
                "count": count,
                "ratio": (count / len(records)) if records else 0.0,
            }
            for stage, count in stage_hit_count.items()
        },
        "stage_endpoints": endpoint_map,
        "stage_models": {stage: model_map.get(stage, "default") for stage in stage_order},
    }
    write_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
