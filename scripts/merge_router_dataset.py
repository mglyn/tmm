#!/usr/bin/env python3
"""Merge per-stage prediction files into oracle router labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from build_router_dataset import (
    DEFAULT_STAGE_ORDER,
    pick_oracle_stage,
    write_json,
    write_jsonl,
)


def read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def split_records(records: List[Dict], train_ratio: float, seed: int):
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    train_size = int(len(shuffled) * train_ratio)
    return shuffled[:train_size], shuffled[train_size:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge stage prediction caches into router dataset.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stage-file",
        action="append",
        required=True,
        help="Repeatable key=value argument, e.g. stage2=/path/to/stage2.jsonl",
    )
    args = parser.parse_args()

    stage_files: Dict[str, Path] = {}
    for item in args.stage_file:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got: {item}")
        key, value = item.split("=", 1)
        stage_files[key.strip()] = Path(value.strip())

    stage_order = [stage for stage in DEFAULT_STAGE_ORDER if stage in stage_files]
    missing = [stage for stage in DEFAULT_STAGE_ORDER if stage not in stage_files]
    if missing:
        raise ValueError(f"Missing stage files for: {', '.join(missing)}")

    per_stage = {stage: read_jsonl(path) for stage, path in stage_files.items()}
    lengths = {stage: len(rows) for stage, rows in per_stage.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Stage files have inconsistent lengths: {lengths}")

    merged_records: List[Dict] = []
    stage_hit_count = {stage: 0 for stage in stage_order}
    correct_any_count = 0
    label_source_count = {"correct_stage": 0, "least_error_stage": 0}

    total = lengths[stage_order[0]]
    for idx in range(total):
        base = per_stage[stage_order[0]][idx]
        sample_id = base["sample_id"]
        question = base["question"]
        gold_answers = base["gold_answers"]

        stage_predictions = {}
        for stage in stage_order:
            row = per_stage[stage][idx]
            if row["sample_id"] != sample_id:
                raise ValueError(f"Sample id mismatch at row {idx}: {stage} has {row['sample_id']}, expected {sample_id}")
            stage_predictions[stage] = row["prediction"]

        oracle_stage, correct_stages, error_scores, label_source = pick_oracle_stage(
            question=question,
            gold_answers=gold_answers,
            stage_predictions=stage_predictions,
            stage_order=stage_order,
        )

        if correct_stages:
            correct_any_count += 1
        stage_hit_count[oracle_stage] += 1
        label_source_count[label_source] = label_source_count.get(label_source, 0) + 1

        merged_records.append(
            {
                "sample_id": sample_id,
                "question": question,
                "gold_answers": gold_answers,
                "oracle_stage": oracle_stage,
                "oracle_stage_id": stage_order.index(oracle_stage),
                "label_source": label_source,
                "correct_stages": correct_stages,
                "stage_predictions": stage_predictions,
                "stage_correctness": {stage: stage in correct_stages for stage in stage_order},
                "stage_error_scores": error_scores,
            }
        )

    train_records, val_records = split_records(merged_records, train_ratio=args.train_ratio, seed=args.seed)
    clean_records = [record for record in merged_records if record["label_source"] == "correct_stage"]
    clean_train_records, clean_val_records = split_records(clean_records, train_ratio=args.train_ratio, seed=args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "router_all.jsonl", merged_records)
    write_jsonl(output_dir / "router_train.jsonl", train_records)
    write_jsonl(output_dir / "router_val.jsonl", val_records)
    write_jsonl(output_dir / "router_all_clean.jsonl", clean_records)
    write_jsonl(output_dir / "router_train_clean.jsonl", clean_train_records)
    write_jsonl(output_dir / "router_val_clean.jsonl", clean_val_records)
    write_json(
        output_dir / "summary.json",
        {
            "num_samples": len(merged_records),
            "num_clean_samples": len(clean_records),
            "train_ratio": args.train_ratio,
            "seed": args.seed,
            "correct_any_rate": (correct_any_count / len(merged_records)) if merged_records else 0.0,
            "label_source_distribution": {
                source: {
                    "count": count,
                    "ratio": (count / len(merged_records)) if merged_records else 0.0,
                }
                for source, count in label_source_count.items()
            },
            "oracle_distribution": {
                stage: {
                    "count": count,
                    "ratio": (count / len(merged_records)) if merged_records else 0.0,
                }
                for stage, count in stage_hit_count.items()
            },
            "stage_files": {stage: str(path) for stage, path in stage_files.items()},
        },
    )


if __name__ == "__main__":
    main()
