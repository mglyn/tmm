#!/usr/bin/env python3
"""Prepare a manual annotation pack for E10 error analysis.

This script aligns per-sample predictions from two systems, keeps failure cases,
and exports CSV/JSON files for manual labeling with a shared error taxonomy.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ERROR_LABELS = [
    "value extraction",
    "arithmetic",
    "multi-step reasoning",
    "answer format",
    "legend/axis mapping",
    "counting/dense perception",
]


def load_records(path: Path, system_name: str) -> Dict[int, Dict]:
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("results", [])
    else:
        raise ValueError(f"Unsupported file format: {path}")

    mapped: Dict[int, Dict] = {}
    for row in records:
        sample_id = int(row["sample_id"])
        question = row["question"]
        gold_answers = row.get("gold_answer", row.get("gold_answers", []))
        prediction = row.get("pred_answer", row.get("final_prediction", row.get("prediction", "")))
        correct = row.get("correct")
        if correct is None:
            correct = infer_correct_from_prediction(prediction, gold_answers)
        mapped[sample_id] = {
            "sample_id": sample_id,
            "question": question,
            "gold_answers": gold_answers,
            "prediction": prediction,
            "correct": correct,
            "system_name": system_name,
        }
    return mapped


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s\.]", "", text)
    return " ".join(text.split())


def infer_correct_from_prediction(prediction: str, gold_answers: Iterable[str]) -> bool:
    pred_norm = normalize_text(prediction)
    for gold in gold_answers:
        gold_norm = normalize_text(gold)
        if pred_norm == gold_norm:
            return True
        if gold_norm and (gold_norm in pred_norm or pred_norm in gold_norm):
            return True
        pred_num = extract_first_number(prediction)
        gold_num = extract_first_number(str(gold))
        if pred_num is not None and gold_num is not None:
            abs_err = abs(pred_num - gold_num)
            rel_err = abs_err / (abs(gold_num) + 1e-8)
            if abs_err < 0.01 or rel_err < 0.01:
                return True
    return False


def heuristic_error_tags(question: str, prediction: str, gold_answers: Iterable[str]) -> List[str]:
    q = question.lower()
    pred = str(prediction).lower()
    gold_text = " | ".join(str(item).lower() for item in gold_answers)

    tags: List[str] = []

    if any(token in q for token in ["difference", "sum", "total", "average", "ratio", "more than", "less than"]):
        tags.append("arithmetic")
    if any(token in q for token in ["how many", "count", "number of bars", "number of slices"]):
        tags.append("counting/dense perception")
    if any(token in q for token in ["legend", "axis", "label", "bar", "line", "slice", "color"]):
        tags.append("legend/axis mapping")
    if any(token in q for token in ["highest", "lowest", "4th", "trend", "which", "compare", "more", "less"]):
        tags.append("multi-step reasoning")
    if any(token in pred for token in ["the chart", "according to", "there are", "shows"]) and len(pred.split()) > 6:
        tags.append("answer format")

    gold_num = extract_first_number(gold_text)
    pred_num = extract_first_number(pred)
    if gold_num is not None and pred_num is not None and abs(gold_num - pred_num) > 0.01:
        tags.append("value extraction")

    if not tags:
        tags.append("value extraction")
    return list(dict.fromkeys(tags))


def extract_first_number(text: str) -> float | None:
    match = re.search(r"-?\d+\.?\d*", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def build_annotation_rows(
    baseline_records: Dict[int, Dict],
    target_records: Dict[int, Dict],
    sample_size: int,
    seed: int,
) -> List[Dict]:
    aligned_ids = sorted(set(baseline_records) & set(target_records))
    candidates: List[Dict] = []

    for sample_id in aligned_ids:
        base = baseline_records[sample_id]
        target = target_records[sample_id]
        if base["correct"] and target["correct"]:
            continue
        candidates.append(
            {
                "sample_id": sample_id,
                "question": base["question"],
                "gold_answers": base["gold_answers"],
                "baseline_prediction": base["prediction"],
                "baseline_correct": base["correct"],
                "target_prediction": target["prediction"],
                "target_correct": target["correct"],
                "suggested_error_tags": heuristic_error_tags(
                    question=base["question"],
                    prediction=target["prediction"] if not target["correct"] else base["prediction"],
                    gold_answers=base["gold_answers"],
                ),
                "manual_error_tags": "",
                "notes": "",
            }
        )

    rng = random.Random(seed)
    if sample_size and sample_size < len(candidates):
        candidates = rng.sample(candidates, sample_size)
        candidates.sort(key=lambda row: row["sample_id"])
    return candidates


def save_outputs(rows: List[Dict], output_dir: Path) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "error_analysis_samples.json"
    csv_path = output_dir / "error_analysis_samples.csv"
    guide_path = output_dir / "error_taxonomy_guide.txt"

    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys()) if rows else [
            "sample_id",
            "question",
            "gold_answers",
            "baseline_prediction",
            "baseline_correct",
            "target_prediction",
            "target_correct",
            "suggested_error_tags",
            "manual_error_tags",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = row.copy()
            serialized["gold_answers"] = json.dumps(serialized["gold_answers"], ensure_ascii=False)
            serialized["suggested_error_tags"] = "|".join(serialized["suggested_error_tags"])
            writer.writerow(serialized)

    guide_lines = [
        "E10 error taxonomy guide",
        "",
        "Allowed labels:",
        *[f"- {label}" for label in ERROR_LABELS],
        "",
        "Annotation rule:",
        "- Label the primary error cause for each failed sample.",
        "- If both systems fail differently, note the target system's main failure in `manual_error_tags` and describe the contrast in `notes`.",
    ]
    guide_path.write_text("\n".join(guide_lines), encoding="utf-8")
    return json_path, csv_path, guide_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare manual annotation pack for E10 error analysis.")
    parser.add_argument("--baseline_file", type=str, required=True, help="JSON or JSONL with baseline per-sample predictions.")
    parser.add_argument("--target_file", type=str, required=True, help="JSON or JSONL with target per-sample predictions.")
    parser.add_argument("--baseline_name", type=str, default="baseline")
    parser.add_argument("--target_name", type=str, default="target")
    parser.add_argument("--sample_size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    baseline_records = load_records(Path(args.baseline_file), args.baseline_name)
    target_records = load_records(Path(args.target_file), args.target_name)
    rows = build_annotation_rows(
        baseline_records=baseline_records,
        target_records=target_records,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    json_path, csv_path, guide_path = save_outputs(rows, Path(args.output_dir))

    summary = {
        "baseline_name": args.baseline_name,
        "target_name": args.target_name,
        "num_aligned_samples": len(set(baseline_records) & set(target_records)),
        "num_annotation_samples": len(rows),
        "output_json": str(json_path),
        "output_csv": str(csv_path),
        "taxonomy_guide": str(guide_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
