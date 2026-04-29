from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmm_chart.config import load_experiment_config
from tmm_chart.eval.analysis import (
    build_corrupted_benchmark,
    build_error_taxonomy,
    build_stage_task_profile,
    create_scaled_manifests,
    sample_difficulty_audit,
    summarize_difficulty_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supporting analysis utilities for TMM experiments.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument(
        "--task",
        choices=["scaling", "robustness", "difficulty_audit", "difficulty_summary", "error_taxonomy", "task_profile"],
        required=True,
    )
    parser.add_argument("--input", default=None, help="Optional input path for task-specific commands.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    root = config.root_dir
    manifests_dir = root / "data" / "synthetic" / "manifests"
    analysis_dir = root / "outputs" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "scaling":
        manifest = manifests_dir / "stage2_basic_vqa.jsonl"
        created = create_scaled_manifests(manifest, [0.1, 0.25, 0.5, 1.0, 1.5], analysis_dir / "scaling", config.dataset["seed"])
        print(json.dumps({"created": [str(path) for path in created]}, indent=2, ensure_ascii=False))
    elif args.task == "robustness":
        generated = build_corrupted_benchmark(root / "data" / "benchmarks", "ChartQA", config.evaluation["robustness"]["corruptions"])
        print(json.dumps({"generated": [str(path) for path in generated]}, indent=2, ensure_ascii=False))
    elif args.task == "difficulty_audit":
        output = sample_difficulty_audit(
            manifests_dir / "stage3_reasoning_vqa.jsonl",
            config.dataset["difficulty"]["manual_audit_size"],
            analysis_dir / "difficulty_audit.jsonl",
            config.dataset["seed"],
        )
        print(json.dumps({"audit_path": str(output)}, indent=2, ensure_ascii=False))
    elif args.task == "difficulty_summary":
        input_path = Path(args.input) if args.input else analysis_dir / "difficulty_audit.jsonl"
        if not input_path.is_absolute():
            input_path = root / input_path
        output = summarize_difficulty_audit(
            input_path,
            analysis_dir / "difficulty_summary.json",
        )
        print(json.dumps({"summary_path": str(output)}, indent=2, ensure_ascii=False))
    elif args.task == "error_taxonomy":
        if not args.input:
            raise ValueError("--input is required for error_taxonomy and should point to a predictions jsonl.")
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = root / input_path
        output = build_error_taxonomy(input_path, analysis_dir / "error_taxonomy.json")
        print(json.dumps({"error_taxonomy": str(output)}, indent=2, ensure_ascii=False))
    else:
        output = build_stage_task_profile(manifests_dir, analysis_dir / "stage_task_profile.json")
        print(json.dumps({"task_profile": str(output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
