from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmm_chart.config import load_experiment_config
from tmm_chart.data.pipeline import DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick smoke test for dataset pipeline.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--limit", type=int, default=10, help="Small chart count for smoke testing.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    summary = DatasetBuilder(config.dataset, config.root_dir).build(limit=args.limit)
    manifest_root = config.root_dir / "data" / "synthetic" / "manifests"
    required = [
        manifest_root / "charts.jsonl",
        manifest_root / "stage1_description.jsonl",
        manifest_root / "stage2_basic_vqa.jsonl",
        manifest_root / "stage3_reasoning_vqa.jsonl",
        manifest_root / "stage4_visual_analysis.jsonl",
        manifest_root / "stage5_code_generation.jsonl",
        manifest_root / "router_holdout.jsonl",
    ]
    status = {str(path.name): path.exists() for path in required}
    print(json.dumps({"summary": summary, "artifacts": status}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
