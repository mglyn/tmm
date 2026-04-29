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
    parser = argparse.ArgumentParser(description="Build synthetic chart dataset and stage manifests.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--limit", type=int, default=None, help="Optional small-scale build for smoke testing.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    builder = DatasetBuilder(config.dataset, config.root_dir)
    summary = builder.build(limit=args.limit)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
