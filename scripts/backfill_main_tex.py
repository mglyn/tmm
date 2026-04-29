from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmm_chart.config import load_experiment_config
from tmm_chart.eval.reporting import backfill_tex_from_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill main.tex using aggregated experiment metrics.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--tex", default=None, help="Optional override target tex path.")
    parser.add_argument("--metrics", default=None, help="Optional override metrics json path.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    target_tex = config.root_dir / (args.tex or config.evaluation["paper_backfill"]["target_tex"])
    metrics_path = config.root_dir / (args.metrics or config.evaluation["paper_backfill"]["metrics_json"])
    backfill_tex_from_metrics(target_tex, metrics_path)
    print(f"Updated {target_tex} from {metrics_path}")


if __name__ == "__main__":
    main()
