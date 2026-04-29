from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmm_chart.config import load_experiment_config
from tmm_chart.router.router import train_router


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the stage-aware router.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--limit", type=int, default=None, help="Optional router sample limit for smoke testing.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    artifacts = train_router(config.raw, config.root_dir, limit=args.limit)
    print(
        json.dumps(
            {
                "model_path": str(artifacts.model_path),
                "metrics_path": str(artifacts.metrics_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
