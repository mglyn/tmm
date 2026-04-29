from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmm_chart.config import load_experiment_config
from tmm_chart.train.trainer import train_curriculum, train_standard_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Standard SFT or curriculum stages.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    parser.add_argument("--mode", choices=["sft", "curriculum"], required=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for smoke testing.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.mode == "sft":
        artifact = train_standard_sft(config.raw, config.root_dir, limit=args.limit)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "stage_name": artifact.stage_name,
                    "output_dir": str(artifact.output_dir),
                    "sample_count": artifact.sample_count,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        artifacts = train_curriculum(config.raw, config.root_dir, limit=args.limit)
        payload = [
            {"stage_name": item.stage_name, "output_dir": str(item.output_dir), "sample_count": item.sample_count}
            for item in artifacts
        ]
        print(json.dumps({"mode": args.mode, "artifacts": payload}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
