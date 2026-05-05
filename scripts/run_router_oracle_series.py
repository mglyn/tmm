#!/usr/bin/env python3
"""Run router oracle generation for multiple ChartQA splits sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def log(message: str) -> None:
    print(message, flush=True)


def write_status(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run router oracle generation for val/train splits sequentially.")
    parser.add_argument("--base_root", type=str, default=str(PROJECT_ROOT / "router_data"))
    parser.add_argument("--model_path", type=str, default=str(PROJECT_ROOT / "models" / "Qwen2.5-VL-7B-Instruct"))
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(PROJECT_ROOT / "legacy" / "chart_vqa_synthesis_1" / "chartqa_dataset"),
    )
    parser.add_argument("--splits", nargs="+", default=["val", "train"])
    args = parser.parse_args()

    base_root = Path(args.base_root)
    base_root.mkdir(parents=True, exist_ok=True)
    status_path = base_root / "oracle_series_status.json"

    status = {
        "started_at": int(time.time()),
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "splits": args.splits,
        "split_status": {},
    }
    write_status(status_path, status)

    for split in args.splits:
        work_root = base_root / f"{split}_pipeline"
        status["split_status"][split] = "running"
        write_status(status_path, status)
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_router_full_pipeline.py"),
            "--work_root",
            str(work_root),
            "--split",
            split,
            "--model_path",
            args.model_path,
            "--dataset_path",
            args.dataset_path,
        ]
        log("[series] " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        status["split_status"][split] = "completed"
        write_status(status_path, status)

    status["finished_at"] = int(time.time())
    write_status(status_path, status)
    log("[series] completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
