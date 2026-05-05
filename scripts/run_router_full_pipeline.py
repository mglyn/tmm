#!/usr/bin/env python3
"""Run the full single-GPU router data pipeline sequentially."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import requests
from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
DATASET_ROOT = PROJECT_ROOT / "legacy" / "chart_vqa_synthesis_1" / "chartqa_dataset"
MODEL_ROOT = PROJECT_ROOT / "models" / "Qwen2.5-VL-7B-Instruct"
LLAMA_FACTORY_SRC = PROJECT_ROOT / "legacy" / "chart_vqa_synthesis_1" / "LLaMA-Factory-main" / "src"

DEFAULT_STAGES = {
    "stage2": {
        "port": 8002,
        "adapter": str(PROJECT_ROOT / "legacy" / "chart_vqa_synthesis_1" / "outputs" / "models" / "stage2_basic_vqa"),
    },
    "stage3": {
        "port": 8003,
        "adapter": str(PROJECT_ROOT / "legacy" / "chart_vqa_synthesis_1" / "outputs" / "models" / "stage3_reasoning_vqa"),
    },
    "stage4": {
        "port": 8004,
        "adapter": str(PROJECT_ROOT / "legacy" / "chart_vqa_synthesis_1" / "outputs" / "models" / "stage4_visual_analysis"),
    },
    "stage5": {
        "port": 8005,
        "adapter": str(PROJECT_ROOT / "legacy" / "chart_vqa_synthesis_1" / "outputs" / "models" / "stage5_code_generation"),
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def run(cmd: List[str], env: Dict[str, str] | None = None) -> None:
    log("[run] " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def expected_split_size(dataset_path: str, split: str, sample_limit: int | None) -> int:
    dataset = load_from_disk(dataset_path)[split]
    if sample_limit is not None:
        return min(sample_limit, len(dataset))
    return len(dataset)


def cache_is_complete(output_file: Path, expected_rows: int) -> bool:
    summary_path = output_file.with_suffix(".summary.json")
    if not output_file.exists() or output_file.stat().st_size == 0:
        return False
    if not summary_path.exists():
        return False

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    actual_rows = count_jsonl_rows(output_file)
    summary_rows = int(summary.get("num_samples", -1))
    if summary_rows != expected_rows or actual_rows != expected_rows:
        log(
            f"[cache] incomplete cache detected file={output_file} "
            f"summary_rows={summary_rows} actual_rows={actual_rows} expected_rows={expected_rows}"
        )
        return False

    return True


def start_stage(stage: str, port: int, adapter_path: str, model_path: str, work_root: Path) -> subprocess.Popen:
    log_dir = work_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}_api.log"

    env = os.environ.copy()
    env["API_HOST"] = "0.0.0.0"
    env["API_PORT"] = str(port)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    # Avoid logging full request payloads with base64 images to disk.
    env.setdefault("API_VERBOSE", "0")
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(LLAMA_FACTORY_SRC)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = ":".join(pythonpath_parts)

    cmd = [
        "llamafactory-cli",
        "api",
        "--model_name_or_path",
        model_path,
        "--template",
        "qwen2_vl",
        "--infer_backend",
        "huggingface",
        "--infer_dtype",
        "bfloat16",
        "--trust_remote_code",
        "True",
        "--image_max_pixels",
        "589824",
        "--image_min_pixels",
        "1024",
        "--adapter_name_or_path",
        adapter_path,
        "--finetuning_type",
        "lora",
    ]

    with log_path.open("ab") as log_handle:
        process = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )

    (work_root / "pids").mkdir(parents=True, exist_ok=True)
    (work_root / "pids" / f"{stage}_api.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "pid": process.pid,
                "port": port,
                "adapter_path": adapter_path,
                "log_file": str(log_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"[stage] started stage={stage} pid={process.pid} port={port} log_file={log_path}")
    return process


def wait_for_health(stage: str, port: int, process: subprocess.Popen, timeout: int) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    start = time.time()
    retries = 0
    while time.time() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError(f"{stage} API exited early with code {process.returncode}")
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            log(f"[health] {stage} ready on {url}")
            return
        except Exception:
            retries += 1
            if retries == 1 or retries % 6 == 0:
                elapsed = int(time.time() - start)
                log(f"[health] waiting stage={stage} elapsed={elapsed}s url={url}")
            time.sleep(10)

    raise TimeoutError(f"Timed out waiting for {stage} API health: {url}")


def stop_stage(stage: str, process: subprocess.Popen) -> None:
    if process.poll() is not None:
        log(f"[stop] {stage} already exited code={process.returncode}")
        return

    log(f"[stop] terminating {stage} pid={process.pid}")
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return

    for _ in range(30):
        if process.poll() is not None:
            return
        time.sleep(1)

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except ProcessLookupError:
        return


def collect_stage(stage: str, port: int, dataset_path: str, split: str, sample_limit: int | None, work_root: Path) -> Path:
    cache_dir = work_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_file = cache_dir / f"{stage}.jsonl"
    expected_rows = expected_split_size(dataset_path=dataset_path, split=split, sample_limit=sample_limit)

    if cache_is_complete(output_file=output_file, expected_rows=expected_rows):
        log(f"[skip] existing cache found for {stage}: {output_file}")
        return output_file
    log(f"[collect] begin stage={stage} output_file={output_file}")

    cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "collect_stage_predictions.py"),
        "--stage",
        stage,
        "--api_base",
        f"http://127.0.0.1:{port}",
        "--dataset_path",
        dataset_path,
        "--split",
        split,
        "--output_file",
        str(output_file),
    ]
    if sample_limit is not None:
        cmd.extend(["--sample_limit", str(sample_limit)])

    run(cmd)
    log(f"[collect] finished stage={stage} output_file={output_file}")
    return output_file


def merge_router(work_root: Path) -> None:
    log(f"[merge] begin output_dir={work_root / 'merged'}")
    cmd = [
        sys.executable,
        str(SCRIPTS_ROOT / "merge_router_dataset.py"),
        "--output_dir",
        str(work_root / "merged"),
    ]
    for stage in ["stage2", "stage3", "stage4", "stage5"]:
        cmd.extend(["--stage-file", f"{stage}={work_root / 'cache' / f'{stage}.jsonl'}"])
    run(cmd)
    log(f"[merge] finished output_dir={work_root / 'merged'}")


def update_status(work_root: Path, payload: Dict) -> None:
    (work_root / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[status] {json.dumps(payload, ensure_ascii=False)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run sequential router data collection on a single GPU.")
    parser.add_argument("--model_path", type=str, default=str(MODEL_ROOT))
    parser.add_argument("--dataset_path", type=str, default=str(DATASET_ROOT))
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--work_root", type=str, default=str(PROJECT_ROOT / "router_data" / "full_pipeline"))
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--health_timeout", type=int, default=1800)
    args = parser.parse_args()

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    status = {
        "started_at": int(time.time()),
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "sample_limit": args.sample_limit,
        "stage_status": {},
    }
    update_status(work_root, status)

    for stage in ["stage2", "stage3", "stage4", "stage5"]:
        stage_cfg = DEFAULT_STAGES[stage]
        output_file = work_root / "cache" / f"{stage}.jsonl"
        expected_rows = expected_split_size(dataset_path=args.dataset_path, split=args.split, sample_limit=args.sample_limit)
        if cache_is_complete(output_file=output_file, expected_rows=expected_rows):
            status["stage_status"][stage] = "cached"
            update_status(work_root, status)
            continue

        process = None
        try:
            status["stage_status"][stage] = "starting"
            update_status(work_root, status)
            process = start_stage(
                stage=stage,
                port=stage_cfg["port"],
                adapter_path=stage_cfg["adapter"],
                model_path=args.model_path,
                work_root=work_root,
            )
            wait_for_health(stage=stage, port=stage_cfg["port"], process=process, timeout=args.health_timeout)
            status["stage_status"][stage] = "collecting"
            update_status(work_root, status)
            collect_stage(
                stage=stage,
                port=stage_cfg["port"],
                dataset_path=args.dataset_path,
                split=args.split,
                sample_limit=args.sample_limit,
                work_root=work_root,
            )
            status["stage_status"][stage] = "completed"
            update_status(work_root, status)
        finally:
            if process is not None:
                stop_stage(stage, process)

    status["merge_status"] = "running"
    update_status(work_root, status)
    merge_router(work_root)
    status["merge_status"] = "completed"
    status["finished_at"] = int(time.time())
    update_status(work_root, status)
    log("[done] full router pipeline completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
