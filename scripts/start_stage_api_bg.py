#!/usr/bin/env python3
"""Start a LLaMA-Factory API process in background without shell indirection."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Start stage API in background.")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--pid_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path = Path(args.pid_file)
    pid_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["API_HOST"] = "0.0.0.0"
    env["API_PORT"] = str(args.port)
    # Prefer fully local startup once the base model has been downloaded.
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")

    cmd = [
        "llamafactory-cli",
        "api",
        "--model_name_or_path",
        args.model_name_or_path,
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
        args.adapter_path,
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

    pid_path.write_text(
        json.dumps(
            {
                "pid": process.pid,
                "port": args.port,
                "adapter_path": args.adapter_path,
                "log_file": str(log_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"pid": process.pid, "port": args.port, "log_file": str(log_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
