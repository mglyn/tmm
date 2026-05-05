#!/usr/bin/env python3
"""Background downloader for Qwen2.5-VL-7B-Instruct via Hugging Face."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download


def log(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    repo_id = os.environ.get("HF_REPO_ID", "Qwen/Qwen2.5-VL-7B-Instruct")
    local_dir = Path(os.environ.get("HF_LOCAL_DIR", "/data/models/Qwen2.5-VL-7B-Instruct"))
    cache_dir = Path(os.environ.get("HF_HUB_CACHE", "/data/.cache/huggingface/hub"))
    max_retries = int(os.environ.get("HF_MAX_RETRIES", "20"))
    sleep_seconds = int(os.environ.get("HF_RETRY_SLEEP", "30"))

    local_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = local_dir / "download_manifest.json"
    manifest = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "cache_dir": str(cache_dir),
        "pid": os.getpid(),
        "started_at": int(time.time()),
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    allow_patterns = [
        "*.json",
        "*.txt",
        "*.model",
        "*.tiktoken",
        "*.py",
        "*.safetensors",
        "*.md",
    ]

    for attempt in range(1, max_retries + 1):
        try:
            log(f"[download] attempt={attempt} repo_id={repo_id}")
            path = snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=str(local_dir),
                cache_dir=str(cache_dir),
                resume_download=True,
                local_dir_use_symlinks=False,
                allow_patterns=allow_patterns,
            )
            log(f"[download] completed path={path}")
            return 0
        except Exception as exc:  # noqa: BLE001
            log(f"[download] failed attempt={attempt} error={exc!r}")
            if attempt == max_retries:
                return 1
            time.sleep(sleep_seconds)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
