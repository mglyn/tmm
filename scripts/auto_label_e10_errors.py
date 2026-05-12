#!/usr/bin/env python3
"""Auto-label E10 error analysis samples via DeepSeek API.

Reads the JSON annotation pack produced by prepare_e10_error_analysis.py,
sends each failed sample to the LLM for error-type classification, and
writes back the labels to JSON and CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

ERROR_LABELS = [
    "value extraction",
    "arithmetic",
    "multi-step reasoning",
    "answer format",
    "legend/axis mapping",
    "counting/dense perception",
]

SYSTEM_PROMPT = (
    "You are an expert annotator for chart VQA error analysis. "
    "Given a chart question, the gold answer, and the model's incorrect prediction, "
    "classify the PRIMARY error cause into exactly one of the following six categories:\n"
    "1. value extraction – the model misread or failed to extract a specific numerical value from the chart.\n"
    "2. arithmetic – the model made a calculation mistake (wrong sum, average, difference, ratio, etc.).\n"
    "3. multi-step reasoning – the model failed to chain multiple operations or interpret the question logic correctly.\n"
    "4. answer format – the model produced a correct or near-correct answer but in the wrong format (verbose text instead of concise answer, wrong unit, etc.).\n"
    "5. legend/axis mapping – the model confused bars/lines/slices, misidentified colors, or mismapped labels to visual elements.\n"
    "6. counting/dense perception – the model miscounted visual elements or failed on densely packed chart regions.\n\n"
    "Reply with ONLY the category name (lowercase, exactly as listed above). No explanation."
)


def build_user_prompt(sample: Dict) -> str:
    question = sample.get("question", "")
    gold = sample.get("gold_answers", [])
    prediction = sample.get("target_prediction", "")
    gold_str = json.dumps(gold, ensure_ascii=False) if isinstance(gold, list) else str(gold)
    return (
        f"Question: {question}\n"
        f"Gold answer: {gold_str}\n"
        f"Model prediction: {prediction}\n"
    )


def classify_sample(
    sample: Dict,
    api_base: str,
    api_key: str,
    model: str,
    retries: int = 3,
    retry_delay: float = 2.0,
    request_timeout: float = 15.0,
) -> Optional[str]:
    prompt = build_user_prompt(sample)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 16,
        "temperature": 0.0,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(
                f"{api_base}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=(5.0, request_timeout),
            )
            if resp.status_code == 200:
                label = resp.json()["choices"][0]["message"]["content"].strip().lower()
                for valid in ERROR_LABELS:
                    if valid in label:
                        return valid
                return label.split("\n")[0]
            elif resp.status_code == 429:
                wait = retry_delay * (2 ** (attempt - 1))
                print(f"  rate-limited, waiting {wait:.0f}s (attempt {attempt}/{retries})")
                time.sleep(wait)
                continue
            else:
                print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt < retries:
                    time.sleep(retry_delay)
                continue
        except Exception as exc:
            print(f"  request error: {exc}")
            if attempt < retries:
                time.sleep(retry_delay)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Auto-label E10 error analysis samples.")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to error_analysis_samples.json from prepare_e10_error_analysis.py")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Path to save the labeled JSON")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path to save the labeled CSV")
    parser.add_argument("--api_base", type=str, default="https://api.deepseek.com",
                        help="DeepSeek API base URL")
    parser.add_argument("--api_key", type=str, required=True,
                        help="API key")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="Model name")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between requests in seconds")
    parser.add_argument("--max_consecutive_failures", type=int, default=10,
                        help="Auto-exit after N consecutive failed requests")
    parser.add_argument("--max_total_minutes", type=float, default=20,
                        help="Auto-exit after N minutes total elapsed")
    parser.add_argument("--request_timeout", type=float, default=15.0,
                        help="Per-request read timeout in seconds")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already labeled samples in output_json")
    args = parser.parse_args()

    samples: List[Dict] = json.loads(Path(args.input_json).read_text(encoding="utf-8"))

    done_labels: Dict[int, str] = {}
    if args.resume and Path(args.output_json).exists():
        existing = json.loads(Path(args.output_json).read_text(encoding="utf-8"))
        done_labels = {
            s["sample_id"]: s["manual_error_tags"]
            for s in existing
            if s.get("manual_error_tags")
        }
        print(f"Resuming: {len(done_labels)} samples already labeled")

    total = len(samples)
    consecutive_failures = 0
    start_time = time.time()
    max_seconds = args.max_total_minutes * 60

    for idx, sample in enumerate(samples, start=1):
        sid = sample["sample_id"]
        if sid in done_labels:
            sample["manual_error_tags"] = done_labels[sid]
            consecutive_failures = 0
            continue

        elapsed = time.time() - start_time
        if elapsed > max_seconds:
            print(f"\nTotal time {elapsed:.0f}s exceeded limit {max_seconds:.0f}s, exiting.")
            break

        print(f"[{idx}/{total}] sample_id={sid}  question={sample['question'][:60]}...")
        label = classify_sample(sample, api_base=args.api_base, api_key=args.api_key, model=args.model,
                                request_timeout=args.request_timeout)
        if label is None:
            consecutive_failures += 1
            print(f"  -> FAILED (consecutive: {consecutive_failures}/{args.max_consecutive_failures})")
        else:
            consecutive_failures = 0
            print(f"  -> {label}")
        sample["manual_error_tags"] = label or ""
        sample["notes"] = sample.get("notes", "")

        if consecutive_failures >= args.max_consecutive_failures:
            print(f"\n{consecutive_failures} consecutive failures, auto-exiting.")
            break

        if idx % 10 == 0 or idx == total:
            Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output_json).write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

        time.sleep(args.delay)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_csv:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(samples[0].keys()) if samples else []
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(samples)
        print(f"CSV saved to {csv_path}")

    summary: Dict[str, int] = {}
    for sample in samples:
        tag = sample.get("manual_error_tags", "") or "unlabeled"
        if tag not in ERROR_LABELS:
            tag = "unlabeled"
        summary[tag] = summary.get(tag, 0) + 1

    print("\n=== Label Distribution ===")
    for label in ERROR_LABELS:
        print(f"  {label}: {summary.get(label, 0)}")
    print(f"  unlabeled: {summary.get('unlabeled', 0)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
