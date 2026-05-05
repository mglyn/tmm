#!/usr/bin/env python3
"""Collect predictions for one stage API over a local ChartQA split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_from_disk

from build_router_dataset import (
    StageClient,
    encode_image_to_base64,
    ensure_stage_health,
    query_stage,
    write_json,
    write_jsonl,
)


def log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect single-stage predictions for router data building.")
    parser.add_argument("--stage", type=str, required=True, help="Stage name, e.g. stage2")
    parser.add_argument("--api_base", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="default")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    client = StageClient(name=args.stage, api_base=args.api_base.rstrip("/"), model_name=args.model_name)
    log(f"[collect] checking health stage={args.stage} api_base={args.api_base}")
    ensure_stage_health([client], timeout=args.timeout)

    dataset = load_from_disk(args.dataset_path)[args.split]
    if args.start_index:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.sample_limit is not None:
        dataset = dataset.select(range(min(args.sample_limit, len(dataset))))
    total_samples = len(dataset)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    log(
        f"[collect] stage={args.stage} split={args.split} total_samples={total_samples} "
        f"output_file={output_path}"
    )
    records = []
    with output_path.open("a", encoding="utf-8") as handle:
        for relative_idx, sample in enumerate(dataset):
            sample_id = args.start_index + relative_idx
            question = sample["query"]
            gold_answers = sample["label"] if isinstance(sample["label"], list) else [sample["label"]]
            image_b64 = encode_image_to_base64(sample["image"])
            prediction = query_stage(
                client=client,
                image_b64=image_b64,
                question=question,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            record = {
                "sample_id": sample_id,
                "question": question,
                "gold_answers": gold_answers,
                "prediction": prediction,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

            done = relative_idx + 1
            if done == 1 or done == total_samples or (args.log_every > 0 and done % args.log_every == 0):
                log(
                    f"[collect] stage={args.stage} progress={done}/{total_samples} "
                    f"sample_id={sample_id}"
                )

    write_json(
        output_path.with_suffix(".summary.json"),
        {
            "stage": args.stage,
            "api_base": args.api_base,
            "model_name": args.model_name,
            "dataset_path": args.dataset_path,
            "split": args.split,
            "num_samples": len(records),
        },
    )
    log(
        json.dumps(
            {"stage": args.stage, "num_samples": len(records), "output_file": str(output_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
