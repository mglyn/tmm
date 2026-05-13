#!/usr/bin/env python3
"""Extract pooled multimodal router features from a frozen Qwen2.5-VL backbone.

This script aligns with the paper-level SAR design:
1. take a chart-question pair (image, question)
2. run the frozen multimodal backbone
3. pool the final hidden states into h(I, q)
4. save features for lightweight router classifier training
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image


# Work around broken optional scientific stack packages in the current
# environment. Transformers may import them transitively for unrelated features,
# but Qwen2.5-VL feature extraction here does not rely on them.
_original_find_spec = importlib.util.find_spec


def _patched_find_spec(name, package=None):
    blocked_prefixes = ("scipy", "sklearn")
    if any(str(name) == prefix or str(name).startswith(prefix + ".") for prefix in blocked_prefixes):
        return None
    return _original_find_spec(name, package)


importlib.util.find_spec = _patched_find_spec

try:
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
except ImportError:  # pragma: no cover - fallback for older installs
    from transformers import AutoProcessor, AutoModelForImageTextToText

    Qwen2_5_VLForConditionalGeneration = None


def read_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model(model_path: str, dtype: torch.dtype, device_map: str | None):
    if Qwen2_5_VLForConditionalGeneration is not None:
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    return AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )


def build_inputs(processor, images: Sequence[Image.Image], questions: Sequence[str], device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        texts: List[str] = []
        for question in questions:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            texts.append(text)

        batch = processor(
            text=texts,
            images=list(images),
            padding=True,
            return_tensors=None,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to build multimodal inputs with the current processor. "
            "Please verify the installed transformers version supports Qwen2.5-VL."
        ) from exc

    converted: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value is None:
            continue
        if torch.is_tensor(value):
            tensor = value
        elif isinstance(value, np.ndarray):
            tensor = torch.tensor(value.tolist())
        elif isinstance(value, list):
            tensor = torch.tensor(value)
        else:
            tensor = torch.as_tensor(value)

        if key in {"pixel_values", "pixel_values_videos"}:
            tensor = tensor.to(device=device, dtype=torch.float32)
        else:
            tensor = tensor.to(device)
        converted[key] = tensor

    return converted


def pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract pooled multimodal features for formal SAR training.")
    parser.add_argument("--router_file", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, required=True, choices=["train", "val", "test"])
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/data/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--device_map", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=256)
    args = parser.parse_args()

    start_time = time.time()
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    router_records = read_jsonl(Path(args.router_file))
    if args.max_samples is not None:
        router_records = router_records[: args.max_samples]

    dataset = load_from_disk(args.dataset_path)[args.dataset_split]
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)
    model = load_model(args.model_path, dtype=dtype, device_map=args.device_map)
    model.eval()
    if args.device_map is None:
        model.to(device)

    all_features: List[torch.Tensor] = []
    all_labels: List[int] = []
    all_sample_ids: List[int] = []
    all_questions: List[str] = []

    for start_idx in range(0, len(router_records), args.batch_size):
        batch_records = router_records[start_idx : start_idx + args.batch_size]
        batch_feature_rows: List[torch.Tensor] = []

        for record in batch_records:
            sample_id = int(record["sample_id"])
            sample = dataset[sample_id]
            image = sample["image"]
            question = record["question"]
            label = int(record["oracle_stage_id"])

            with torch.no_grad():
                batch_inputs = build_inputs(processor, images=[image], questions=[question], device=device)
                outputs = model(**batch_inputs, output_hidden_states=True, return_dict=True)

                hidden_states = outputs.hidden_states[-1]
                attention_mask = batch_inputs["attention_mask"]
                pooled = pool_hidden_states(hidden_states, attention_mask).detach().cpu().to(torch.float16)

            batch_feature_rows.append(pooled)
            all_labels.append(label)
            all_sample_ids.append(sample_id)
            all_questions.append(question)

        all_features.append(torch.cat(batch_feature_rows, dim=0))

        done = start_idx + len(batch_records)
        if done == len(router_records) or (args.save_every > 0 and done % args.save_every == 0):
            feature_tensor = torch.cat(all_features, dim=0)
            payload = {
                "features": feature_tensor,
                "labels": torch.tensor(all_labels, dtype=torch.long),
                "sample_ids": torch.tensor(all_sample_ids, dtype=torch.long),
                "questions": all_questions,
                "stage_names": ["stage2", "stage3", "stage4", "stage5"],
                "router_file": args.router_file,
                "dataset_path": args.dataset_path,
                "dataset_split": args.dataset_split,
                "model_path": args.model_path,
                "pooling": "mean_last_hidden_state_over_attention_mask",
            }
            torch.save(payload, output_path)
            save_json(
                output_path.with_suffix(".summary.json"),
                {
                    "num_samples": len(all_labels),
                    "feature_dim": int(feature_tensor.shape[1]),
                    "dataset_split": args.dataset_split,
                    "elapsed_seconds": round(time.time() - start_time, 2),
                    "output_file": str(output_path),
                },
            )
            print(
                json.dumps(
                    {
                        "done": done,
                        "total": len(router_records),
                        "feature_dim": int(feature_tensor.shape[1]),
                        "elapsed_seconds": round(time.time() - start_time, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
