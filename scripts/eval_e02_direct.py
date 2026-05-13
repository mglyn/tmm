#!/usr/bin/env python3
"""Direct model inference for E02 cross-dataset evaluation.

Loads Qwen2.5-VL-7B + stage2 LoRA adapter directly (no API needed).
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
from datasets import load_from_disk
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_router_dataset import is_answer_correct


def load_model_and_infer(args: argparse.Namespace) -> int:
    device = args.device
    dtype = torch.bfloat16

    print(f"[E02] Loading base model: {args.model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[E02] Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    print(f"[E02] Loading dataset: {args.dataset_path}")
    ds = load_from_disk(args.dataset_path)
    split_name = "test" if "test" in ds else list(ds.keys())[0]
    split = ds[split_name]

    records = []
    for idx in range(len(split)):
        row = split[idx]
        if args.dataset_name == "plotqa":
            image = row["image"]
            question = row["query"]
            gold = row["answer"] if isinstance(row["answer"], list) else [str(row["answer"])]
        elif args.dataset_name == "figureqa":
            import io as _io
            img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            image = Image.open(_io.BytesIO(img_bytes)) if isinstance(img_bytes, bytes) else img_bytes
            for qa in row["qa"]:
                records.append({"sample_id": len(records), "image": image, "question": qa["question"], "gold_answers": [qa["answer"]]})
            continue
        else:
            raise ValueError(f"Unknown dataset: {args.dataset_name}")
        records.append({"sample_id": idx, "image": image, "question": question, "gold_answers": gold})

    if args.sample_limit:
        records = records[:args.sample_limit]

    total = len(records)
    print(f"[E02] {total} records to evaluate")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    correct = 0
    predictions = []
    start_time = time.time()
    pred_path = output_dir / "predictions.jsonl"

    with pred_path.open("w", encoding="utf-8") as handle:
        for idx, rec in enumerate(records):
            image = rec["image"]
            question = rec["question"]
            gold = rec["gold_answers"]

            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text, images=image, padding=True, return_tensors=None)

            converted = {}
            for key, value in inputs.items():
                if value is None:
                    continue
                tensor = torch.as_tensor(value)
                if key in {"pixel_values", "pixel_values_videos"}:
                    tensor = tensor.to(device=device, dtype=torch.float32)
                else:
                    tensor = tensor.to(device)
                converted[key] = tensor
            inputs = converted

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                                pad_token_id=processor.tokenizer.eos_token_id)
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = generated_ids[:, prompt_len:]
            prediction = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

            is_correct = is_answer_correct(prediction, gold)
            correct += int(is_correct)

            row = {"sample_id": rec["sample_id"], "question": question, "gold_answers": gold,
                   "prediction": prediction, "correct": is_correct}
            predictions.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            done = idx + 1
            if done == 1 or done == total or (args.log_every > 0 and done % args.log_every == 0):
                acc = correct / done
                elapsed = time.time() - start_time
                avg = elapsed / done
                print(json.dumps({"done": done, "total": total, "accuracy": round(acc, 4), "elapsed_seconds": round(elapsed, 1), "avg_seconds": round(avg, 3)}, ensure_ascii=False))

    accuracy = correct / total
    summary = {
        "dataset": args.dataset_name,
        "adapter": args.adapter_name,
        "num_samples": total,
        "num_correct": correct,
        "accuracy": round(accuracy, 4),
        "accuracy_percent": round(accuracy * 100, 2),
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=["plotqa", "figureqa"])
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/data/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--adapter_path", type=str,
                        default="/root/autodl-tmp/data/legacy/chart_vqa_synthesis_1/outputs/models/stage2_basic_vqa")
    parser.add_argument("--adapter_name", type=str, default="stage2")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()
    raise SystemExit(load_model_and_infer(args))
