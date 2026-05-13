#!/usr/bin/env python3
"""E02 cross-dataset evaluation: PlotQA + FigureQA with SAR router.

Requires:
  - PlotQA at /root/autodl-tmp/data/datasets/plotqa (jinaai/plotqa format)
  - FigureQA at /root/autodl-tmp/data/datasets/figureqa (vikhyatk/figureqa format)
  - 4 stage APIs running on ports 8002-8005 (or reuse cached predictions)
  - Router checkpoint

Usage:
  python scripts/eval_e02_cross_dataset.py --dataset plotqa --output_dir /root/autodl-tmp/data/router_runs/e02_plotqa
"""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import requests
from datasets import load_from_disk
from PIL import Image
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

from build_router_dataset import is_answer_correct, DEFAULT_STAGE_ORDER

STAGE_PORTS = {"stage2": 8002, "stage3": 8003, "stage4": 8004, "stage5": 8005}


def query_stage_api(stage: str, image: Image.Image, question: str, max_tokens: int = 128, timeout: int = 120) -> str:
    port = STAGE_PORTS[stage]
    import base64, io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": question},
        ]}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    resp = requests.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=payload, timeout=timeout)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"].strip()
    return ""


def load_dataset_and_normalize(dataset_path: str, dataset_name: str, sample_limit: Optional[int] = None) -> List[Dict]:
    ds = load_from_disk(dataset_path)
    records = []
    split_name = "test" if "test" in ds else list(ds.keys())[0]
    split = ds[split_name]

    for idx in range(len(split)):
        row = split[idx]
        if dataset_name == "plotqa":
            image = row["image"]
            question = row["query"]
            gold = row["answer"] if isinstance(row["answer"], list) else [str(row["answer"])]
        elif dataset_name == "figureqa":
            img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            image = Image.open(__import__('io').BytesIO(img_bytes)) if isinstance(img_bytes, bytes) else img_bytes
            qa_pairs = row["qa"]
            for qa in qa_pairs:
                records.append({
                    "sample_id": len(records),
                    "image": image,
                    "question": qa["question"],
                    "gold_answers": [qa["answer"]],
                })
            continue
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        records.append({
            "sample_id": idx,
            "image": image,
            "question": question,
            "gold_answers": gold,
        })

    if sample_limit:
        records = records[:sample_limit]
    return records


def load_router(checkpoint_path: str, device: str = "cuda"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    hidden_dims = config.get("hidden_dims")
    if hidden_dims is None:
        hidden_dims = [config["hidden_dim"]] * config.get("num_hidden_layers", 1)

    layers = []
    in_dim = config["input_dim"]
    for hd in hidden_dims:
        layers.append(nn.Linear(in_dim, hd))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.get("dropout", 0.1)))
        in_dim = hd
    layers.append(nn.Linear(in_dim, config["num_classes"]))

    router = nn.Sequential(*layers)
    router.load_state_dict(checkpoint["model_state_dict"])
    router.to(device)
    router.eval()
    return router, config


def eval_cross_dataset(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[E02] Loading {args.dataset} from {args.dataset_path} ...")
    records = load_dataset_and_normalize(args.dataset_path, args.dataset, sample_limit=args.sample_limit)
    total = len(records)
    print(f"[E02] {total} samples ready")

    print(f"[E02] Loading router from {args.router_checkpoint} ...")
    router, config = load_router(args.router_checkpoint, device=args.device)
    input_dim = config["input_dim"]

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model_path = config.get("model_path", "/root/autodl-tmp/data/models/Qwen2.5-VL-7B-Instruct")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )
    backbone.eval()

    stage_usage = {s: 0 for s in DEFAULT_STAGE_ORDER}
    correct = 0
    predictions = []

    start_time = time.time()

    for idx, record in enumerate(records):
        # Extract multimodal feature
        image = record["image"]
        question = record["question"]
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, images=image, return_tensors="pt", padding=True).to(args.device)

        with torch.no_grad():
            outputs = backbone(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            attn_mask = inputs.attention_mask
            mask = attn_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            pooled = pooled.to(torch.float32)

        logits = router(pooled)
        probs = torch.softmax(logits, dim=-1)[0]

        # Apply routing policy
        pred_id = int(torch.argmax(probs).item())
        raw_stage = DEFAULT_STAGE_ORDER[pred_id]
        prob_dict = {s: float(probs[i]) for i, s in enumerate(DEFAULT_STAGE_ORDER)}

        if args.routing_policy == "stage2_fallback" and raw_stage != "stage2" and prob_dict["stage2"] >= args.stage2_abs_threshold:
            selected = "stage2"
        elif args.routing_policy == "top1":
            selected = raw_stage
        else:
            selected = raw_stage

        stage_usage[selected] += 1

        # Query the selected stage API
        prediction = query_stage_api(selected, image, question, max_tokens=args.max_tokens, timeout=args.timeout)
        is_correct = is_answer_correct(prediction, record["gold_answers"])
        correct += int(is_correct)

        predictions.append({
            "sample_id": record["sample_id"],
            "question": question,
            "gold_answers": record["gold_answers"],
            "predicted_stage": selected,
            "prediction": prediction,
            "correct": is_correct,
        })

        done = idx + 1
        if done == 1 or done == total or (args.log_every > 0 and done % args.log_every == 0):
            acc = correct / done
            elapsed = time.time() - start_time
            print(json.dumps({"done": done, "total": total, "accuracy": round(acc, 4), "elapsed_seconds": round(elapsed, 1)}, ensure_ascii=False))

    accuracy = correct / total
    summary = {
        "dataset": args.dataset,
        "num_samples": total,
        "num_correct": correct,
        "accuracy": round(accuracy, 4),
        "accuracy_percent": round(accuracy * 100, 2),
        "elapsed_seconds": round(time.time() - start_time, 1),
        "routing_policy": args.routing_policy,
        "stage2_abs_threshold": args.stage2_abs_threshold,
        "stage_usage": {s: stage_usage[s] for s in DEFAULT_STAGE_ORDER},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "predictions.jsonl").write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in predictions), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E02 cross-dataset evaluation")
    parser.add_argument("--dataset", type=str, required=True, choices=["plotqa", "figureqa"])
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--router_checkpoint", type=str,
                        default="/root/autodl-tmp/data/router_runs/formal_sar_tune_nowt_ls005/checkpoints/best.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--routing_policy", type=str, default="stage2_fallback")
    parser.add_argument("--stage2_abs_threshold", type=float, default=0.20)
    args = parser.parse_args()
    raise SystemExit(eval_cross_dataset(args))
