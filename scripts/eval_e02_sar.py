#!/usr/bin/env python3
"""E02 cross-dataset SAR evaluation with router-selected stage adapters."""

from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_router_dataset import is_answer_correct, DEFAULT_STAGE_ORDER


DEFAULT_ADAPTER_PATHS = {
    "stage2": "/root/autodl-tmp/data/legacy/chart_vqa_synthesis_1/outputs/models/stage2_basic_vqa",
    "stage3": "/root/autodl-tmp/data/legacy/chart_vqa_synthesis_1/outputs/models/stage3_reasoning_vqa",
    "stage4": "/root/autodl-tmp/data/legacy/chart_vqa_synthesis_1/outputs/models/stage4_visual_analysis",
    "stage5": "/root/autodl-tmp/data/legacy/chart_vqa_synthesis_1/outputs/models/stage5_code_generation",
}


def pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


def load_router(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]
    num_labels = len(config.get("stage_names", 4))
    hidden_dims = config.get("hidden_dims_resolved") or config.get("hidden_dims") or [config.get("hidden_dim", 1024)]
    input_dim = config["input_dim"]

    class RouterBlock(nn.Module):
        def __init__(self, in_d, out_d, dp, res):
            super().__init__()
            self.linear = nn.Linear(in_d, out_d)
            self.activation = nn.GELU()
            self.dropout = nn.Dropout(dp)
            self.use_residual = res and in_d == out_d
        def forward(self, x):
            h = self.dropout(self.activation(self.linear(x)))
            return h + x if self.use_residual else h

    class MultimodalRouter(nn.Module):
        def __init__(self):
            super().__init__()
            dims = [input_dim] + list(hidden_dims)
            self.input_norm = nn.LayerNorm(input_dim)
            self.blocks = nn.ModuleList([RouterBlock(dims[i], dims[i+1], config.get("dropout", 0.1), config.get("residual", False)) for i in range(len(dims)-1)])
            self.output = nn.Linear(dims[-1] if hidden_dims else input_dim, num_labels)
        def forward(self, x):
            h = self.input_norm(x)
            for b in self.blocks:
                h = b(h)
            return self.output(h)

    router = MultimodalRouter()
    router.load_state_dict(checkpoint["model_state_dict"])
    router.to(device)
    router.eval()
    return router


def load_existing_predictions(pred_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not pred_path.exists() or pred_path.stat().st_size == 0:
        return rows
    cleaned_lines: List[str] = []
    encountered_invalid = False
    with pred_path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.replace(b"\x00", b"").decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
                cleaned_lines.append(line)
            except json.JSONDecodeError:
                encountered_invalid = True
                break
    if encountered_invalid:
        with pred_path.open("w", encoding="utf-8") as handle:
            for line in cleaned_lines:
                handle.write(line + "\n")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, choices=["plotqa", "figureqa"])
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="/root/autodl-tmp/data/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--router_checkpoint", type=str,
                        default="/root/autodl-tmp/data/router_runs/formal_sar_tune_nowt_ls005/checkpoints/best.pt")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sample_limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--routing_policy", type=str, default="stage2_fallback")
    parser.add_argument("--stage2_abs_threshold", type=float, default=0.20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--empty_cache_every", type=int, default=1)
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.jsonl"

    print(f"[E02-SAR] Loading backbone: {args.model_path}", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype=dtype, device_map="auto", trust_remote_code=True,
    )
    model.config.output_hidden_states = True
    print(f"[E02-SAR] Loading adapters for stages: {', '.join(DEFAULT_STAGE_ORDER)}", flush=True)
    # Keep LoRA weights in their native low-precision dtype to avoid a large fp32
    # memory spike during adapter loading on limited GPUs.
    model = PeftModel.from_pretrained(
        model,
        DEFAULT_ADAPTER_PATHS["stage2"],
        adapter_name="stage2",
        autocast_adapter_dtype=False,
        low_cpu_mem_usage=True,
    )
    for stage in DEFAULT_STAGE_ORDER[1:]:
        model.load_adapter(
            DEFAULT_ADAPTER_PATHS[stage],
            adapter_name=stage,
            autocast_adapter_dtype=False,
            low_cpu_mem_usage=True,
        )
    model.set_adapter("stage2")
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    print(f"[E02-SAR] Loading router: {args.router_checkpoint}", flush=True)
    router = load_router(args.router_checkpoint, device=device)

    print(f"[E02-SAR] Loading dataset: {args.dataset_path}", flush=True)
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
        records.append({"sample_id": idx, "image": image, "question": question, "gold_answers": gold})

    if args.sample_limit:
        records = records[:args.sample_limit]
    total = len(records)
    print(f"[E02-SAR] {total} records to evaluate", flush=True)

    existing_rows = load_existing_predictions(pred_path) if args.resume else []
    stage_usage = {s: 0 for s in DEFAULT_STAGE_ORDER}
    correct = 0
    resumed_count = 0
    for row in existing_rows:
        correct += int(bool(row.get("correct", False)))
        stage = row.get("selected_stage")
        if stage in stage_usage:
            stage_usage[stage] += 1
        resumed_count += 1

    if resumed_count:
        print(f"[E02-SAR] Resuming from {resumed_count} existing predictions: {pred_path}", flush=True)
    if resumed_count > total:
        raise ValueError(f"Existing predictions exceed dataset size: {resumed_count} > {total}")

    start_time = time.time()

    file_mode = "a" if resumed_count else "w"
    with pred_path.open(file_mode, encoding="utf-8") as handle:
        for idx, rec in enumerate(records[resumed_count:], start=resumed_count):
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

            # `set_adapter()` in PEFT may toggle `requires_grad` internally.
            # Avoid creating inference tensors here so adapter switching after
            # router scoring stays valid when resuming long eval runs.
            with torch.no_grad(), model.disable_adapter():
                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[-1]
                attn_mask = inputs["attention_mask"]
                feat = pool_hidden_states(hidden.to(torch.float32), attn_mask)
                logits = router(feat)
                probs = torch.softmax(logits, dim=-1)[0]
                del outputs, hidden, feat, logits

            pred_id = int(torch.argmax(probs).item())
            raw_stage = DEFAULT_STAGE_ORDER[pred_id]
            prob_dict = {s: round(float(probs[i]), 6) for i, s in enumerate(DEFAULT_STAGE_ORDER)}

            selected_stage = raw_stage
            if args.routing_policy == "stage2_fallback" and raw_stage != "stage2" and prob_dict["stage2"] >= args.stage2_abs_threshold:
                selected_stage = "stage2"

            stage_usage[selected_stage] += 1

            model.set_adapter(selected_stage)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                                                pad_token_id=processor.tokenizer.eos_token_id)
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = generated_ids[:, prompt_len:]
            prediction = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
            del generated_ids, inputs, probs
            if device.startswith("cuda") and args.empty_cache_every > 0 and ((idx + 1) % args.empty_cache_every == 0):
                torch.cuda.empty_cache()

            is_correct = is_answer_correct(prediction, gold)
            correct += int(is_correct)

            row = {"sample_id": rec["sample_id"], "question": question, "gold_answers": gold,
                   "prediction": prediction, "correct": is_correct,
                   "raw_predicted_stage": raw_stage, "selected_stage": selected_stage,
                   "router_probabilities": prob_dict, "routing_policy": args.routing_policy}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

            done = idx + 1
            if done == 1 or done == total or (args.log_every > 0 and done % args.log_every == 0):
                acc = correct / done
                elapsed = time.time() - start_time
                avg = elapsed / done
                print(json.dumps({"done": done, "total": total, "accuracy": round(acc, 4), "stage2_pct": round(stage_usage["stage2"]/done*100,1), "elapsed_seconds": round(elapsed, 1), "avg_seconds": round(avg, 3)}, ensure_ascii=False), flush=True)

    total_elapsed = time.time() - start_time
    accuracy = correct / total
    summary = {
        "dataset": args.dataset_name,
        "method": "SAR",
        "num_samples": total,
        "num_correct": correct,
        "accuracy": round(accuracy, 4),
        "accuracy_percent": round(accuracy * 100, 2),
        "elapsed_seconds": round(total_elapsed, 1),
        "routing_policy": args.routing_policy,
        "stage2_abs_threshold": args.stage2_abs_threshold,
        "stage_usage": {s: stage_usage[s] for s in DEFAULT_STAGE_ORDER},
        "adapter_paths": DEFAULT_ADAPTER_PATHS,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
