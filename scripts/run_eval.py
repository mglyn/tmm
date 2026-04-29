from __future__ import annotations

import argparse
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tmm_chart.config import load_experiment_config
from tmm_chart.eval.benchmarks import (
    load_benchmark_samples,
    normalize_answer,
    save_prediction_bundle,
)
from tmm_chart.eval.reporting import export_paper_metrics
from tmm_chart.router.router import load_router_checkpoint, predict_stage


def load_generation_stack(model_cfg: dict[str, Any], adapter_path: Path | None = None) -> tuple[Any, Any, torch.device]:
    processor = AutoProcessor.from_pretrained(
        model_cfg["processor_name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation=model_cfg.get("attn_implementation"),
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, processor, device


def generate_answer(model: Any, processor: Any, device: torch.device, image_path: str, question: str) -> str:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}]
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"<image>\nUser: {question}\nAssistant:"
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=128)
    decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
    if "Assistant:" in decoded:
        decoded = decoded.split("Assistant:")[-1]
    return decoded.strip()


def _strategy_paths(root_dir: Path) -> dict[str, Path | None]:
    checkpoint_root = root_dir / "outputs" / "checkpoints"
    mapping = {
        "zero_shot": None,
        "standard_sft": checkpoint_root / "standard_sft",
        "stage2_basic_vqa": checkpoint_root / "stage2_basic_vqa",
        "stage3_reasoning_vqa": checkpoint_root / "stage3_reasoning_vqa",
        "stage4_visual_analysis": checkpoint_root / "stage4_visual_analysis",
        "stage5_code_generation": checkpoint_root / "stage5_code_generation",
    }
    return {name: path for name, path in mapping.items() if path is None or path.exists()}


def run_fixed_strategy(
    benchmark_name: str,
    strategy_name: str,
    adapter_path: Path | None,
    root_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    samples = load_benchmark_samples(root_dir, benchmark_name)
    model, processor, device = load_generation_stack(config["model"], adapter_path=adapter_path)
    predictions = []
    for sample in samples:
        prediction = generate_answer(model, processor, device, sample.image_path, sample.question)
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "question": sample.question,
                "answer": sample.answer,
                "prediction": prediction,
            }
        )
    return save_prediction_bundle(
        root_dir / "outputs" / "eval",
        benchmark_name,
        strategy_name,
        predictions,
        config["evaluation"]["answer_normalization"],
    )


def _similarity(prediction: str, answer: str) -> float:
    return SequenceMatcher(None, prediction, answer).ratio()


def run_oracle_routing(
    benchmark_name: str,
    stage_paths: dict[str, Path],
    root_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    samples = load_benchmark_samples(root_dir, benchmark_name)
    stacks = {
        stage_name: load_generation_stack(config["model"], adapter_path=adapter_path)
        for stage_name, adapter_path in stage_paths.items()
        if stage_name.startswith("stage")
    }
    norm_cfg = config["evaluation"]["answer_normalization"]
    predictions = []
    for sample in samples:
        best_stage = None
        best_prediction = ""
        best_score = -1.0
        for stage_name, (model, processor, device) in stacks.items():
            prediction = generate_answer(model, processor, device, sample.image_path, sample.question)
            normalized_prediction = normalize_answer(prediction, lowercase=norm_cfg["lowercase"], strip_punctuation=norm_cfg["strip_punctuation"])
            normalized_answer = normalize_answer(sample.answer, lowercase=norm_cfg["lowercase"], strip_punctuation=norm_cfg["strip_punctuation"])
            score = 1.0 if normalized_prediction == normalized_answer else _similarity(normalized_prediction, normalized_answer)
            if score > best_score:
                best_stage = stage_name
                best_prediction = prediction
                best_score = score
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "question": sample.question,
                "answer": sample.answer,
                "prediction": best_prediction,
                "oracle_stage": best_stage,
            }
        )
    return save_prediction_bundle(
        root_dir / "outputs" / "eval",
        benchmark_name,
        "oracle_routing",
        predictions,
        norm_cfg,
    )


def run_predicted_routing(
    benchmark_name: str,
    stage_paths: dict[str, Path],
    root_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    samples = load_benchmark_samples(root_dir, benchmark_name)
    router_ckpt = root_dir / "outputs" / "router" / "stage_aware_router.pt"
    if not router_ckpt.exists():
        raise FileNotFoundError(f"Router checkpoint not found: {router_ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    router, stage_to_index = load_router_checkpoint(router_ckpt, device)
    backbone, processor, backbone_device = load_generation_stack(config["model"])
    stacks = {
        stage_name: load_generation_stack(config["model"], adapter_path=adapter_path)
        for stage_name, adapter_path in stage_paths.items()
        if stage_name in config["router"]["deployable_stages"]
    }

    predictions = []
    for sample in samples:
        predicted_stage = predict_stage(
            router=router,
            stage_to_index=stage_to_index,
            backbone=backbone,
            processor=processor,
            image_path=sample.image_path,
            question=sample.question,
            device=backbone_device,
        )
        model, stage_processor, stage_device = stacks[predicted_stage]
        prediction = generate_answer(model, stage_processor, stage_device, sample.image_path, sample.question)
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "question": sample.question,
                "answer": sample.answer,
                "prediction": prediction,
                "predicted_stage": predicted_stage,
            }
        )
    return save_prediction_bundle(
        root_dir / "outputs" / "eval",
        benchmark_name,
        "predicted_router",
        predictions,
        config["evaluation"]["answer_normalization"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation and aggregate metrics.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    root_dir = config.root_dir
    strategy_paths = _strategy_paths(root_dir)
    if len(strategy_paths) <= 1:
        raise FileNotFoundError("No checkpoints found under outputs/checkpoints.")

    all_metrics = []
    fixed_stage_metrics: dict[str, dict[str, Any]] = {}
    for benchmark_name in config.evaluation["benchmarks"]:
        benchmark_file = root_dir / "data" / "benchmarks" / benchmark_name.lower() / "test.jsonl"
        if not benchmark_file.exists():
            print(f"[skip] Missing benchmark file: {benchmark_file}")
            continue
        for strategy_name, adapter_path in strategy_paths.items():
            metrics = run_fixed_strategy(benchmark_name, strategy_name, adapter_path, root_dir, config.raw)
            all_metrics.append(metrics)
            fixed_stage_metrics.setdefault(benchmark_name, {})[strategy_name] = metrics

        deployable = {k: v for k, v in fixed_stage_metrics[benchmark_name].items() if k.startswith("stage")}
        best_name = max(deployable, key=lambda name: deployable[name]["accuracy"])
        metric = dict(deployable[best_name])
        metric["strategy"] = "best_single_stage"
        metric["selected_stage"] = best_name
        all_metrics.append(metric)
        save_path = root_dir / "outputs" / "eval" / f"{benchmark_name.lower()}_best_single_stage_metrics.json"
        save_path.write_text(json.dumps(metric, indent=2), encoding="utf-8")

        stage_paths = {name: path for name, path in strategy_paths.items() if path is not None}
        all_metrics.append(run_oracle_routing(benchmark_name, stage_paths, root_dir, config.raw))
        all_metrics.append(run_predicted_routing(benchmark_name, stage_paths, root_dir, config.raw))

    paper_metrics_path = export_paper_metrics(root_dir)
    print(json.dumps({"metrics_count": len(all_metrics), "paper_metrics": str(paper_metrics_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
