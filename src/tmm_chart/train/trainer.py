from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainingArguments

from ..utils.common import ensure_dir, read_jsonl, set_global_seed, write_json


@dataclass
class TrainingArtifact:
    stage_name: str
    output_dir: Path
    sample_count: int


class VisionLanguageJsonlDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class VisionLanguageCollator:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = []
        images = []
        for item in features:
            image = Image.open(item["image_path"]).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image_path"]},
                        {"type": "text", "text": item["prompt"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": item["target"]}]},
            ]
            if hasattr(self.processor, "apply_chat_template"):
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                text = f"<image>\nUser: {item['prompt']}\nAssistant: {item['target']}"
            texts.append(text)
            images.append(image)

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def load_model_and_processor(model_cfg: dict[str, Any]) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(
        model_cfg["processor_name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_cfg["name_or_path"],
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        torch_dtype=_dtype_from_name(model_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=model_cfg.get("attn_implementation"),
    )
    return model, processor


def apply_lora(model: Any, lora_cfg: dict[str, Any]) -> Any:
    peft_cfg = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_cfg)


def build_training_arguments(
    stage_name: str,
    output_dir: Path,
    training_cfg: dict[str, Any],
    stage_cfg: dict[str, Any],
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=max(1, training_cfg["effective_batch_size"] // training_cfg["gradient_accumulation_steps"]),
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        num_train_epochs=training_cfg["num_train_epochs"],
        learning_rate=stage_cfg["learning_rate"],
        warmup_ratio=training_cfg["warmup_ratio"],
        weight_decay=training_cfg["weight_decay"],
        logging_steps=training_cfg["logging_steps"],
        save_strategy=training_cfg["save_strategy"],
        eval_strategy=training_cfg["eval_strategy"],
        bf16=training_cfg.get("bf16", False),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        remove_unused_columns=False,
        report_to=[],
        seed=training_cfg["seed"],
        dataloader_pin_memory=False,
        save_total_limit=2,
        run_name=stage_name,
    )


def _load_stage_records(manifest_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records = read_jsonl(manifest_path)
    return records[:limit] if limit else records


def train_single_stage(
    stage_name: str,
    manifest_path: Path,
    config: dict[str, Any],
    output_root: Path,
    base_adapter_path: Path | None = None,
    limit: int | None = None,
) -> TrainingArtifact:
    set_global_seed(config["training"]["seed"])
    model, processor = load_model_and_processor(config["model"])
    if base_adapter_path:
        model = PeftModel.from_pretrained(model, str(base_adapter_path), is_trainable=True)
    else:
        model = apply_lora(model, config["training"]["lora"])

    records = _load_stage_records(manifest_path, limit=limit)
    dataset = VisionLanguageJsonlDataset(records)
    output_dir = ensure_dir(output_root / stage_name)

    trainer = Trainer(
        model=model,
        args=build_training_arguments(
            stage_name=stage_name,
            output_dir=output_dir,
            training_cfg=config["training"],
            stage_cfg=config["training"]["stages"].get(stage_name, config["training"]["stages"]["stage2_basic_vqa"]),
        ),
        train_dataset=dataset,
        data_collator=VisionLanguageCollator(processor),
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    write_json(
        output_dir / "training_summary.json",
        {"stage_name": stage_name, "sample_count": len(records), "manifest": str(manifest_path)},
    )
    return TrainingArtifact(stage_name=stage_name, output_dir=output_dir, sample_count=len(records))


def train_standard_sft(config: dict[str, Any], root_dir: Path, limit: int | None = None) -> TrainingArtifact:
    manifests_dir = root_dir / "data" / "synthetic" / "manifests"
    combined = []
    for stage_name in config["training"]["stages"]:
        combined.extend(_load_stage_records(manifests_dir / f"{stage_name}.jsonl"))
    if limit:
        combined = combined[:limit]
    temp_manifest = manifests_dir / "_standard_sft_combined.jsonl"
    from ..utils.common import write_jsonl

    write_jsonl(temp_manifest, combined)
    return train_single_stage(
        stage_name="standard_sft",
        manifest_path=temp_manifest,
        config=config,
        output_root=root_dir / "outputs" / "checkpoints",
        limit=None,
    )


def train_curriculum(config: dict[str, Any], root_dir: Path, limit: int | None = None) -> list[TrainingArtifact]:
    manifests_dir = root_dir / "data" / "synthetic" / "manifests"
    checkpoint_root = root_dir / "outputs" / "checkpoints"
    artifacts: list[TrainingArtifact] = []
    previous_adapter: Path | None = None

    for stage_name in config["training"]["stages"]:
        artifact = train_single_stage(
            stage_name=stage_name,
            manifest_path=manifests_dir / f"{stage_name}.jsonl",
            config=config,
            output_root=checkpoint_root,
            base_adapter_path=previous_adapter,
            limit=limit,
        )
        artifacts.append(artifact)
        previous_adapter = artifact.output_dir
    return artifacts
