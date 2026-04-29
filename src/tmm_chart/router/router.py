from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoModelForVision2Seq, AutoProcessor

from ..utils.common import read_jsonl, set_global_seed, write_json


class RouterDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], stage_to_index: dict[str, int]) -> None:
        self.records = records
        self.stage_to_index = stage_to_index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.records[index]
        return {
            "image_path": item["image_path"],
            "question": item["question"],
            "label": self.stage_to_index[item["oracle_stage"]],
        }


class StageAwareRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


@dataclass
class RouterArtifacts:
    model_path: Path
    metrics_path: Path


def _pool_hidden_state(model: Any, processor: Any, image_path: str, question: str, device: torch.device) -> torch.Tensor:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    return hidden.mean(dim=1).squeeze(0).cpu()


def train_router(
    config: dict[str, Any],
    root_dir: Path,
    limit: int | None = None,
) -> RouterArtifacts:
    set_global_seed(config["training"]["seed"])
    router_cfg = config["router"]
    records = read_jsonl(root_dir / "data" / "synthetic" / "manifests" / "router_holdout.jsonl")
    if limit:
        records = records[:limit]

    stage_to_index = {name: idx for idx, name in enumerate(router_cfg["deployable_stages"])}
    dataset = RouterDataset(records, stage_to_index)

    processor = AutoProcessor.from_pretrained(
        config["model"]["processor_name_or_path"],
        trust_remote_code=config["model"].get("trust_remote_code", True),
    )
    backbone = AutoModelForVision2Seq.from_pretrained(
        config["model"]["name_or_path"],
        trust_remote_code=config["model"].get("trust_remote_code", True),
        torch_dtype=torch.bfloat16,
    )
    backbone.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device)

    embeddings = []
    labels = []
    for item in dataset:
        embeddings.append(_pool_hidden_state(backbone, processor, item["image_path"], item["question"], device))
        labels.append(item["label"])
    features = torch.stack(embeddings)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    router = StageAwareRouter(
        input_dim=features.shape[-1],
        hidden_dim=router_cfg["hidden_dim"],
        num_classes=len(stage_to_index),
        dropout=router_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(router.parameters(), lr=router_cfg["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    router.train()
    for _ in range(router_cfg["num_epochs"]):
        optimizer.zero_grad()
        logits = router(features.to(device))
        loss = criterion(logits, labels_tensor.to(device))
        loss.backward()
        optimizer.step()

    router.eval()
    with torch.no_grad():
        logits = router(features.to(device))
        predictions = logits.argmax(dim=-1).cpu().tolist()

    accuracy = accuracy_score(labels, predictions)
    output_dir = root_dir / "outputs" / "router"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "stage_aware_router.pt"
    metrics_path = output_dir / "metrics.json"
    torch.save(
        {
            "state_dict": router.state_dict(),
            "stage_to_index": stage_to_index,
            "input_dim": features.shape[-1],
            "hidden_dim": router_cfg["hidden_dim"],
            "dropout": router_cfg["dropout"],
        },
        model_path,
    )
    write_json(metrics_path, {"accuracy": accuracy, "num_samples": len(labels)})
    return RouterArtifacts(model_path=model_path, metrics_path=metrics_path)


def load_router_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[StageAwareRouter, dict[str, int]]:
    payload = torch.load(checkpoint_path, map_location=device)
    router = StageAwareRouter(
        input_dim=payload["input_dim"],
        hidden_dim=payload["hidden_dim"],
        num_classes=len(payload["stage_to_index"]),
        dropout=payload["dropout"],
    ).to(device)
    router.load_state_dict(payload["state_dict"])
    router.eval()
    return router, payload["stage_to_index"]


def predict_stage(
    router: StageAwareRouter,
    stage_to_index: dict[str, int],
    backbone: Any,
    processor: Any,
    image_path: str,
    question: str,
    device: torch.device,
) -> str:
    embedding = _pool_hidden_state(backbone, processor, image_path, question, device).to(device)
    with torch.no_grad():
        logits = router(embedding.unsqueeze(0))
        prediction = logits.argmax(dim=-1).item()
    index_to_stage = {index: stage for stage, index in stage_to_index.items()}
    return index_to_stage[prediction]
