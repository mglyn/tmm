#!/usr/bin/env python3
"""Train the formal SAR classifier on extracted multimodal features."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


STAGE_NAMES = ["stage2", "stage3", "stage4", "stage5"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Optional[Path], payload: Dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


class RouterBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float, residual: bool):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.use_residual = residual and in_dim == out_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(features)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        if self.use_residual:
            hidden = hidden + features
        return hidden


class MultimodalRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float, num_labels: int, residual: bool = False):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)
        self.input_norm = nn.LayerNorm(input_dim)
        self.blocks = nn.ModuleList(
            [
                RouterBlock(in_dim=dims[idx], out_dim=dims[idx + 1], dropout=dropout, residual=residual)
                for idx in range(len(dims) - 1)
            ]
        )
        final_dim = dims[-1] if hidden_dims else input_dim
        self.output = nn.Linear(final_dim, num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_norm(features)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


@dataclass
class Metrics:
    loss: float
    accuracy: float
    macro_f1: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: List[List[int]]
    prediction_distribution: Dict[str, float]


def confusion_from_predictions(predictions: Sequence[int], labels: Sequence[int], num_labels: int) -> List[List[int]]:
    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    for gold, pred in zip(labels, predictions):
        matrix[gold][pred] += 1
    return matrix


def macro_f1_from_confusion(matrix: Sequence[Sequence[int]]) -> float:
    f1_scores: List[float] = []
    num_labels = len(matrix)
    for class_id in range(num_labels):
        tp = matrix[class_id][class_id]
        fp = sum(matrix[row][class_id] for row in range(num_labels) if row != class_id)
        fn = sum(matrix[class_id][col] for col in range(num_labels) if col != class_id)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(f1_scores) / max(len(f1_scores), 1)


def per_class_accuracy_from_confusion(matrix: Sequence[Sequence[int]]) -> Dict[str, float]:
    return {
        stage_name: (matrix[class_id][class_id] / sum(matrix[class_id])) if sum(matrix[class_id]) > 0 else 0.0
        for class_id, stage_name in enumerate(STAGE_NAMES)
    }


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    predictions: List[int] = []
    labels_list: List[int] = []

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_examples += batch_size
            predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

    matrix = confusion_from_predictions(predictions, labels_list, len(STAGE_NAMES))
    accuracy = sum(matrix[i][i] for i in range(len(STAGE_NAMES))) / max(total_examples, 1)
    prediction_counter = Counter(predictions)
    prediction_distribution = {
        stage_name: prediction_counter.get(class_id, 0) / max(total_examples, 1)
        for class_id, stage_name in enumerate(STAGE_NAMES)
    }
    return Metrics(
        loss=(total_loss / max(total_examples, 1)),
        accuracy=accuracy,
        macro_f1=macro_f1_from_confusion(matrix),
        per_class_accuracy=per_class_accuracy_from_confusion(matrix),
        confusion_matrix=matrix,
        prediction_distribution=prediction_distribution,
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def compute_class_weights(labels: torch.Tensor, mode: str) -> torch.Tensor:
    counts = Counter(labels.tolist())
    total = int(labels.numel())
    weights: List[float] = []
    for class_id in range(len(STAGE_NAMES)):
        count = counts.get(class_id, 1)
        balanced = total / (len(STAGE_NAMES) * count)
        if mode == "balanced":
            weight = balanced
        elif mode == "sqrt_balanced":
            weight = math.sqrt(balanced)
        elif mode == "none":
            weight = 1.0
        else:
            raise ValueError(f"Unsupported class_weight_mode: {mode}")
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32)


def parse_hidden_dims(args: argparse.Namespace) -> List[int]:
    if args.hidden_dims:
        return [int(part.strip()) for part in args.hidden_dims.split(",") if part.strip()]
    return [args.hidden_dim] * args.num_hidden_layers


def load_feature_file(path: Path):
    payload = torch.load(path, map_location="cpu")
    features = payload["features"].float()
    labels = payload["labels"].long()
    return payload, features, labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the formal multimodal SAR classifier.")
    parser.add_argument("--train_features", type=str, required=True)
    parser.add_argument("--val_features", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--hidden_dims", type=str, default=None, help="Comma-separated hidden dims, e.g. 1536,768")
    parser.add_argument("--num_hidden_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--class_weight_mode", type=str, default="balanced", choices=["none", "balanced", "sqrt_balanced"])
    parser.add_argument("--selection_metric", type=str, default="macro_f1", choices=["macro_f1", "accuracy"])
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args.log_file) if args.log_file else output_dir / "train.log"

    train_payload, train_features, train_labels = load_feature_file(Path(args.train_features))
    val_payload, val_features, val_labels = load_feature_file(Path(args.val_features))
    input_dim = int(train_features.shape[1])
    hidden_dims = parse_hidden_dims(args)

    train_loader = DataLoader(
        FeatureDataset(train_features, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        FeatureDataset(val_features, val_labels),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device(args.device)
    model = MultimodalRouter(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        num_labels=len(STAGE_NAMES),
        residual=args.residual,
    ).to(device)

    class_weights = compute_class_weights(train_labels, mode=args.class_weight_mode).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=None if args.class_weight_mode == "none" else class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    config_payload = vars(args).copy()
    config_payload["input_dim"] = input_dim
    config_payload["hidden_dims_resolved"] = hidden_dims
    config_payload["class_weights"] = class_weights.detach().cpu().tolist()
    config_payload["stage_names"] = STAGE_NAMES
    config_payload["train_feature_source"] = train_payload.get("router_file")
    config_payload["val_feature_source"] = val_payload.get("router_file")
    config_payload["log_file"] = str(log_file)
    save_json(output_dir / "config.json", config_payload)
    save_json(log_file.with_suffix(".meta.json"), {"config": config_payload})

    best_selection_score = -math.inf
    history: List[Dict] = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_payload = {
            "epoch": epoch,
            "train_loss_epoch": train_loss,
            "train_metrics": asdict(train_metrics),
            "val_metrics": asdict(val_metrics),
            "epoch_seconds": round(time.time() - epoch_start, 2),
        }
        history.append(epoch_payload)
        save_json(output_dir / "history.json", {"epochs": history})

        epoch_log = {
            "epoch": epoch,
            "train_acc": round(train_metrics.accuracy, 4),
            "train_f1": round(train_metrics.macro_f1, 4),
            "val_acc": round(val_metrics.accuracy, 4),
            "val_f1": round(val_metrics.macro_f1, 4),
            "val_pred_dist": {k: round(v, 4) for k, v in val_metrics.prediction_distribution.items()},
            "epoch_seconds": round(time.time() - epoch_start, 2),
        }
        print(json.dumps(epoch_log, ensure_ascii=False), flush=True)
        append_jsonl(log_file, epoch_log)

        selection_score = val_metrics.macro_f1 if args.selection_metric == "macro_f1" else val_metrics.accuracy
        if selection_score > best_selection_score:
            best_selection_score = selection_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_selection_score": best_selection_score,
                    "input_dim": input_dim,
                    "stage_names": STAGE_NAMES,
                    "config": config_payload,
                },
                output_dir / "best.pt",
            )
            save_json(
                output_dir / "best_metrics.json",
                {
                    "best_epoch": epoch,
                    "selection_metric": args.selection_metric,
                    "best_selection_score": best_selection_score,
                    "best_val_metrics": asdict(val_metrics),
                    "elapsed_seconds": round(time.time() - start_time, 2),
                },
            )

    save_json(
        output_dir / "final_summary.json",
        {
            "elapsed_seconds": round(time.time() - start_time, 2),
            "num_train_samples": int(train_labels.numel()),
            "num_val_samples": int(val_labels.numel()),
            "input_dim": input_dim,
            "selection_metric": args.selection_metric,
            "best_selection_score": best_selection_score,
        },
    )
    append_jsonl(
        log_file,
        {
            "event": "training_complete",
            "elapsed_seconds": round(time.time() - start_time, 2),
            "selection_metric": args.selection_metric,
            "best_selection_score": round(best_selection_score, 4),
            "best_metrics_file": str(output_dir / "best_metrics.json"),
            "history_file": str(output_dir / "history.json"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
