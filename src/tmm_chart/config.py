from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def project(self) -> dict[str, Any]:
        return self.raw["project"]

    @property
    def dataset(self) -> dict[str, Any]:
        return self.raw["dataset"]

    @property
    def training(self) -> dict[str, Any]:
        return self.raw["training"]

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def router(self) -> dict[str, Any]:
        return self.raw["router"]

    @property
    def evaluation(self) -> dict[str, Any]:
        return self.raw["evaluation"]

    @property
    def root_dir(self) -> Path:
        base = Path(self.raw["project"].get("root_dir", "."))
        return (self.path.parent.parent / base).resolve()

    def resolve_dir(self, key: str) -> Path:
        value = self.raw["project"][key]
        return (self.root_dir / value).resolve()


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig(raw=raw, path=config_path)
