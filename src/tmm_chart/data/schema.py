from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ChartRecord:
    chart_id: str
    chart_type: str
    scenario: str
    title: str
    x_label: str
    y_label: str
    categories: list[str]
    values: list[float]
    metadata: dict[str, Any]
    image_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageSample:
    sample_id: str
    chart_id: str
    stage_name: str
    prompt: str
    target: str
    question: str | None = None
    answer: str | None = None
    difficulty: int = 1
    chart_type: str = ""
    scenario: str = ""
    image_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
