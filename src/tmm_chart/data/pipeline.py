from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .chart_generator import generate_chart_records
from .schema import ChartRecord, StageSample
from .stage_builder import build_stage_samples
from ..utils.common import ensure_dir, write_json, write_jsonl


class DatasetBuilder:
    def __init__(self, config: dict[str, Any], root_dir: Path) -> None:
        self.config = config
        self.root_dir = root_dir
        self.data_root = ensure_dir(root_dir / "data")
        self.synthetic_root = ensure_dir(self.data_root / "synthetic")
        self.charts_root = ensure_dir(self.synthetic_root / "charts")
        self.manifest_root = ensure_dir(self.synthetic_root / "manifests")

    def build(self, limit: int | None = None) -> dict[str, Any]:
        image_cfg = self.config["image"]
        chart_count = limit or self.config["synthetic_chart_count"]
        records = generate_chart_records(
            chart_count=chart_count,
            chart_types=self.config["chart_types"],
            scenarios=self.config["scenarios"],
            output_dir=self.charts_root,
            seed=self.config["seed"],
            width=image_cfg["width"],
            height=image_cfg["height"],
            dpi=image_cfg["dpi"],
        )
        stage_samples = build_stage_samples(records, seed=self.config["seed"])
        router_manifest = self._build_router_split(stage_samples)
        self._persist(records, stage_samples, router_manifest)
        summary = self._summarize(records, stage_samples, router_manifest)
        write_json(self.manifest_root / "summary.json", summary)
        return summary

    def _build_router_split(self, stage_samples: dict[str, list[StageSample]]) -> list[dict[str, Any]]:
        holdout_size = self.config["router_holdout_count"]
        deployable = [
            stage for stage in ["stage2_basic_vqa", "stage3_reasoning_vqa", "stage4_visual_analysis", "stage5_code_generation"]
        ]
        qa_pool: list[StageSample] = []
        for stage_name in deployable:
            qa_pool.extend(stage_samples[stage_name])
        qa_pool = [sample for sample in qa_pool if sample.question]
        qa_pool.sort(key=lambda item: item.sample_id)
        selected = qa_pool[: min(holdout_size, len(qa_pool))]
        payload = []
        for sample in selected:
            payload.append(
                {
                    "sample_id": sample.sample_id,
                    "chart_id": sample.chart_id,
                    "question": sample.question,
                    "answer": sample.answer,
                    "image_path": sample.image_path,
                    "oracle_stage": sample.stage_name,
                }
            )
        write_jsonl(self.manifest_root / "router_holdout.jsonl", payload)
        return payload

    def _persist(
        self,
        records: list[ChartRecord],
        stage_samples: dict[str, list[StageSample]],
        router_manifest: list[dict[str, Any]],
    ) -> None:
        write_jsonl(self.manifest_root / "charts.jsonl", [record.to_dict() for record in records])
        for stage_name, items in stage_samples.items():
            write_jsonl(self.manifest_root / f"{stage_name}.jsonl", [item.to_dict() for item in items])
        write_json(self.manifest_root / "router_holdout_meta.json", {"count": len(router_manifest)})

    def _summarize(
        self,
        records: list[ChartRecord],
        stage_samples: dict[str, list[StageSample]],
        router_manifest: list[dict[str, Any]],
    ) -> dict[str, Any]:
        stage_counts = {stage_name: len(items) for stage_name, items in stage_samples.items()}
        chart_type_counts: dict[str, int] = defaultdict(int)
        scenario_counts: dict[str, int] = defaultdict(int)
        difficulty_counts: dict[str, int] = defaultdict(int)
        for record in records:
            chart_type_counts[record.chart_type] += 1
            scenario_counts[record.scenario] += 1
        for stage_items in stage_samples.values():
            for item in stage_items:
                difficulty_counts[str(item.difficulty)] += 1
        return {
            "chart_count": len(records),
            "stage_counts": stage_counts,
            "router_holdout_count": len(router_manifest),
            "chart_type_counts": dict(chart_type_counts),
            "scenario_counts": dict(scenario_counts),
            "difficulty_counts": dict(sorted(difficulty_counts.items())),
        }
