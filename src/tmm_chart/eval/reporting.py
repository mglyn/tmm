from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.common import read_json, write_json


def collect_metrics(metrics_dir: Path) -> dict[str, Any]:
    bundle: dict[str, Any] = {"tables": {}, "narrative": {}, "abstract": {}, "introduction": {}}
    for metrics_path in metrics_dir.rglob("*_metrics.json"):
        payload = read_json(metrics_path)
        benchmark = payload.get("benchmark", "unknown")
        strategy = payload.get("strategy", metrics_path.stem)
        bundle["tables"].setdefault(benchmark, {})[strategy] = payload
    return bundle


def export_paper_metrics(root_dir: Path) -> Path:
    metrics_dir = root_dir / "outputs" / "eval"
    bundle = collect_metrics(metrics_dir)
    target = root_dir / "outputs" / "paper" / "final_metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, bundle)
    export_todo_exp_fills(root_dir, bundle)
    export_generated_tex(root_dir, bundle)
    return target


def build_tex_replacements(metrics: dict[str, Any]) -> dict[str, str]:
    chartqa = metrics.get("tables", {}).get("ChartQA", {})
    sft = chartqa.get("standard_sft", {}).get("accuracy")
    full = chartqa.get("predicted_router", {}).get("accuracy")
    best_stage = chartqa.get("best_single_stage", {}).get("accuracy")
    replacements: dict[str, str] = {}
    if full is not None:
        replacements["{{CHARTQA_MAIN}}"] = f"{full:.2f}"
    if sft is not None and full is not None:
        replacements["{{GAIN_VS_SFT}}"] = f"{full - sft:.2f}"
    if best_stage is not None and full is not None:
        replacements["{{GAIN_VS_BEST_STAGE}}"] = f"{full - best_stage:.2f}"
    return replacements


def build_todo_exp_fills(metrics: dict[str, Any]) -> dict[str, Any]:
    chartqa = metrics.get("tables", {}).get("ChartQA", {})
    return {
        "TODO-EXP-01": {
            "chartqa_main": chartqa.get("predicted_router", {}).get("accuracy"),
            "gain_vs_sft": _safe_gap(chartqa, "predicted_router", "standard_sft"),
            "gain_vs_best_stage": _safe_gap(chartqa, "predicted_router", "best_single_stage"),
        },
        "TODO-EXP-04": metrics.get("tables", {}).get("ChartQA", {}),
        "TODO-EXP-06": metrics.get("tables", {}),
        "TODO-EXP-15": {
            bench: table.get("predicted_router")
            for bench, table in metrics.get("tables", {}).items()
            if "predicted_router" in table
        },
    }


def export_todo_exp_fills(root_dir: Path, metrics: dict[str, Any]) -> Path:
    target = root_dir / "outputs" / "paper" / "todo_exp_fills.json"
    write_json(target, build_todo_exp_fills(metrics))
    return target


def export_generated_tex(root_dir: Path, metrics: dict[str, Any]) -> Path:
    replacements = build_tex_replacements(metrics)
    target = root_dir / "outputs" / "paper" / "generated_metrics.tex"
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated metric macros.",
        rf"\newcommand{{\ChartQAMain}}{{{replacements.get('{{CHARTQA_MAIN}}', '0.00')}}}",
        rf"\newcommand{{\GainVsSFT}}{{{replacements.get('{{GAIN_VS_SFT}}', '0.00')}}}",
        rf"\newcommand{{\GainVsBestStage}}{{{replacements.get('{{GAIN_VS_BEST_STAGE}}', '0.00')}}}",
    ]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _safe_gap(table: dict[str, Any], lhs: str, rhs: str) -> float | None:
    left = table.get(lhs, {}).get("accuracy")
    right = table.get(rhs, {}).get("accuracy")
    if left is None or right is None:
        return None
    return round(left - right, 2)


def backfill_tex_from_metrics(tex_path: Path, metrics_path: Path) -> None:
    metrics = read_json(metrics_path)
    replacements = build_tex_replacements(metrics)
    content = tex_path.read_text(encoding="utf-8")
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    tex_path.write_text(content, encoding="utf-8")
