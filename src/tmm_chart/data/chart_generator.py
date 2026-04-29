from __future__ import annotations

import math
import uuid
from pathlib import Path
from random import Random

import matplotlib.pyplot as plt
import numpy as np

from .schema import ChartRecord


SCENARIO_LABELS: dict[str, tuple[str, str, str]] = {
    "business": ("Monthly Revenue", "Month", "Revenue (M USD)"),
    "science": ("Experiment Outcomes", "Condition", "Measurement"),
    "economics": ("Regional GDP Growth", "Region", "Growth (%)"),
    "healthcare": ("Patient Admissions", "Week", "Admissions"),
    "climate": ("Seasonal Rainfall", "Month", "Rainfall (mm)"),
    "education": ("Course Enrollments", "Course", "Students"),
    "finance": ("Portfolio Allocation", "Asset", "Weight (%)"),
    "sports": ("Match Statistics", "Team", "Score"),
    "energy": ("Power Generation", "Source", "Output (GWh)"),
    "transportation": ("Transit Usage", "Line", "Trips (K)"),
}


def _random_categories(rng: Random, count: int) -> list[str]:
    base = [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
        "Epsilon",
        "Zeta",
        "Eta",
        "Theta",
        "Iota",
        "Kappa",
    ]
    rng.shuffle(base)
    return base[:count]


def _random_values(rng: Random, count: int, chart_type: str) -> list[float]:
    if chart_type == "scatter":
        return [round(rng.uniform(5, 95), 2) for _ in range(count)]
    return [round(rng.uniform(10, 100), 2) for _ in range(count)]


def _render_chart(record: ChartRecord, output_dir: Path, width: int, height: int, dpi: int) -> str:
    figure_size = (width / dpi, height / dpi)
    fig, ax = plt.subplots(figsize=figure_size, dpi=dpi)
    categories = record.categories
    values = record.values

    if record.chart_type == "bar":
        ax.bar(categories, values, color="#4C72B0")
    elif record.chart_type == "line":
        ax.plot(categories, values, marker="o", linewidth=2.2, color="#55A868")
    elif record.chart_type == "pie":
        ax.pie(values, labels=categories, autopct="%1.1f%%", startangle=140)
    elif record.chart_type == "scatter":
        x_axis = np.linspace(1, len(values), len(values))
        ax.scatter(x_axis, values, c=values, cmap="viridis", s=70)
        ax.set_xticks(x_axis, categories)
    elif record.chart_type == "area":
        x_axis = np.arange(len(values))
        ax.fill_between(x_axis, values, color="#C44E52", alpha=0.35)
        ax.plot(x_axis, values, color="#C44E52")
        ax.set_xticks(x_axis, categories)
    else:
        raise ValueError(f"Unsupported chart type: {record.chart_type}")

    ax.set_title(record.title)
    if record.chart_type != "pie":
        ax.set_xlabel(record.x_label)
        ax.set_ylabel(record.y_label)
        ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    image_path = output_dir / f"{record.chart_id}.png"
    fig.savefig(image_path)
    plt.close(fig)
    return str(image_path)


def generate_chart_records(
    chart_count: int,
    chart_types: list[str],
    scenarios: list[str],
    output_dir: Path,
    seed: int,
    width: int,
    height: int,
    dpi: int,
) -> list[ChartRecord]:
    rng = Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[ChartRecord] = []

    for index in range(chart_count):
        chart_type = chart_types[index % len(chart_types)]
        scenario = scenarios[index % len(scenarios)]
        title, x_label, y_label = SCENARIO_LABELS[scenario]
        category_count = 6 if chart_type != "pie" else 5
        categories = _random_categories(rng, category_count)
        values = _random_values(rng, category_count, chart_type)
        chart_id = f"{scenario}_{chart_type}_{uuid.uuid4().hex[:10]}"
        stats = {
            "max": max(values),
            "min": min(values),
            "mean": round(sum(values) / len(values), 2),
            "range": round(max(values) - min(values), 2),
            "argmax": categories[int(np.argmax(values))],
            "argmin": categories[int(np.argmin(values))],
            "sum": round(sum(values), 2),
            "trend": "increasing"
            if values[-1] > values[0] and chart_type in {"line", "area"}
            else "mixed",
            "variance_proxy": round(float(np.var(values)), 2),
        }
        record = ChartRecord(
            chart_id=chart_id,
            chart_type=chart_type,
            scenario=scenario,
            title=title,
            x_label=x_label,
            y_label=y_label,
            categories=categories,
            values=values,
            metadata=stats,
            image_path="",
        )
        record.image_path = _render_chart(record, output_dir, width, height, dpi)
        records.append(record)
    return records
