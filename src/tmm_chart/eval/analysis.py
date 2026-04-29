from __future__ import annotations

import io
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter

from ..utils.common import read_jsonl, write_json, write_jsonl


def create_scaled_manifests(manifest_path: Path, scales: list[float], output_dir: Path, seed: int) -> list[Path]:
    records = read_jsonl(manifest_path)
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    for scale in scales:
        count = max(1, int(len(shuffled) * scale))
        subset = shuffled[:count]
        scale_name = str(scale).replace(".", "p")
        target = output_dir / f"{manifest_path.stem}_{scale_name}.jsonl"
        write_jsonl(target, subset)
        created.append(target)
    return created


def build_corrupted_benchmark(
    benchmark_root: Path,
    benchmark_name: str,
    corruptions: list[str],
) -> list[Path]:
    source_path = benchmark_root / benchmark_name.lower() / "test.jsonl"
    records = read_jsonl(source_path)
    generated_paths: list[Path] = []
    for corruption in corruptions:
        target_dir = benchmark_root / f"{benchmark_name.lower()}_{corruption}"
        image_dir = target_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        output_records = []
        for item in records:
            source_image = Image.open(item["image_path"]).convert("RGB")
            corrupted = apply_corruption(source_image, corruption)
            target_image = image_dir / Path(item["image_path"]).name
            corrupted.save(target_image)
            updated = dict(item)
            updated["image_path"] = str(target_image)
            output_records.append(updated)
        jsonl_path = target_dir / "test.jsonl"
        write_jsonl(jsonl_path, output_records)
        generated_paths.append(jsonl_path)
    return generated_paths


def apply_corruption(image: Image.Image, corruption: str) -> Image.Image:
    if corruption == "jpeg":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=35)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if corruption == "gaussian_blur":
        return image.filter(ImageFilter.GaussianBlur(radius=2))
    if corruption == "low_resolution":
        down = image.resize((image.width // 2, image.height // 2))
        return down.resize(image.size)
    if corruption == "color_jitter":
        return ImageEnhance.Color(image).enhance(1.7)
    if corruption == "partial_occlusion":
        occluded = image.copy()
        for x in range(image.width // 3, image.width // 3 * 2):
            for y in range(image.height // 3, image.height // 3 * 2):
                occluded.putpixel((x, y), (0, 0, 0))
        return occluded
    raise ValueError(f"Unsupported corruption: {corruption}")


def sample_difficulty_audit(stage_manifest: Path, sample_size: int, output_path: Path, seed: int) -> Path:
    records = read_jsonl(stage_manifest)
    rng = random.Random(seed)
    sampled = rng.sample(records, min(sample_size, len(records)))
    audit_rows = []
    for item in sampled:
        audit_rows.append(
            {
                "sample_id": item["sample_id"],
                "question": item.get("question", item["prompt"]),
                "llm_difficulty": item["difficulty"],
                "manual_difficulty": None,
                "notes": "",
            }
        )
    write_jsonl(output_path, audit_rows)
    return output_path


def summarize_difficulty_audit(audit_path: Path, output_path: Path) -> Path:
    rows = read_jsonl(audit_path)
    comparable = [row for row in rows if row["manual_difficulty"] is not None]
    if not comparable:
        payload = {"count": 0, "agreement_rate": None, "mean_absolute_deviation": None}
    else:
        agreements = [int(int(row["manual_difficulty"]) == int(row["llm_difficulty"])) for row in comparable]
        deviations = [abs(int(row["manual_difficulty"]) - int(row["llm_difficulty"])) for row in comparable]
        payload = {
            "count": len(comparable),
            "agreement_rate": round(sum(agreements) / len(agreements), 4),
            "mean_absolute_deviation": round(sum(deviations) / len(deviations), 4),
        }
    write_json(output_path, payload)
    return output_path


def build_error_taxonomy(
    prediction_path: Path,
    output_path: Path,
    label_rules: dict[str, list[str]] | None = None,
) -> Path:
    if label_rules is None:
        label_rules = {
            "value_extraction": ["read", "extract", "value"],
            "arithmetic": ["sum", "total", "difference", "average", "percentage"],
            "multi_step_reasoning": ["compare", "between", "ratio", "trend"],
            "answer_format": ["yes", "no", "%"],
            "legend_axis_mapping": ["legend", "axis", "series"],
            "counting_dense_perception": ["how many", "count"],
        }
    rows = read_jsonl(prediction_path)
    counts: Counter[str] = Counter()
    for row in rows:
        if row["prediction"] == row["answer"]:
            continue
        question = row["question"].lower()
        assigned = False
        for label, keywords in label_rules.items():
            if any(keyword in question for keyword in keywords):
                counts[label] += 1
                assigned = True
                break
        if not assigned:
            counts["other"] += 1
    write_json(output_path, dict(counts))
    return output_path


def build_stage_task_profile(manifests_dir: Path, output_path: Path) -> Path:
    mapping = defaultdict(dict)
    for manifest_path in manifests_dir.glob("stage*.jsonl"):
        records = read_jsonl(manifest_path)
        stage_name = manifest_path.stem
        mapping[stage_name]["sample_count"] = len(records)
        mapping[stage_name]["difficulty_mean"] = round(
            sum(record["difficulty"] for record in records) / max(1, len(records)), 2
        )
    write_json(output_path, mapping)
    return output_path
