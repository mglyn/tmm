from __future__ import annotations

import ast
from collections import defaultdict
from random import Random

from .schema import ChartRecord, StageSample


STAGES = [
    "stage1_description",
    "stage2_basic_vqa",
    "stage3_reasoning_vqa",
    "stage4_visual_analysis",
    "stage5_code_generation",
]


def score_difficulty(question: str, answer: str, metadata: dict) -> int:
    tokens = question.lower()
    score = 1
    if any(keyword in tokens for keyword in ["sum", "difference", "total", "average", "percentage"]):
        score = 2
    if any(keyword in tokens for keyword in ["between", "compared", "combined", "ratio", "trend"]):
        score = max(score, 3)
    if any(keyword in tokens for keyword in ["highest and lowest", "variance", "distribution", "interpret"]):
        score = max(score, 4)
    if metadata.get("requires_domain_hint"):
        score = 5
    return min(score, 5)


def validate_python_trace(code: str, expected_answer: str) -> bool:
    try:
        compiled = compile(ast.parse(code), "<crs>", "exec")
    except SyntaxError:
        return False

    namespace: dict[str, object] = {}
    try:
        exec(compiled, {}, namespace)
    except Exception:
        return False
    answer = namespace.get("answer")
    if answer is None:
        return False
    return str(answer).strip() == str(expected_answer).strip()


def _description_sample(record: ChartRecord) -> StageSample:
    trend = record.metadata.get("trend", "mixed")
    target = (
        f"This is a {record.chart_type} chart about {record.title.lower()}. "
        f"The x-axis is {record.x_label.lower()} and the y-axis is {record.y_label.lower()}. "
        f"The highest category is {record.metadata['argmax']} at {record.metadata['max']}, "
        f"the lowest category is {record.metadata['argmin']} at {record.metadata['min']}, "
        f"and the overall trend is {trend}."
    )
    return StageSample(
        sample_id=f"{record.chart_id}_s1",
        chart_id=record.chart_id,
        stage_name="stage1_description",
        prompt="Describe the chart in detail.",
        target=target,
        difficulty=1,
        chart_type=record.chart_type,
        scenario=record.scenario,
        image_path=record.image_path,
    )


def _basic_vqa_sample(record: ChartRecord, rng: Random) -> StageSample:
    index = rng.randrange(len(record.categories))
    category = record.categories[index]
    value = record.values[index]
    question = f"What is the value for {category}?"
    answer = str(value)
    return StageSample(
        sample_id=f"{record.chart_id}_s2",
        chart_id=record.chart_id,
        stage_name="stage2_basic_vqa",
        prompt=question,
        target=answer,
        question=question,
        answer=answer,
        difficulty=1,
        chart_type=record.chart_type,
        scenario=record.scenario,
        image_path=record.image_path,
    )


def _reasoning_vqa_sample(record: ChartRecord) -> StageSample:
    left_name, right_name = record.categories[0], record.categories[1]
    left_value, right_value = record.values[0], record.values[1]
    answer = round(left_value + right_value, 2)
    question = f"What is the sum of {left_name} and {right_name}?"
    code = "\n".join(
        [
            f"values = {{'{left_name}': {left_value}, '{right_name}': {right_value}}}",
            f"left = values['{left_name}']",
            f"right = values['{right_name}']",
            "answer = round(left + right, 2)",
        ]
    )
    if not validate_python_trace(code, str(answer)):
        raise ValueError(f"Invalid CRS trace for chart {record.chart_id}")
    difficulty = score_difficulty(question, str(answer), {})
    return StageSample(
        sample_id=f"{record.chart_id}_s3",
        chart_id=record.chart_id,
        stage_name="stage3_reasoning_vqa",
        prompt=question,
        target=code,
        question=question,
        answer=str(answer),
        difficulty=difficulty,
        chart_type=record.chart_type,
        scenario=record.scenario,
        image_path=record.image_path,
        metadata={"expected_answer": str(answer)},
    )


def _visual_analysis_samples(record: ChartRecord) -> list[StageSample]:
    samples: list[StageSample] = []
    samples.append(
        StageSample(
            sample_id=f"{record.chart_id}_s4_count",
            chart_id=record.chart_id,
            stage_name="stage4_visual_analysis",
            prompt="How many data elements are shown in the chart?",
            target=str(len(record.categories)),
            question="How many data elements are shown in the chart?",
            answer=str(len(record.categories)),
            difficulty=1,
            chart_type=record.chart_type,
            scenario=record.scenario,
            image_path=record.image_path,
        )
    )
    samples.append(
        StageSample(
            sample_id=f"{record.chart_id}_s4_type",
            chart_id=record.chart_id,
            stage_name="stage4_visual_analysis",
            prompt="What type of chart is this?",
            target=record.chart_type,
            question="What type of chart is this?",
            answer=record.chart_type,
            difficulty=1,
            chart_type=record.chart_type,
            scenario=record.scenario,
            image_path=record.image_path,
        )
    )
    samples.append(
        StageSample(
            sample_id=f"{record.chart_id}_s4_peak",
            chart_id=record.chart_id,
            stage_name="stage4_visual_analysis",
            prompt="Which category has the highest value?",
            target=record.metadata["argmax"],
            question="Which category has the highest value?",
            answer=record.metadata["argmax"],
            difficulty=2,
            chart_type=record.chart_type,
            scenario=record.scenario,
            image_path=record.image_path,
        )
    )
    return samples


def _code_generation_sample(record: ChartRecord) -> StageSample:
    target = "\n".join(
        [
            "import matplotlib.pyplot as plt",
            f"categories = {record.categories!r}",
            f"values = {record.values!r}",
            "plt.figure(figsize=(8, 6))",
            f"plt.title({record.title!r})",
            "plt.bar(categories, values) if len(values) else None",
            f"plt.xlabel({record.x_label!r})",
            f"plt.ylabel({record.y_label!r})",
            "plt.tight_layout()",
            "plt.show()",
        ]
    )
    return StageSample(
        sample_id=f"{record.chart_id}_s5",
        chart_id=record.chart_id,
        stage_name="stage5_code_generation",
        prompt="Generate Python matplotlib code that reconstructs this chart.",
        target=target,
        difficulty=4,
        chart_type=record.chart_type,
        scenario=record.scenario,
        image_path=record.image_path,
    )


def build_stage_samples(records: list[ChartRecord], seed: int) -> dict[str, list[StageSample]]:
    rng = Random(seed)
    samples: dict[str, list[StageSample]] = defaultdict(list)
    for record in records:
        samples["stage1_description"].append(_description_sample(record))
        samples["stage2_basic_vqa"].append(_basic_vqa_sample(record, rng))
        samples["stage3_reasoning_vqa"].append(_reasoning_vqa_sample(record))
        samples["stage4_visual_analysis"].extend(_visual_analysis_samples(record))
        samples["stage5_code_generation"].append(_code_generation_sample(record))

    for stage_name, stage_samples in samples.items():
        stage_samples.sort(key=lambda item: (item.difficulty, item.sample_id))
    return samples
