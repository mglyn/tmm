# 实验执行 Runbook

本 runbook 对应计划中的剩余工程任务，默认在仓库根目录执行。

## 1. 数据构建

```bash
python scripts/build_dataset.py --config configs/base_experiment.yaml
```

小规模验证：

```bash
python scripts/run_smoke_test.py --config configs/base_experiment.yaml --limit 20
```

## 2. 基线训练

标准 SFT：

```bash
python scripts/train_experiment.py --config configs/base_experiment.yaml --mode sft
```

建议先用小样本试跑：

```bash
python scripts/train_experiment.py --config configs/base_experiment.yaml --mode sft --limit 64
```

## 3. 五阶段 Curriculum

```bash
python scripts/train_experiment.py --config configs/base_experiment.yaml --mode curriculum
```

输出 checkpoint：

- `outputs/checkpoints/stage1_description`
- `outputs/checkpoints/stage2_basic_vqa`
- `outputs/checkpoints/stage3_reasoning_vqa`
- `outputs/checkpoints/stage4_visual_analysis`
- `outputs/checkpoints/stage5_code_generation`

## 4. Router

```bash
python scripts/train_router.py --config configs/base_experiment.yaml
```

输出：

- `outputs/router/stage_aware_router.pt`
- `outputs/router/metrics.json`

## 5. 主评测

将各 benchmark 按如下形式放入：

- `data/benchmarks/chartqa/test.jsonl`
- `data/benchmarks/plotqa/test.jsonl`
- `data/benchmarks/figureqa/test.jsonl`
- `data/benchmarks/chartbench/test.jsonl`
- `data/benchmarks/opencqa/test.jsonl`

字段建议统一为：

```json
{
  "sample_id": "chartqa_0001",
  "image_path": "absolute/or/relative/path.png",
  "question": "What is the value for ...?",
  "answer": "42",
  "metadata": {}
}
```

运行统一评测：

```bash
python scripts/run_eval.py --config configs/base_experiment.yaml
```

## 6. 期刊扩展分析

数据规模切分：

```bash
python scripts/run_analysis_suite.py --config configs/base_experiment.yaml --task scaling
```

难度人工校验表：

```bash
python scripts/run_analysis_suite.py --config configs/base_experiment.yaml --task difficulty_audit
```

完成人工标注后汇总：

```bash
python scripts/run_analysis_suite.py --config configs/base_experiment.yaml --task difficulty_summary
```

构造 ChartQA corruption 版本：

```bash
python scripts/run_analysis_suite.py --config configs/base_experiment.yaml --task robustness
```

从预测文件生成错误分类统计：

```bash
python scripts/run_analysis_suite.py --config configs/base_experiment.yaml --task error_taxonomy --input outputs/eval/chartqa_predicted_router_predictions.jsonl
```

导出 stage-task profile：

```bash
python scripts/run_analysis_suite.py --config configs/base_experiment.yaml --task task_profile
```

## 7. 论文回填

统一评测结束后会自动生成：

- `outputs/paper/final_metrics.json`
- `outputs/paper/todo_exp_fills.json`
- `outputs/paper/generated_metrics.tex`

如果你把 `tmm_paper/main.tex` 中关键数字改成宏或占位符，可以直接运行：

```bash
python scripts/backfill_main_tex.py --config configs/base_experiment.yaml
```

编译论文：

```bash
powershell -ExecutionPolicy Bypass -File tmm_paper/build.ps1
```

## 8. 推荐真实执行顺序

1. `smoke_test`
2. `standard_sft`
3. `curriculum`
4. `router`
5. `ChartQA` 主结果
6. 核心消融
7. 跨数据集
8. 多 seed / 数据规模 / 难度可靠性 / 错误分析 / 鲁棒性
