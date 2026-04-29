# TMM Chart Understanding Pipeline

本仓库现在除了实验工程外，还把论文源码集中放在 `tmm_paper/` 下，目标对齐 `tmm_paper/main.tex` 中定义的 `2D-CL + CRS + DACDS + Stage-Aware Router` 期刊版方案。

## 目录

- `configs/`: 训练、数据、评测与论文回填配置
- `data/`: 原始数据、中间产物与生成后的 manifest
- `outputs/`: checkpoint、预测结果、汇总指标
- `reports/`: 论文规格冻结、结果映射与实验记录
- `scripts/`: 可直接运行的入口脚本
- `src/tmm_chart/`: 核心 Python 包
- `tmm_paper/`: 论文源码、参考文献、模板类文件和编译产物
- `autodl/`: AutoDL 环境配置与上机脚本

## 推荐执行顺序

1. 安装依赖：`pip install -r requirements.txt`
   如需把包安装到当前环境，可额外执行：`pip install -e .`
2. 生成一小批 synthetic 数据：
   `python scripts/build_dataset.py --config configs/base_experiment.yaml --limit 20`
3. 跑 smoke test：
   `python scripts/run_smoke_test.py --config configs/base_experiment.yaml`
4. 训练标准 SFT：
   `python scripts/train_experiment.py --config configs/base_experiment.yaml --mode sft`
5. 训练五阶段 curriculum：
   `python scripts/train_experiment.py --config configs/base_experiment.yaml --mode curriculum`
6. 训练路由器：
   `python scripts/train_router.py --config configs/base_experiment.yaml`
7. 评测并汇总：
   `python scripts/run_eval.py --config configs/base_experiment.yaml`
8. 编译论文：
   `powershell -ExecutionPolicy Bypass -File build.ps1`

## 当前默认约定

- 主模型默认使用 `Qwen/Qwen2.5-VL-7B-Instruct`
- 数据流水线默认生成五阶段样本，并为 `S3` 执行 CRS 代码校验
- 训练与评测结果全部以 JSON/JSONL/CSV 写入 `outputs/`
- 论文中的 TODO-EXP 对应用 `reports/todo_exp_map.json` 统一管理
- 论文回填目标默认是 `tmm_paper/main.tex`

## 说明

这套工程优先保证“单卡 5090 可落地”和“实验闭环完整”。由于真实训练与评测依赖外部模型权重、基准数据集与 API 凭据，仓库内实现的是可直接运行的工程骨架、配置系统和实验脚本；首次运行前需要补充模型下载权限、基准数据路径以及可选的 LLM 标注接口配置。
