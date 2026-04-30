# TMM 实验任务总入口

本仓库用于把 `tmm_paper/main.tex` 中的 TMM 期刊版实验从零跑完：先在本地合成图表数据，再把数据和工程同步到 AutoDL 训练，最后完成路由、评测和论文回填。按下面的子任务一个接一个完成，不跳着跑。

## 子任务顺序

### subtask0: `0autoDL配置.md`

目标：在 AutoDL RTX 5090 上准备训练环境，确认 GPU、CUDA、PyTorch 和项目训练依赖正常。注意：合成图表数据不需要上服务器，优先在本地完成。

完成标志：

- `nvidia-smi` 能看到 RTX 5090
- `torch.cuda.is_available()` 为 `True`
- `bash autodl/first_run.sh` 能跑通

### subtask1: `1合成数据.md`

目标：在本地生成论文需要的 synthetic chart 数据、五阶段训练 manifest、CRS 校验结果和 router holdout 集；生成后再随项目一起同步到 AutoDL。

完成标志：

- `data/synthetic/charts/` 下有合成图表
- `data/synthetic/manifests/` 下有 `S1~S5` 五阶段 JSONL
- `data/synthetic/manifests/router_holdout.jsonl` 存在
- `data/synthetic/manifests/summary.json` 中样本数量合理

### subtask2: 基线训练

目标：训练 `Standard LoRA SFT`，作为所有主结果和消融实验的地板线。

核心命令：

```bash
python scripts/train_experiment.py --config configs/base_experiment.yaml --mode sft
```

### subtask3: 五阶段 Curriculum 训练

目标：按 `S1 -> S2 -> S3 -> S4 -> S5` 顺序训练 `2D-CL`，并保存每个 stage 的 LoRA adapter。

核心命令：

```bash
python scripts/train_experiment.py --config configs/base_experiment.yaml --mode curriculum
```

### subtask4: Stage-Aware Router

目标：基于 held-out routing set 训练路由器，完成 `S2/S3/S4/S5` 专家选择。

核心命令：

```bash
python scripts/train_router.py --config configs/base_experiment.yaml
```

### subtask5: 主评测

目标：跑 `ChartQA / PlotQA / FigureQA / ChartBench / OpenCQA`，生成主结果、路由结果和跨数据集结果。

核心命令：

```bash
python scripts/run_eval.py --config configs/base_experiment.yaml
```

### subtask6: 期刊扩展实验

目标：补齐 TMM 需要的数据规模、难度可靠性、鲁棒性、错误分析和 stage-specific profile。

核心命令示例：

```bash
python scripts/run_analysis_suite.py --config configs/base_experiment.yaml --task scaling
python scripts/run_analysis_suite.py --config configs/base_experiment.yaml --task difficulty_audit
python scripts/run_analysis_suite.py --config configs/base_experiment.yaml --task robustness
```

### subtask7: 论文回填与编译

目标：把真实实验结果回填到论文，并编译 `tmm_paper/main.tex`。

核心命令：

```bash
python scripts/backfill_main_tex.py --config configs/base_experiment.yaml
powershell -ExecutionPolicy Bypass -File tmm_paper/build.ps1
```

## 目录说明

- `0autoDL配置.md`: AutoDL 上机、训练环境配置、显卡检查
- `1合成数据.md`: 本地 synthetic chart 数据构建
- `autodl/`: AutoDL 环境脚本
- `configs/`: 训练、数据、评测与论文回填配置
- `data/`: 原始数据、中间产物与生成后的 manifest
- `outputs/`: checkpoint、预测结果、汇总指标
- `reports/`: 论文规格冻结、结果映射与实验记录
- `scripts/`: 可直接运行的入口脚本
- `src/tmm_chart/`: 核心 Python 包
- `tmm_paper/`: 论文源码、参考文献、计划文档和编译产物

## 当前默认约定

- 主模型默认使用 `Qwen/Qwen2.5-VL-7B-Instruct`
- 数据流水线默认生成五阶段样本，并为 `S3` 执行 CRS 代码校验
- 训练与评测结果全部写入 `outputs/`
- 论文回填目标默认是 `tmm_paper/main.tex`
