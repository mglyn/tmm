# GitHub Upload Note

本文档说明当前工程上传到 GitHub 后，对 `legacy/` 和 `models/` 的外部依赖，以及本地导出包与大目录的体积。

## 1. 当前上传包不包含的内容

本次导出与建议上传到 GitHub 的内容，不包含以下大目录：

- `legacy/`
- `models/`

原因：

- 这两部分体积较大，不适合直接放进 GitHub 仓库
- 当前 router 数据生成脚本和 stage API 启动脚本仍然依赖其中的文件

## 2. 对 `legacy/` 的依赖

如果需要重新生成 router 数据，当前脚本依赖 `legacy/` 中的以下内容：

- 数据集目录：
  - `legacy/chart_vqa_synthesis_1/chartqa_dataset`
- 已训练好的 stage adapter：
  - `legacy/chart_vqa_synthesis_1/outputs/models/stage2_basic_vqa`
  - `legacy/chart_vqa_synthesis_1/outputs/models/stage3_reasoning_vqa`
  - `legacy/chart_vqa_synthesis_1/outputs/models/stage4_visual_analysis`
  - `legacy/chart_vqa_synthesis_1/outputs/models/stage5_code_generation`
- 本地 `LLaMA-Factory` 源码：
  - `legacy/chart_vqa_synthesis_1/LLaMA-Factory-main/src`

说明：

- 仅用于使用现有 `router_data/*/merged/*.jsonl` 训练 router 分类器时，不再强依赖 `legacy/chart_vqa_synthesis_1/chartqa_dataset`
- 但如果要重新跑 `run_router_full_pipeline.py`、重新采集 `stage2~stage5` 预测或重建 oracle 数据，则必须有上述 `legacy/` 内容

## 3. 对 `models/` 的依赖

如果需要重新启动 stage API 或重新生成 router 数据，当前脚本依赖：

- 基座模型目录：
  - `models/Qwen2.5-VL-7B-Instruct`

说明：

- 仅使用已经导出的 `router_data/*/merged/*.jsonl` 做下游 router 训练时，不必保留该模型目录
- 但只要需要重新启动 `stage2~stage5` API，`models/Qwen2.5-VL-7B-Instruct` 就是必需的

## 4. 当前本地体积

基于当前本地目录统计：

- 导出包：
  - `build/exports/tmm_router_export_clean_20260506_011222.tar.gz`
  - 约 `11.86 MB`
- `legacy/`：
  - 约 `5.83 GB`
- `models/`：
  - 约 `15.46 GB`
- `router_data/`：
  - 约 `81.73 MB`
- 当前整个工程目录：
  - 约 `21.40 GB`

## 5. GitHub 上传建议

建议上传到 GitHub 的内容：

- `scripts/`
- `router_data/*/merged/`
- `router_data/*/status.json`
- `README.md`
- `ROUTER_DATA.md`
- `GITHUB_UPLOAD_NOTE.md`
- 论文相关文件，如 `tmm_paper/`

不建议上传到 GitHub 的内容：

- `legacy/`
- `models/`
- `.cache/`
- `tools/`
- `router_data/*/cache/`
- `router_data/*/logs/`
- `router_data/*/pids/`

## 6. 最小复现说明

如果别人从 GitHub 拉取后想直接继续 router 小模型训练：

- 只需要仓库中的 `router_data/train_pipeline/merged/*.jsonl` 与 `router_data/val_pipeline/merged/*.jsonl`
- 不需要立刻恢复 `legacy/` 和 `models/`

如果别人想重新生成 router 数据：

- 必须额外准备 `legacy/` 和 `models/`
- 并保证本地 Python 环境具备 `datasets`、`requests` 等依赖
