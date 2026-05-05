# Router Data Quickstart

本文档说明如何基于现有 `stage2~stage5` adapter 扩出 `Stage-Aware Router (SAR)` 的训练数据。

核心思路：

1. 对同一批 `ChartQA` 样本，同时调用 `stage2~stage5`
2. 保存每个 stage 的预测答案
3. 根据正确性和最小误差规则生成 `oracle_stage`
4. 输出 `router_train.jsonl` 和 `router_val.jsonl`，供后续 router 分类器训练

---

## 1. 依赖

- 已能启动各 stage 的推理 API
- 本地存在 `ChartQA` 的 `load_from_disk` 数据目录
- Python 环境已安装：
  - `datasets`
  - `requests`

脚本位置：

- `/data/scripts/build_router_dataset.py`

---

## 2. 建议的 API 端口

建议四个 stage 分别占用四个端口：

- `stage2 -> 8002`
- `stage3 -> 8003`
- `stage4 -> 8004`
- `stage5 -> 8005`

如果你已经把 Windows 版 `START_API.ps1` 迁到 Linux，可以按这个端口映射启动。

---

## 3. 最小运行命令

先在小样本上 smoke test：

```bash
python /data/scripts/build_router_dataset.py \
  --dataset_path /data/legacy/chart_vqa_synthesis_1/chartqa_dataset \
  --split test \
  --sample_limit 100 \
  --stage-endpoint stage2=http://localhost:8002 \
  --stage-endpoint stage3=http://localhost:8003 \
  --stage-endpoint stage4=http://localhost:8004 \
  --stage-endpoint stage5=http://localhost:8005 \
  --output_dir /data/router_data/chartqa_router_smoke
```

如果 smoke test 正常，再跑全量：

```bash
python /data/scripts/build_router_dataset.py \
  --dataset_path /data/legacy/chart_vqa_synthesis_1/chartqa_dataset \
  --split test \
  --stage-endpoint stage2=http://localhost:8002 \
  --stage-endpoint stage3=http://localhost:8003 \
  --stage-endpoint stage4=http://localhost:8004 \
  --stage-endpoint stage5=http://localhost:8005 \
  --output_dir /data/router_data/chartqa_router_full
```

如果需要显式指定 API 模型名，可以增加：

```bash
  --stage-model stage2=default \
  --stage-model stage3=default \
  --stage-model stage4=default \
  --stage-model stage5=default
```

---

## 4. 输出文件

脚本会输出：

- `router_all.jsonl`
- `router_train.jsonl`
- `router_val.jsonl`
- `summary.json`

默认按 `0.9 / 0.1` 划分 train/val。

---

## 5. 单条样本格式

```json
{
  "dataset_path": "/data/legacy/chart_vqa_synthesis_1/chartqa_dataset",
  "dataset_split": "test",
  "sample_id": 123,
  "question": "What is the difference between A and B?",
  "gold_answers": ["12"],
  "oracle_stage": "stage3",
  "oracle_stage_id": 1,
  "label_source": "correct_stage",
  "correct_stages": ["stage2", "stage3"],
  "stage_predictions": {
    "stage2": "12",
    "stage3": "12",
    "stage4": "14",
    "stage5": "10"
  },
  "stage_correctness": {
    "stage2": true,
    "stage3": true,
    "stage4": false,
    "stage5": false
  },
  "stage_error_scores": {
    "stage2": 0.0,
    "stage3": 0.0,
    "stage4": 0.1667,
    "stage5": 0.1667
  }
}
```

---

## 6. 当前 oracle 规则

### 6.1 有正确 stage 时

- 先找所有答对的 stage
- 如果有多个 stage 同时答对，用问题启发式打破并列：
  - arithmetic/comparison 类优先 `stage3`
  - visual/layout 类优先 `stage4`
  - code/script 类优先 `stage5`
  - 其他默认优先 `stage2`

### 6.2 全错时

- 比较每个 stage 对 gold answer 的误差分数
- 选误差最小的 stage 作为 `least_error_stage`

这套规则的目的不是最终论文结论，而是先稳定扩出 router 的初版训练数据。

---

## 7. 推荐流程

1. 先跑 `100` 样本 smoke test
2. 检查 `summary.json` 里的 oracle 分布是否严重塌到单一 stage
3. 抽查 `router_all.jsonl` 的前几十条，确认标签逻辑基本合理
4. 再跑全量
5. 后续如果需要更强 oracle，可以再替换 tie-break 或引入 token-level confidence

---

## 8. 下一步

这份数据跑出来之后，下一步通常是两件事：

1. 训练一个轻量 router 分类器
2. 做 `E05` 的 `Stage2/3/4/5 vs Oracle vs Predicted Routing`

如果你要继续，我下一步可以直接帮你补：

- Linux 版多端口 stage API 启动脚本
- router 分类器训练脚本
- routing 评测脚本
