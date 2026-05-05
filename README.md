# IEEE TMM Experimental README

## 1. 目标

本 README 用于把 `tmm_paper/todo.md` 中的实验清单整理成一份可以直接执行的期刊扩展实验方案。

当前任务背景如下：

- 论文主线已经确定为：`2D-CL + CRS + DACDS + Stage-Aware Router (SAR)`
- 旧工程位于 `legacy/`
- 旧稿件是 ACM MM 版本，当前目标是扩展为 IEEE TMM 期刊版本
- 当前主要资源是 `AutoDL + RTX 5090`

这份文档的目标不是重复论文文字，而是回答下面四个实际问题：

- 先跑哪些实验，才能尽快形成可投稿闭环
- 每个实验怎么设计，比较哪些方法，产出什么结果
- 哪些部分可以直接复用 `legacy/`，哪些部分需要新补代码
- 单卡 `5090` 下如何安排优先级、时间和风险

---

## 2. 当前工程映射

### 2.1 论文目录

- `tmm_paper/main.tex`
- `tmm_paper/todo.md`
- `tmm_paper/plan.md`

### 2.2 可复用旧工程

主要复用目录：

- `legacy/chart_vqa_synthesis_1/`
- `legacy/ABLATION/`

其中已经存在的可复用内容包括：

- 五阶段训练流程
- ChartQA 评测脚本
- 部分消融实验数据准备与训练配置
- 多模型配置草稿

### 2.3 现有能力与缺口

已经比较完整的部分：

- `2D-CL` 五阶段训练
- `CRS` 与 `DACDS` 的基础流程
- `ChartQA` 评测入口
- `w/o difficulty`、`w/o inherit`、`random` 等消融框架
- `Qwen2.5-VL-3B` 与 `InternVL2-8B` 的配置草稿

还需要新增或补齐的部分：

- `SAR` 路由器训练与推理代码
- `PlotQA / FigureQA / ChartBench / OpenCQA` 评测适配
- 多随机种子批量运行与汇总
- 数据规模切分脚本
- 错误分析、难度可靠性、鲁棒性实验脚本
- Linux 版统一入口脚本

结论很明确：

> 旧工程足够支撑 `E01 / E03 / E04 / E07` 的主体复现，但 `E02 / E05 / E06 / E08 / E09 / E10 / E11 / E12 / E13 / E15` 需要新补实验脚本或评测整理。

---

## 3. AutoDL 5090 执行原则

### 3.1 默认硬件假设

- GPU: `RTX 5090 32GB`
- OS: Linux
- Python: `3.10+`
- 训练框架: `LLaMA-Factory`
- 精度: `bf16`

### 3.2 推荐资源策略

- 所有主实验默认按单卡单进程设计
- 优先保留 `Qwen2.5-VL-7B` 作为主模型
- `3B` 跨模型实验作为轻量补充
- `InternVL2-8B` 仅在前面主线跑通后再补
- 所有新实验先跑 `100~300` 样本 smoke test，再跑全量

### 3.3 5090 上的现实约束

- `7B` 五阶段完整训练可以作为主线实验单位
- 多种子和多比例实验会迅速放大总训练时长
- 路由实验如果设计成“复用已训练 stage adapter + 训练一个轻量分类器”，性价比最高
- 不建议一开始同时铺开 `ChartBench + OpenCQA + PlotQA + FigureQA + 多模型 + 多seed`

### 3.4 环境建议

- 先在 AutoDL 建一个独立环境，例如 `tmm5090`
- 若默认镜像对 `5090` 支持有问题，优先复用旧工程中已经验证过的 `PyTorch + CUDA` 组合
- 所有路径从 Windows 风格统一迁移到 Linux 风格
- 统一把中间结果和最终结果写到新目录，避免污染 `legacy/` 原始产物

---

## 4. 推荐目录规范

建议在 AutoDL 上新建如下实验目录：

```text
/data/
├── legacy/
├── tmm_paper/
├── tmm_runs/
│   ├── data/
│   ├── checkpoints/
│   ├── eval/
│   ├── router/
│   ├── logs/
│   ├── tables/
│   └── figures/
└── tmm_env/
```

建议统一命名规则：

- 训练输出：`/data/tmm_runs/checkpoints/<exp_id>/<model>/<setting>/`
- 评测输出：`/data/tmm_runs/eval/<exp_id>/<dataset>/<model>.json`
- 表格汇总：`/data/tmm_runs/tables/<table_name>.csv`
- 论文作图：`/data/tmm_runs/figures/<figure_name>.pdf`

实验 ID 建议直接对应 `todo.md`：

- `E01_main_results`
- `E02_cross_dataset`
- `E03_multiseed`
- `E05_routing`
- `E06_data_scaling`
- `E07_ablation`

---

## 5. 期刊最小闭环

如果目标是先形成一个“能投 TMM 的最小实验包”，优先级如下：

### 5.1 P0 必跑

- `E01` 主结果实验
- `E02` 跨数据集主结果
- `E03` 多随机种子实验
- `E05` Routing 核心实验
- `E06` 数据规模实验
- `E07` 核心消融实验
- `E10` 错误分析
- `E08` 难度可靠性实验

### 5.2 P1 第二阶段补强

- `E04` 跨模型实验
- `E09` 鲁棒性实验
- `E11` Stage-specific 内部任务评测
- `E12` CRS 专项验证
- `E13` Inheritance 专项验证
- `E14` 定性案例
- `E15` 效率与部署成本

### 5.3 单卡 5090 推荐顺序

1. `E01`
2. `E07`
3. `E05`
4. `E03`
5. `E02`
6. `E06`
7. `E10`
8. `E08`
9. `E04`
10. `E09 / E11 / E12 / E13 / E14 / E15`

这样安排的原因是：

- 先保住主结果、机制解释和新方法主线
- 再补泛化和 data efficiency
- 最后做更像期刊加分项的分析实验

---

## 6. 方法与对照统一定义

为避免后续表格命名混乱，建议统一使用下面的模型命名。

### 6.1 训练方法

- `Base`: 零样本基座模型
- `SFT`: Standard LoRA SFT
- `1D-Task`: 只按 task/stage 排序
- `1D-Diff`: 只按 difficulty 排序
- `2D-CL`: 不带 router 的二维课程学习
- `2D-CL + SAR`: 完整方法

### 6.2 Router 相关方法

- `Stage2-only`
- `Stage3-only`
- `Stage4-only`
- `Stage5-only`
- `Oracle Routing`
- `Pred Routing (SAR)`

### 6.3 统一基座模型

- 主模型：`Qwen2.5-VL-7B`
- 轻量对照：`Qwen2.5-VL-3B`
- 跨架构补充：`InternVL2-8B`

### 6.4 统一指标

- 主指标：`Accuracy`
- 稳定性指标：`mean ± std`
- 路由指标：`routing accuracy`, `oracle gap`
- 鲁棒性指标：`drop under corruption`
- 效率指标：`latency / sample`, `GPU memory`, `adapter storage`

---

## 7. 实验方案总表

| 编号 | 名称 | 优先级 | 是否需要重新训练 | 是否需要新代码 | 主要产出 |
| --- | --- | --- | --- | --- | --- |
| E01 | 主结果实验 | P0 | 是 | 否，少量整理 | 主结果表 |
| E02 | 跨数据集主结果 | P0 | 否或少量 | 是 | transfer 表 |
| E03 | 多随机种子 | P0 | 是 | 是，批量脚本 | mean ± std |
| E04 | 跨模型 | P1 | 是 | 否，配置补齐 | model 泛化表 |
| E05 | Routing 核心实验 | P0 | 部分 | 是 | routing 表 |
| E06 | 数据规模 | P0 | 是 | 是，切分脚本 | scaling 表 |
| E07 | 核心消融 | P0 | 是 | 部分已有 | ablation 表 |
| E08 | 难度可靠性 | P0 | 否 | 是 | reliability 指标 |
| E09 | 鲁棒性 | P1 | 否 | 是 | robustness 表 |
| E10 | 错误分析 | P0 | 否 | 是 | error taxonomy 表 |
| E11 | Stage-specific 评测 | P1 | 否 | 部分已有 | stage-task matrix |
| E12 | CRS 专项 | P1 | 是 | 是 | CRS 分析表 |
| E13 | Inheritance 专项 | P1 | 是 | 部分已有 | inheritance 分析表 |
| E14 | 定性案例 | P1 | 否 | 否 | qualitative figure |
| E15 | 效率与部署 | P1 | 否 | 是 | efficiency 表 |

---

## 8. 分实验设计

## E01 主结果实验

### 目标

- 给出 TMM 稿件最核心的主结果
- 证明 `2D-CL + SAR` 相比 `SFT` 和单 stage adapter 的总体收益

### 推荐设置

- 基座模型：`Qwen2.5-VL-7B`
- 训练数据：当前 `12K synthetic`
- 测试数据：`ChartQA`

### 对比方法

- `Base`
- `SFT`
- `Best single-stage adapter`
- `2D-CL`
- `2D-CL + SAR`

### 结果产出

- `ChartQA accuracy`
- 参数规模
- 训练数据规模
- 相对 `SFT` 的绝对提升

### 工程实现

- 训练主线优先复用 `legacy/chart_vqa_synthesis_1/start_training.ps1`
- Linux 环境下改写为 bash 或 yaml 驱动版本
- `SFT` 建议定义为单阶段混合训练基线，而不是只拿 `Stage2-only` 代替

### 风险提醒

- 旧工程中最终最佳 stage 可能不是 `Stage5`
- 主结果表必须统一“最终模型”的定义，否则会被审稿人质疑比较不公平

---

## E02 跨数据集主结果

### 目标

- 证明方法不是只在 `ChartQA` 上有效
- 支撑 generalization 和 zero-shot transfer

### 推荐设置

- 训练仍然只在 synthetic 数据上完成
- 测试集按优先级接入：
  - 第一优先：`PlotQA`
  - 第二优先：`FigureQA`
  - 第三优先：`ChartBench`
  - 第四优先：`OpenCQA`

### 为什么这样排

- `PlotQA / FigureQA` 更接近标准 benchmark，接入难度通常低于开放式真实数据集
- `ChartBench / OpenCQA` 对 TMM 更加分，但工程适配成本通常更高

### 对比方法

- `SFT`
- `Best single-stage adapter`
- `2D-CL + SAR`

### 结果产出

- 每个 benchmark 的总体分数
- 评价协议说明
- 是否 zero-shot
- 是否使用后处理

### 工程实现

- `legacy` 中现成的是 `ChartQA` evaluator
- 需要新建统一评测接口，例如：
  - `eval_plotqa.py`
  - `eval_figureqa.py`
  - `eval_chartbench.py`
  - `eval_opencqa.py`
- 所有评测脚本统一输出到 `json + csv`

### 最小落地建议

- 如果资源和时间紧，先做 `ChartQA + PlotQA + FigureQA`
- `ChartBench / OpenCQA` 至少补一个

---

## E03 多随机种子实验

### 目标

- 回应单次实验不稳定的质疑
- 为正文提供 `mean ± std`

### 推荐设置

- 数据集：`ChartQA`
- 模型：`Qwen2.5-VL-7B`
- 种子：至少 `3 seeds`
- 若时间允许，扩展到 `5 seeds`

### 对比方法

- `SFT`
- `1D-Diff`
- `1D-Task`
- `2D-CL`
- `2D-CL + SAR`

### 结果产出

- 每个方法的 `mean ± std`
- 原始 seed 结果
- 可选：配对显著性检验

### 工程实现

- 需要新增批量运行脚本，例如 `run_multiseed.sh`
- 统一 seed 列表：`[42, 3407, 2025]`
- 不要每个实验临时改 seed，避免后续复现困难

### 计算建议

- 先只对 `SFT / 2D-CL / 2D-CL + SAR` 做 `3 seeds`
- 若结果已经稳定，再补 `1D-Diff / 1D-Task`

---

## E04 跨模型实验

### 目标

- 证明方法不依赖单一 backbone

### 推荐设置

- 模型：
  - `Qwen2.5-VL-7B`
  - `Qwen2.5-VL-3B`
  - `InternVL2-8B`
- 数据集：`ChartQA`

### 对比方法

- `SFT`
- `2D-CL + SAR`

### 结果产出

- 各模型最终准确率
- 相对 `SFT` 的提升

### 工程实现

- `legacy/ABLATION/configs/multi_model_configs.yaml` 已有基础配置
- 建议先把 `3B` 跑通
- `InternVL2-8B` 留到主线稳定后再跑

### 单卡建议

- 单 `5090` 下优先保住 `7B + 3B`
- `InternVL2-8B` 作为期刊加分项，不必阻塞主稿

---

## E05 Routing 核心实验

### 目标

- 验证 `SAR` 的真实价值
- 这是 TMM 相比 ACM MM 最核心的新方法实验

### 推荐设置

- 基础 stage adapter：复用已训练好的 `Stage2~Stage5`
- 路由器输入：图像特征 + 问题文本特征
- 路由器输出：`{2, 3, 4, 5}` 四分类

### 路由标签设计

- 使用离线评测生成 oracle label
- 对每个样本，让 `Stage2~Stage5` 分别作答
- 选择答对且置信最高的 stage，或直接选择最优 stage

### 对比方法

- `Stage2-only`
- `Stage3-only`
- `Stage4-only`
- `Stage5-only`
- `Oracle Routing`
- `Pred Routing (SAR)`

### 结果产出

- 每种策略的分数
- `SAR` 相对 best single stage 的提升
- `SAR` 相对 oracle 的 gap
- 路由器分类准确率

### 工程实现建议

- 不建议一开始做复杂端到端联合训练
- 第一版直接做轻量 router：
  - 输入可使用 backbone pooled feature
  - 或者先用 question-only classifier 做弱基线
- 路由器单独训练，stage adapter 保持冻结

### 最关键的论文价值

- 没有这组实验，`SAR` 只停留在概念上
- 有了这组实验，TMM 扩展才真正从“补实验”升级为“新系统”

---

## E06 数据规模实验

### 目标

- 证明课程学习在低资源场景更有效
- 支撑 data efficiency claim

### 推荐设置

- 数据比例：`10% / 25% / 50% / 100%`
- 可选扩展：`150%` 或 `200%`

### 对比方法

- `SFT`
- `1D-Task`
- `2D-CL + SAR`

### 结果产出

- 不同数据规模下的准确率
- 收敛趋势
- 训练成本

### 工程实现

- 需要新增数据切分脚本
- 每个比例固定一个主切分，并保留切分清单
- 不建议每次重新随机采样，否则 scaling 曲线噪声过大

### 计算建议

- 如果预算有限，先跑 `10% / 50% / 100%`
- 补图时再加 `25%`

---

## E07 核心消融实验

### 目标

- 证明每个组件都有效

### 推荐设置

- 数据集：`ChartQA`
- 模型：`Qwen2.5-VL-7B`

### 对比配置

- `Full 2D-CL + SAR`
- `w/o Router`
- `w/o Staging`
- `w/o Difficulty`
- `w/o CRS`
- `w/o Inheritance`
- `SFT`

### 结果产出

- 每个配置的准确率
- Full 相对各配置的下降值

### 工程实现

- `legacy/ABLATION/` 已有 `w/o difficulty`、`w/o inherit`、`random` 等框架
- `w/o Router` 是新加项，需要把完整方法回退为固定单 stage 或 oracle-free 推理
- `w/o CRS` 和 `w/o Staging` 需要检查旧脚本是否完整实现，必要时补齐

### 表格建议

- 正文放主表
- 附录放更细粒度组件说明

---

## E08 难度可靠性实验

### 目标

- 验证 `DACDS` 的 difficulty score 可信

### 推荐设置

- 从训练数据中随机抽 `200~300` 个样本
- 人工标注难度等级
- 与 LLM 自动难度做比较

### 指标

- `agreement rate`
- `MAE`
- `Spearman correlation`

### 扩展对比

- `easy-to-hard`
- `hard-to-easy`
- `random`

### 工程实现

- 需要导出标注子集
- 制作一个轻量人工标注表格
- 标注结果和自动分数汇总为 csv

### 最小可接受版本

- 先完成抽样对齐验证
- 排序策略对比放在有余力时再补

---

## E09 鲁棒性实验

### 目标

- 检验方法在图像质量退化下是否更稳

### 推荐设置

- 基于 `ChartQA` 构造 corruption 版本
- 扰动类型：
  - JPEG compression
  - Gaussian blur
  - low resolution
  - color jitter
  - partial occlusion

### 对比方法

- `SFT`
- `Best single-stage 2D-CL`
- `2D-CL + SAR`

### 结果产出

- 各扰动下准确率下降值
- 平均 drop

### 工程实现

- 不需要重新训练
- 需要新增 corruption 数据生成脚本
- 评测时保证同一批样本被不同模型共用

---

## E10 错误分析

### 目标

- 说明方法减少了哪类错误
- 避免全文只有最终分数提升

### 推荐设置

- 从 `ChartQA` 失败案例中抽取 `200~300` 个样本
- 建立统一错误标签体系

### 标签建议

- `value extraction`
- `arithmetic`
- `multi-step reasoning`
- `answer format`
- `legend/axis mapping`
- `counting/dense perception`

### 对比方法

- `SFT`
- `2D-CL + SAR`

### 结果产出

- 每类错误占比
- 两种方法的错误分布对比
- 代表性失败案例

### 工程实现

- 需要先保存逐样本预测结果
- 最好做一个半自动标注表
- 后续 `E14` 的定性案例可直接复用这里的样本

---

## E11 Stage-Specific 内部任务评测

### 目标

- 支撑“各 stage 是不同专家”的核心分析

### 推荐设置

- internal task set
- 任务：
  - `T1 Description`
  - `T2 Basic VQA`
  - `T3 Reasoning`
  - `T4 Visual Analysis`
  - `T5 Code Generation`

### 比较对象

- `Baseline`
- `Stage1`
- `Stage2`
- `Stage3`
- `Stage4`
- `Stage5`

### 结果产出

- stage-task performance matrix

### 工程实现

- 旧工程里已经有 internal task 数据组织方式
- 需要重新整理成一张适合正文或附录展示的矩阵表

---

## E12 CRS 专项验证

### 目标

- 解释 `CRS` 为什么有效

### 推荐设置

- `with CRS`
- `without CRS`

### 推荐扩展

- 单独统计 reasoning-heavy 样本
- 若时间允许，增加 `code vs text CoT`

### 结果产出

- reasoning 类问题上的性能差异
- 示例分析

### 工程实现

- 需要单独构造不含 CRS 的 stage3 数据
- 建议与 `E07` 的 `w/o CRS` 共享结果，避免重复训练

---

## E13 Inheritance 专项验证

### 目标

- 解释为什么 adapter inheritance 是最大贡献项

### 推荐设置

- `with inheritance`
- `without inheritance`

### 推荐扩展

- 重点看 `Stage3` 或 reasoning-heavy 子集
- 分析性能变化是否主要来自 reasoning stage

### 结果产出

- inheritance 带来的 gain
- 对 reasoning stage 的影响解释

### 工程实现

- `legacy/ABLATION/` 中已有 `no_inherit` 框架
- 需要把结果整理为论文风格，而不仅是训练日志

---

## E14 定性案例

### 目标

- 为 `SAR`、`CRS` 和错误分析提供直观案例

### 建议准备三类案例

- `SAR` 选择 `Stage2` 修复 format error
- `SAR` 选择 `Stage3` 解决 arithmetic reasoning
- dense chart 上依旧失败的负例

### 结果产出

- 每类 `1~2` 个案例
- 每个案例包含：
  - 问题
  - GT
  - 基线输出
  - 我们方法输出
  - 简短解释

### 工程实现

- 直接从 `E05` 和 `E10` 的保存结果中挑样本
- 不需要额外训练

---

## E15 效率与部署成本

### 目标

- 证明 `SAR` 的收益不是靠巨大部署代价换来的

### 推荐对比

- `Best single stage`
- `2D-CL + SAR`

### 记录指标

- `latency / sample`
- `routing overhead`
- `GPU memory`
- `adapter storage`

### 结果产出

- 效率表
- 性能-效率 tradeoff 说明

### 工程实现

- 不必新训练
- 推理时用固定 batch 和固定样本集合测量
- 重复 3 次取平均，避免偶然波动

---

## 9. 与 legacy 的直接复用关系

### 9.1 可以直接参考或迁移的文件

- `legacy/chart_vqa_synthesis_1/start_training.ps1`
- `legacy/chart_vqa_synthesis_1/evaluate_chartqa.py`
- `legacy/chart_vqa_synthesis_1/CHARTQA_EVAL_GUIDE.md`
- `legacy/ABLATION/training/run_ablation_experiments.ps1`
- `legacy/ABLATION/configs/multi_model_configs.yaml`

### 9.2 建议如何迁移

- 不要继续直接在 `legacy/` 上改
- 在新目录中复制为 Linux 版执行脚本
- 保持数据格式和输出结构兼容，方便复用旧结果分析逻辑

### 9.3 新增脚本建议列表

- `scripts/run_main_results.sh`
- `scripts/run_multiseed.sh`
- `scripts/run_scaling.sh`
- `scripts/run_router_train.sh`
- `scripts/run_router_eval.sh`
- `scripts/run_cross_dataset_eval.sh`
- `scripts/run_robustness_eval.sh`
- `scripts/build_error_taxonomy.py`
- `scripts/build_difficulty_validation_set.py`
- `scripts/collect_results.py`

---

## 10. 单卡 5090 时间预算

下面是建议的粗略预算，按单卡保守估计。

| 实验 | 预计耗时 | 说明 |
| --- | --- | --- |
| E01 | 1 个完整主训练周期 + 评测 | 主结果必须优先 |
| E07 | 3~5 个训练周期 | 可分批完成 |
| E05 | 0.5~1 个训练周期 | router 轻量训练为主 |
| E03 | 3 个训练周期起 | 成本高，建议缩小方法数 |
| E02 | 1~3 天 | 主要花在评测接入 |
| E06 | 3~4 个训练周期 | 可先跑 3 个比例 |
| E08 | 0.5~1 天 | 主要是标注和统计 |
| E10 | 1~2 天 | 主要是样本整理和标注 |
| E09 | 0.5~1 天 | 只评测，不训练 |
| E15 | 0.5 天 | 只测推理 |

结论：

- 单卡情况下，不要幻想一次性全做完
- 最现实路线是先完成 `E01 + E07 + E05 + E03`

---

## 11. 推荐执行节奏

### 第一阶段：先保论文主线

1. 跑 `E01`
2. 跑 `E07`
3. 跑 `E05`
4. 跑 `E03`

### 第二阶段：补泛化与数据效率

1. 跑 `E02`
2. 跑 `E06`
3. 跑 `E10`
4. 跑 `E08`

### 第三阶段：期刊加分项

1. 跑 `E04`
2. 跑 `E09`
3. 跑 `E11`
4. 跑 `E12`
5. 跑 `E13`
6. 整理 `E14`
7. 跑 `E15`

---

## 12. 结果记录规范

建议每个实验都至少产出以下文件：

- `config.yaml`
- `train.log`
- `eval.json`
- `summary.csv`
- `notes.md`

建议每个实验目录下放一个 `README.md`，记录：

- 实验目的
- 训练数据
- 对比方法
- 随机种子
- 最终 checkpoint
- 最终指标
- 异常情况

这样后面回填论文时不会反复翻日志。

---

## 13. 论文回填映射

跑完实验后，优先回填下面这些位置：

- `Main Results`
- `Ablation Study`
- `Multi-Seed and Cross-Model Results`
- `Routing Analysis`
- `Data Efficiency`
- `Robustness and Error Analysis`
- `Difficulty Reliability`
- 摘要中的主结果数字
- 引言、讨论、结论中的总结性数字

特别注意：

- 表格更新后，摘要和结论也必须同步改
- `SAR` 的主结果、routing 表和效率表要前后口径一致
- `best single stage` 的定义在全文必须统一

---

## 14. 当前最推荐的实际开跑方案

如果现在就准备在 AutoDL 上开跑，我建议直接按下面的最小方案推进：

### 第 1 批

- `E01`: `Qwen2.5-VL-7B`, `ChartQA`
- `E07`: 先做 `w/o Difficulty`, `w/o Inheritance`, `w/o CRS`
- `E05`: 先做 `Stage2/3/4/5 + Oracle + 简单 Router`

### 第 2 批

- `E03`: 只做 `SFT`, `2D-CL`, `2D-CL + SAR` 三组 `3 seeds`
- `E02`: 先接 `PlotQA`，再补 `FigureQA`
- `E06`: 先跑 `10% / 50% / 100%`

### 第 3 批

- `E10`
- `E08`
- `E15`

这是当前在单卡 `5090` 条件下，最稳、最现实、最能支撑投稿的一条路线。

---

## 15. 一句话结论

这篇 TMM 稿件当前最重要的，不是继续扩写正文，而是尽快用 `E01 + E07 + E05 + E03` 形成主线闭环，再用 `E02 + E06 + E10 + E08` 把“泛化、数据效率、可解释性、可靠性”补扎实。
