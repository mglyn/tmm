# 待补充实验清单

本文档用于配合当前 TMM 稿件 [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex) 进行实验补充与结果回填。  
当前论文主线为：`2D-CL + CRS + DACDS + Stage-Aware Router (SAR)`。

目标不是“想到什么跑什么”，而是优先完成能支撑投稿的最小实验闭环。

---

## 1. 使用说明

- 本清单按照当前论文结构组织
- 每一项都包含：
  - 实验目标
  - 最低配置
  - 需要产出的结果
  - 对应回填到论文的位置
- 优先级分为：
  - `P0`：投稿前必须完成
  - `P1`：强烈建议完成
  - `P2`：有时间再做

---

## 2. 总览

### P0：必须完成

- 主结果实验
- 多随机种子实验
- 路由实验
- 数据规模实验
- 核心消融实验
- 难度可靠性实验
- 错误分析

### P1：强烈建议完成

- 鲁棒性实验
- 跨模型实验
- 跨数据集实验扩展到 `ChartBench / OpenCQA`
- 定性案例整理

### P2：可选加分

- 更细粒度的路由错误分析
- 更细粒度的 CRS 对比
- 更细粒度的 inheritance 对比
- 额外 supplementary 结果

---

## 3. 逐项实验清单

## 3.1 主结果实验

### 编号

- `E01`

### 优先级

- `P0`

### 实验目标

- 产出整篇论文最核心的主结果表
- 验证 `2D-CL + SAR` 相比 `Standard LoRA SFT` 和单 stage adapter 的总体增益

### 最低配置

- 模型：`Qwen2.5-VL-7B`
- 训练数据：当前 12K synthetic 数据
- 对比方法：
  - Zero-shot base model
  - Standard LoRA SFT
  - Best single-stage adapter
  - `2D-CL + SAR`

### 产出要求

- `ChartQA` 总体准确率
- 如可行，同时给出：
  - 参数规模
  - 训练数据规模
  - 与基线的绝对提升

### 对应回填位置

- `Experiments -> Main Results`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L265-L323)
- 重点表格：
  - `Table \ref{tab:sota}`
  - `Table \ref{tab:transfer}`

### 备注

- 摘要中的主结果数字也要同步更新

---

## 3.2 跨数据集主结果

### 编号

- `E02`

### 优先级

- `P0`

### 实验目标

- 证明方法不是只对 `ChartQA` 有效
- 支撑“generalization”与“dataset transfer”结论

### 最低配置

- 训练：仍然只在当前 synthetic 数据上训练
- 测试数据集：
  - `ChartQA`
  - `PlotQA`
  - `FigureQA`
  - `ChartBench`
  - `OpenCQA`

### 对比方法

- Standard LoRA SFT
- Best single-stage adapter
- `2D-CL + SAR`

### 产出要求

- 每个 benchmark 的总体分数
- 最好再记录：
  - 评价协议
  - 是否 zero-shot
  - 是否需要格式后处理

### 对应回填位置

- `Experiments -> Main Results`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L285-L323)
- `Table \ref{tab:transfer}`

### 备注

- 如果 `ChartBench / OpenCQA` 暂时来不及，可先只跑你最有把握接入的一个，但最终最好两个都补

---

## 3.3 多随机种子实验

### 编号

- `E03`

### 优先级

- `P0`

### 实验目标

- 回应“单次实验不稳”的质疑
- 为论文提供 `mean ± std`

### 最低配置

- 数据集：`ChartQA`
- 模型：`Qwen2.5-VL-7B`
- 种子数：至少 `3 seeds`
- 对比方法：
  - Standard LoRA SFT
  - 1D-Difficulty
  - 1D-Task
  - Best single-stage `2D-CL`
  - `2D-CL + SAR`

### 产出要求

- 每个方法的：
  - mean
  - std
- 最好保留原始 seed 结果，便于后续算显著性

### 对应回填位置

- `Experiments -> Multi-Seed and Cross-Model Results`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L365-L395)
- `Table \ref{tab:seeds}`

### 备注

- 如果有余力，可以额外补一个简单的显著性检验

---

## 3.4 跨模型实验

### 编号

- `E04`

### 优先级

- `P1`

### 实验目标

- 证明方法不依赖单一 backbone

### 最低配置

- 模型：
  - `Qwen2.5-VL-7B`
  - `Qwen2.5-VL-3B`
  - `InternVL2-8B`
- 数据集：`ChartQA`
- 对比方法：
  - Standard LoRA SFT
  - `2D-CL + SAR`

### 产出要求

- 每个模型的最终准确率
- 与 SFT 的提升

### 对应回填位置

- `Experiments -> Multi-Seed and Cross-Model Results`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L396-L417)
- `Table \ref{tab:models}`

### 备注

- 如果资源不够，优先保住 `7B + 3B`

---

## 3.5 Routing 核心实验

### 编号

- `E05`

### 优先级

- `P0`

### 实验目标

- 验证 `SAR` 的真实价值
- 这是方案 A 的核心新增方法实验

### 最低配置

- 数据集：
  - `ChartQA`
  - 最好再补 `ChartBench` 和 `OpenCQA`
- 对比策略：
  - Stage 2 only
  - Stage 3 only
  - Stage 4 only
  - Stage 5 only
  - Oracle routing
  - Predicted routing (`SAR`)

### 产出要求

- 每个策略的分数
- `SAR` 相对 best single stage 的提升
- `SAR` 相对 oracle routing 的 gap

### 对应回填位置

- `Experiments -> Routing Analysis`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L418-L442)
- `Table \ref{tab:routing}`

### 备注

- 没有这组实验，`SAR` 这条主线会很虚

---

## 3.6 数据规模实验

### 编号

- `E06`

### 优先级

- `P0`

### 实验目标

- 证明课程学习在低资源场景尤其有效
- 支撑“data efficiency” claim

### 最低配置

- 数据比例：
  - `10%`
  - `25%`
  - `50%`
  - `100%`
  - `150%` 或 `200%`（可选）
- 对比方法：
  - Standard LoRA SFT
  - 1D-Task
  - `2D-CL + SAR`

### 产出要求

- 不同数据规模下的最终分数
- 最好保留训练成本和收敛趋势

### 对应回填位置

- `Experiments -> Data Efficiency`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L443-L465)
- `Table \ref{tab:scaling}`

### 备注

- 这组实验很重要，因为它直接支撑“结构比规模更关键”的主张

---

## 3.7 核心消融实验

### 编号

- `E07`

### 优先级

- `P0`

### 实验目标

- 证明每个组件都有效

### 最低配置

- 数据集：`ChartQA`
- 对比配置：
  - Full `2D-CL + SAR`
  - w/o Router
  - w/o Staging
  - w/o Difficulty
  - w/o CRS
  - w/o Inheritance
  - Standard LoRA SFT

### 产出要求

- 每个消融配置的准确率
- Full model 相对每个配置的下降值

### 对应回填位置

- `Experiments -> Ablation Study`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L325-L364)
- `Table \ref{tab:ablation}`
- 附录中的 `Table \ref{tab:component_detail}`

### 备注

- 这个表是整篇文章解释机制的基础

---

## 3.8 难度可靠性实验

### 编号

- `E08`

### 优先级

- `P0`

### 实验目标

- 验证 `DACDS` 的 difficulty score 不是拍脑袋

### 最低配置

- 从数据中随机抽样一部分样本，例如 `200~300`
- 人工标注 difficulty level
- 与 LLM 自动打分做比较

### 产出要求

- agreement rate
- mean absolute deviation
- 如果可以，再补：
  - easy-to-hard
  - hard-to-easy
  - random
  三种排序对比结果

### 对应回填位置

- `Analysis -> Difficulty Reliability`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L559-L564)

### 备注

- 如果人工标注成本太高，最少要完成抽样对齐验证

---

## 3.9 鲁棒性实验

### 编号

- `E09`

### 优先级

- `P1`

### 实验目标

- 检验方法在图像质量退化下是否更稳

### 最低配置

- 基于 `ChartQA` 构造 corruption 版本：
  - JPEG compression
  - Gaussian blur
  - Low resolution
  - Color jitter
  - Partial occlusion
- 对比方法：
  - Standard LoRA SFT
  - Best single-stage `2D-CL`
  - `2D-CL + SAR`

### 产出要求

- 每种扰动下的准确率下降值
- 平均 drop

### 对应回填位置

- `Experiments -> Robustness and Error Analysis`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L466-L487)
- `Table \ref{tab:robustness}`

### 备注

- 没有这组实验也能投稿，但有了会明显更像期刊稿

---

## 3.10 错误分析

### 编号

- `E10`

### 优先级

- `P0`

### 实验目标

- 说明方法到底减少了哪类错误
- 避免整篇文章只停留在分数提升

### 最低配置

- 从 `ChartQA` 失败案例中抽样，例如 `200~300`
- 建立错误标签：
  - value extraction
  - arithmetic
  - multi-step reasoning
  - answer format
  - legend/axis mapping
  - counting/dense perception

### 产出要求

- 每类错误占比
- SFT vs `2D-CL + SAR` 的错误分布对比

### 对应回填位置

- `Experiments -> Robustness and Error Analysis`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L489-L509)
- `Table \ref{tab:error}`

### 备注

- 这是很值得做的正文实验，不建议省

---

## 3.11 Stage-Specific 内部任务评测

### 编号

- `E11`

### 优先级

- `P1`

### 实验目标

- 支撑“各 stage 是不同专家”的核心分析结论

### 最低配置

- internal task set
- 任务：
  - T1 Description
  - T2 Basic VQA
  - T3 Reasoning
  - T4 Visual Analysis
  - T5 Code Generation
- 比较对象：
  - Baseline
  - Stage 1
  - Stage 2
  - Stage 3
  - Stage 4
  - Stage 5

### 产出要求

- stage-task performance matrix

### 对应回填位置

- `Analysis -> Stage-Specific Expertise`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L512-L539)
- `Table \ref{tab:stage_performance}`

### 备注

- 这张表是 `SAR` 合理性的直接证据

---

## 3.12 CRS 专项验证

### 编号

- `E12`

### 优先级

- `P1`

### 实验目标

- 解释 `CRS` 为什么有效

### 最低配置

- 对比：
  - with CRS
  - without CRS
- 最好再补：
  - reasoning task 单独统计
  - internal reasoning benchmark

### 产出要求

- reasoning task 上的性能差异
- 示例分析

### 对应回填位置

- `Analysis -> The Role of Code-as-Reasoning Supervision`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L541-L551)

### 备注

- 如果时间够，可进一步扩展为：
  - code vs text CoT
  - verified code vs unverified code

---

## 3.13 Inheritance 专项验证

### 编号

- `E13`

### 优先级

- `P1`

### 实验目标

- 解释为什么 adapter inheritance 是最大贡献项

### 最低配置

- 对比：
  - with inheritance
  - without inheritance
- 最好结合 Stage 3 或 reasoning-heavy setting 单独分析

### 产出要求

- inheritance 的 gain
- 对 reasoning stage 的影响说明

### 对应回填位置

- `Analysis -> Why Adapter Inheritance Matters`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L553-L557)

---

## 3.14 定性案例

### 编号

- `E14`

### 优先级

- `P1`

### 实验目标

- 为 `SAR`、`CRS`、错误类型分析提供直观案例

### 最低配置

- 至少准备三类案例：
  - `SAR` 选择 Stage 2 修复 format error
  - `SAR` 选择 Stage 3 解决 arithmetic reasoning
  - dense chart 上仍然失败的负例

### 产出要求

- 每类至少 `1~2` 个图表问答案例
- 包含：
  - 问题
  - GT
  - 基线输出
  - 我们方法输出
  - 简短解释

### 对应回填位置

- `Analysis -> Qualitative Examples`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L565-L569)

---

## 3.15 效率与部署成本

### 编号

- `E15`

### 优先级

- `P1`

### 实验目标

- 证明 `SAR` 的收益不是靠巨大部署代价换来的

### 最低配置

- 比较：
  - best single stage
  - `2D-CL + SAR`
- 记录：
  - latency / sample
  - routing overhead
  - adapter storage

### 产出要求

- 效率表

### 对应回填位置

- `Additional Results -> Efficiency of Routing`
- [main.tex](file:///c:/Users/12037/Desktop/ieee%20tmm/tmm_paper/main.tex#L674-L691)
- `Table \ref{tab:efficiency}`

---

## 4. 最小可投稿实验包

如果时间紧，建议至少完成下面这 `7` 项：

- `E01` 主结果实验
- `E02` 跨数据集主结果
- `E03` 多随机种子
- `E05` Routing 核心实验
- `E06` 数据规模实验
- `E07` 核心消融实验
- `E10` 错误分析

完成这 7 项后，整篇论文的主线就基本闭环了。

---

## 5. 推荐执行顺序

### 第一阶段：先保住论文主线

1. `E01` 主结果
2. `E07` 核心消融
3. `E05` Routing
4. `E03` 多随机种子

### 第二阶段：补强期刊说服力

1. `E02` 跨数据集
2. `E06` 数据规模
3. `E10` 错误分析
4. `E08` 难度可靠性

### 第三阶段：加分项

1. `E09` 鲁棒性
2. `E11` Stage-specific 评测
3. `E12` CRS 专项
4. `E13` Inheritance 专项
5. `E14` 定性案例
6. `E15` 效率分析

---

## 6. 结果回填提醒

跑完实验后，优先回填以下位置：

- 摘要中的总结果与提升数字
- `Main Results`
- `Ablation Study`
- `Multi-Seed and Cross-Model Results`
- `Routing Analysis`
- `Data Efficiency`
- `Robustness and Error Analysis`
- `Difficulty Reliability`
- `Discussion` 和 `Conclusion` 中的总结性数字

如果表格数字更新了，但摘要、引言、结论没同步，论文会显得很不严谨。

---

## 7. 一句话版本

当前这篇 TMM 稿件最重要的不是继续写文字，而是优先补齐：

> 主结果、路由、多 seed、数据规模、核心消融、错误分析、难度可靠性。

这几项决定了文章能不能站住。

flowchart TD
    A[Chart-Question Input] --> B[Frozen Multimodal Backbone]
    B --> C[Pooled Joint Representation]
    C --> D[Lightweight Router]
    D --> E[Stage Prediction]

    A --> F[Offline Expert Evaluation]
    F --> G[Oracle Routing Labels]

    G --> H[Router Training Loss]
    D --> H

    E --> S2[Stage 2 Adapter]
    E --> S3[Stage 3 Adapter]
    E --> S4[Stage 4 Adapter]
    E --> S5[Stage 5 Adapter]

    S2 --> Y[Generated Answer]
    S3 --> Y
    S4 --> Y
    S5 --> Y

