# TMM 实验 Todo

本文档只保留当前稿件真正还值得补的实验，不再重复已经完成的主结果、消融、多种子、路由、跨模型、跨数据集、难度验证和效率统计。

当前状态：论文约 9.5 页，已经具备 conference-to-journal 的基本骨架，下一步目标不是继续堆结果，而是补齐“完整性、全面性、期刊说服力”。

## 1. 当前判断

已经完成并写入论文的内容：

- 主结果：`2D-CL + SAR` vs `Standard LoRA SFT`
- 跨数据集：`ChartQA / PlotQA / FigureQA`
- 跨模型：`Qwen2.5-VL-7B / Qwen2.5-VL-3B / InternVL2-8B`
- 多随机种子：最终 stage-selection 比较
- Routing analysis：best stage / oracle / predicted routing
- Error analysis：shared failures taxonomy
- Difficulty reliability
- Deployment efficiency

当前最缺的不是“再多一个表”，而是下面三类证据：

1. 数据规模变化下方法是否仍然成立
2. 真实扰动或分布偏移下方法是否更稳健
3. 相比 `Standard SFT`，我们到底减少了哪些错误

## 2. 优先级

### `P0` 投稿前最值得补

- `E16` Data Efficiency / Scaling
- `E17` Robustness on Corrupted ChartQA
- `E18` Comparative Error Analysis (`SFT` vs `2D-CL+SAR`)

### `P1` 强烈建议补

- `E19` New Benchmark Extension (`ChartBench` / `OpenCQA`)
- `E20` Qualitative Case Board

### `P2` 有时间再做

- `E21` Router Failure Analysis
- `E22` CRS stronger ablation
- `E23` Adapter consolidation / distillation pilot

## 3. 设计实验

## 3.1 `E16` Data Efficiency / Scaling

### 目的

把“只用 12K synthetic 也有效”从一句 claim 变成完整结论。

这组实验要回答三个问题：

- 当训练数据变少时，`2D-CL+SAR` 是否更有优势
- 当训练数据变多时，`2D-CL` 的增益是否仍然存在
- curriculum 的收益主要体现在低资源，还是是一个普遍现象

### 最低配置

- 数据比例：`10% / 25% / 50% / 100%`
- 模型：`Qwen2.5-VL-7B`
- 对比：
  - `Standard LoRA SFT`
  - `1D-Task`
  - `1D-Difficulty`
  - `2D-CL`
  - 如资源允许，再加 `2D-CL+SAR`

### 产出

- 每个数据规模下的 `ChartQA` 准确率
- 一张 scaling curve
- 一个简表总结 low-resource gain

### 预期论文位置

- 新增 `Experiments -> Data Efficiency`
- 建议插在 [main.tex](</c:/Users/12037/Desktop/ieee tmm/tmm_paper/main.tex:430>) 的 `Routing Analysis` 之后、`Error Analysis` 之前

### 为什么最重要

这是当前最能体现期刊“完整性”的实验，因为它直接支撑：

- 数据效率
- 2D curriculum 的适用范围
- 与会议版相比的新增说服力

## 3.2 `E17` Robustness on Corrupted ChartQA

### 目的

证明方法不仅更准，而且更稳。

如果不补这组实验，审稿人很容易认为当前方法主要改善了训练组织和答案格式，但并没有显著增强真实视觉扰动下的鲁棒性。

### 最低配置

- 基于 `ChartQA` 构造 corruption 版本
- 扰动类型建议：
  - `JPEG compression`
  - `Gaussian blur`
  - `low resolution`
  - `color jitter`
  - `partial occlusion`
- 对比：
  - `Standard LoRA SFT`
  - `Best single-stage`
  - `2D-CL+SAR`

### 产出

- 每种扰动下的准确率
- 相对 clean accuracy 的 drop
- 平均 drop 汇总表

### 预期论文位置

- 新增 `Experiments -> Robustness Evaluation`
- 建议插在 [main.tex](</c:/Users/12037/Desktop/ieee tmm/tmm_paper/main.tex:482>) 当前 `Error Analysis` 之前

### 写作价值

这组实验最容易把稿件从“方法有效”推进到“方法更全面、更接近真实应用”。

## 3.3 `E18` Comparative Error Analysis (`SFT` vs `2D-CL+SAR`)

### 目的

把现有 error analysis 从“我们还错在哪”升级为“我们比 baseline 少错在哪”。

当前论文已经有 shared failures taxonomy，但还缺少最关键的一步：与 `Standard SFT` 的对比。

### 最低配置

- 从 `ChartQA` 中抽样：
  - `SFT only wrong`
  - `Ours only wrong`
  - `Both wrong`
- 统一使用同一套 taxonomy：
  - `value extraction`
  - `arithmetic`
  - `multi-step reasoning`
  - `answer format`
  - `legend/axis mapping`
  - `counting/dense perception`

### 产出

- `SFT` vs `2D-CL+SAR` 错误分布对比表
- 一张 stacked bar 或 grouped bar
- 一段明确结论：
  - 哪些错误明显下降
  - 哪些错误基本没动

### 预期论文位置

- 替换或扩展 [main.tex](</c:/Users/12037/Desktop/ieee tmm/tmm_paper/main.tex:482>) 当前 `Error Analysis`

### 写作价值

这组实验最适合回答：

- `CRS` 主要修复了什么
- `Stage 2` 的格式学习究竟值不值
- 剩余瓶颈是否真的集中在数值 grounding

## 3.4 `E19` New Benchmark Extension

### 目的

进一步扩大“有效性范围”。

如果 `E16` 和 `E17` 已经完成，这组实验会是最好的加分项；如果资源紧张，它不是第一优先级。

### 推荐数据集

- `ChartBench`
- `OpenCQA`

### 最低配置

- zero-shot 或 minimal adaptation
- 至少比较：
  - `Standard SFT`
  - `Best single-stage`
  - `2D-CL+SAR`

### 产出

- benchmark 总分
- 如可行，附 question-type breakdown

### 预期论文位置

- 扩展 [main.tex](</c:/Users/12037/Desktop/ieee tmm/tmm_paper/main.tex:272>) `Main Results`
  或新增 `Broader Benchmark Evaluation`

### 风险

- 数据集接入成本可能高于 corruption-based robustness
- 若时间有限，优先做 `E17`，不优先做这项

## 3.5 `E20` Qualitative Case Board

### 目的

把论文中的定性分析从文字描述升级成可视案例页。

### 最低配置

至少整理三类样例：

- `SAR` 成功选择更优 stage
- `CRS` 明显帮助 arithmetic reasoning
- dense / cluttered chart 上的失败案例

### 每个样例应包含

- chart image
- question
- ground truth
- `SFT` output
- best stage output
- `SAR` output
- 1-2 句解释

### 预期论文位置

- 扩展 [main.tex](</c:/Users/12037/Desktop/ieee tmm/tmm_paper/main.tex:571>) `Qualitative Examples`
  或放到 appendix 作为单独图页

### 价值

这项不一定显著涨分，但非常有助于提升审稿体验。

## 4. 推荐执行顺序

### 路线 A：最稳妥

1. 做 `E16`
2. 做 `E18`
3. 做 `E17`

这条路线最适合“先把期刊完整性补齐”。

### 路线 B：想快速扩页

1. 做 `E16`
2. 做 `E17`
3. 整理 `E20`

这条路线最容易把 9.5 页稳定推进到 11.5 到 12 页。

### 路线 C：资源充足再上强扩展

1. `E16`
2. `E17`
3. `E18`
4. `E19`

这条路线最像完整期刊扩展版。

## 5. 最小可投稿补充包

如果只补三项，我建议固定为：

1. `E16` Data Efficiency
2. `E17` Robustness
3. `E18` Comparative Error Analysis

原因很简单：

- `E16` 解决“数据效率是否稳健成立”
- `E17` 解决“真实扰动下是否更稳”
- `E18` 解决“到底修复了什么错误”

这三项补完之后，论文会比现在更完整、更全面，也更像期刊稿，而不是 conference extension 的加长版。
