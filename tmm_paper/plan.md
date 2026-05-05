# ACM MM 扩展为 IEEE TMM 期刊文章计划

## 1. 当前工作基础

当前 ACM MM 版本的核心内容已经比较完整，主线清晰：

- 任务：图表理解（Chart Understanding）
- 方法：`2D Curriculum Learning (2D-CL)`
- 两个核心设计：
  - 按任务复杂度逐阶段训练
  - 按样本难度由易到难排序
- 两个关键技术点：
  - `CRS (Code-as-Reasoning Supervision)`
  - `DACDS (Difficulty-Aware Curriculum Data Selection)`
- 当前已有实验：
  - ChartQA 主结果
  - 组件消融
  - 跨模型泛化
  - 零样本跨数据集迁移
  - 若干分析实验和定性案例

这说明论文已经具备“方法 + 主实验 + 分析”的基本结构，但如果要扩展为 IEEE TMM 期刊，现阶段还需要进一步强化以下几点：

- 评测维度还不够宽，主 benchmark 仍然偏集中在 `ChartQA`
- 真实场景与鲁棒性分析还不够充分
- 统计显著性和复现实验不足
- 方法层面目前更像“训练策略创新”，期刊版最好再补一个更强的机制设计
- 当前多阶段 adapter 的部署复杂度被指出了，但尚未形成一个正面的解决方案

因此，TMM 版本不能只做“加长版”，而应当升级为：

> 在 ACM MM 的 2D-CL 基础上，进一步提出更完整的训练与部署框架，并通过更大范围、更系统的实验验证其在图表理解中的有效性、鲁棒性和可扩展性。

---

## 2. 期刊版总体目标

建议将 TMM 扩展目标定为以下三点：

### 目标 1：把“有效”做扎实

不仅证明方法在 `ChartQA` 上有效，还要证明其：

- 对更多 benchmark 有效
- 对不同 chart 类型有效
- 对不同问题类型有效
- 对不同模型规模和架构有效
- 对不同随机种子和数据规模有效

### 目标 2：把“为什么有效”讲透

当前论文已经有部分分析，但还不够系统。期刊版建议进一步回答：

- 2D curriculum 为什么优于 1D curriculum？
- 哪类问题最受益于 CRS？
- difficulty score 是否真的可靠？
- stage inheritance 的收益来自知识传递，还是仅仅来自更好的初始化？
- 2D-CL 是否改善了模型的校准性、稳定性和错误类型分布？

### 目标 3：把“可用性”做出来

当前方法的一个现实问题是要维护多个 stage adapter。期刊版非常适合增加一个“统一部署”方向的扩展，例如：

- 自动路由：根据问题类型选择 adapter
- adapter merging：将多个 stage 能力融合
- 单模型蒸馏：将多阶段专家能力压缩到一个统一模型

这一点很重要，因为它能把论文从“有效训练策略”提升到“可部署的完整系统”。

---

## 3. 期刊版建议主线

建议 TMM 版不要仅仅写成“ACM MM + 更多实验”，而是升级为下面这个版本：

## 推荐主线

### 从 `2D-CL` 扩展到 `2D-CL++: Curriculum Training and Expert Consolidation for Chart Understanding`

核心思路：

- 保留原有的二维课程学习主线
- 新增“专家整合 / 统一推理”模块
- 用更全面的 benchmark、鲁棒性实验和统计分析支撑结论

这样期刊版的新增贡献可以写成：

1. 提出一个更完整的 chart understanding curriculum framework，不仅解决训练顺序，还解决部署阶段的专家选择或融合问题。
2. 引入统一的 stage-aware inference / adapter consolidation 机制，降低多 adapter 带来的部署复杂性。
3. 在更多 benchmark、更多 setting、更多分析维度上系统验证方法。

如果你希望尽量少改方法，最低限度也建议加入一个轻量但明确的新模块，否则容易被审稿人认为只是“conference extension with more experiments”。

---

## 4. 方法扩展建议

下面按“推荐程度”排序。

## 方案 A：加入 `Stage-Aware Router`，优先推荐

### 核心思想

保留五个 stage adapter，不同问题由路由器自动选择最适合的 adapter。

### 可以怎么做

- 输入：图像 + 问题
- 输出：预测最合适的 stage
- 实现方式：
  - 小型分类器
  - 基于 LLM / VLM 的 task-type classifier
  - 基于问题模板和关键词的弱监督路由器

### 价值

- 回答当前 limitation 中的 adapter management 问题
- 让“每个 stage 是专家”这件事真正变成系统设计，而不只是分析现象
- 很适合放在 TMM 里作为新增方法点

### 可新增实验

- Oracle routing vs predicted routing
- Single best adapter vs router-based selection
- Router error analysis
- 路由开销与性能增益权衡

---

## 方案 B：加入 `Adapter Consolidation / Merging`

### 核心思想

把 Stage 1-5 的 adapter 能力合并成一个统一 adapter 或统一模型，兼顾性能与部署便利性。

### 可做方向

- 简单 merge：线性加权融合
- TIES / DARE / task arithmetic 风格的 adapter merge
- 以五个 stage adapter 为教师，对一个 unified adapter 蒸馏

### 价值

- 对应 limitation 中“多 adapter 难部署”
- 方法上有工程和系统价值
- 若效果接近多专家 routing，会很有说服力

### 风险

- 直接 merge 可能效果不稳定
- 若时间有限，建议先做蒸馏版 unified adapter，而不是复杂 merge

---

## 方案 C：加入 `Dynamic Difficulty Curriculum`

### 核心思想

当前 difficulty 是训练前一次性打分。期刊版可升级为动态难度：

- 难度不再固定
- 根据模型当前 loss、置信度、执行错误率、回答长度等动态调整样本排序

### 价值

- 把 `DACDS` 从静态数据排序升级为自适应 curriculum
- 更符合期刊对“方法深化”的期待

### 风险

- 实现复杂度更高
- 训练成本会上升
- 如果最终收益不大，性价比可能不如 Router 或 Consolidation

### 结论

如果时间有限，不建议把它作为第一新增点。

---

## 方案 D：加入 `Multilingual / Cross-Lingual Chart Understanding`

### 核心思想

扩展到中文图表、双语问题或跨语言图表问答。

### 价值

- 更符合 TMM 对多媒体真实应用场景的兴趣
- 与你中文写作环境也比较契合

### 风险

- 需要额外数据构建
- 评价集获取成本较高
- 可能让主线发散

### 结论

作为加分项适合，但不建议替代主线方法增强。

---

## 5. 必做补充实验

下面是我认为期刊版最值得补的实验，其中前 1-6 项基本可以视为“必做”。

## 5.1 更广的 benchmark 评测

当前 limitation 已经提到 `ChartBench` 和 `OpenCQA`，这正好可以作为期刊版最自然的扩展。

### 建议加入的数据集

- `ChartBench`
- `OpenCQA`
- 如可行，再补一个更偏真实场景或网页图表的数据集

### 实验目标

- 证明方法不是只对 ChartQA 有效
- 证明 curriculum 对不同数据来源、问题风格、图表风格都有效

### 结果呈现建议

- 主表：所有 benchmark 的总体结果
- 分表：按 question type / chart type / reasoning type 细分结果

---

## 5.2 多随机种子实验与显著性分析

当前稿件明确写了 single run，这在期刊里偏弱。

### 建议配置

- Full method、Standard SFT、1D-Task、1D-Difficulty 至少做 `3 seeds`
- 报告：
  - mean ± std
  - paired significance test

### 目的

- 强化结果可信度
- 尤其验证 `difficulty ordering` 带来的小幅提升是否稳定

---

## 5.3 数据规模扩展实验

当前文章的亮点之一是“12K 数据也有效”，但期刊版最好回答：

> 当数据规模变化时，2D-CL 还是否有效？

### 建议设置

- 10%
- 25%
- 50%
- 100%
- 150% 或 200%（如果可以扩数据）

### 需要比较的方法

- Standard SFT
- 1D task curriculum
- 1D difficulty curriculum
- Full 2D-CL

### 要回答的问题

- curriculum 在低资源时是否更有优势？
- 数据增多后优势会缩小还是持续存在？
- 哪个模块最依赖低资源设定？

这组实验非常重要，因为它能把“数据效率”从一句 claim 变成完整结论。

---

## 5.4 更细粒度的鲁棒性实验

建议增加真实图表常见扰动下的测试：

- 分辨率变化
- 图像压缩
- 颜色风格变化
- 字体变化
- 遮挡 / 局部模糊
- 坐标轴密集、标签重叠
- 图例位置变化

### 目标

- 检验 2D curriculum 是否真的带来更稳健的视觉-推理能力
- 回应“当前方法主要提升 reasoning，但未根本改善视觉编码”的潜在质疑

### 结果展示建议

- robustness 曲线
- 不同扰动强度下准确率下降幅度
- 与 baseline 的 relative drop 比较

---

## 5.5 更细粒度的错误分析

当前 qualitative analysis 还不够系统，建议做 error taxonomy。

### 可划分错误类型

- OCR / 文本识别错误
- 数值读取错误
- 计数错误
- 算术错误
- 多步推理链断裂
- 输出格式错误
- 图例 / 坐标轴映射错误
- 跨系列比较错误

### 分析目标

- 说明 2D-CL 主要减少了哪类错误
- 说明哪些错误仍然没有解决
- 为后续方法扩展提供依据

### 建议图表

- stacked bar
- confusion-style distribution
- case study table

---

## 5.6 Difficulty 机制验证实验

当前 difficulty score 由大模型打分，但缺少“打分是否可信”的验证。

### 建议补充

- 人工抽样标注一小部分样本难度，与 LLM difficulty 对齐
- 比较不同 difficulty scorer：
  - Qwen2.5-72B
  - GPT 系列或其他开源 LLM
  - 简单规则难度
- 比较不同 curriculum 排序方式：
  - easy-to-hard
  - hard-to-easy
  - random
  - competence-based sampling

### 目标

- 证明 difficulty 分数不是拍脑袋
- 证明 curriculum 的收益确实来自合理排序，而非偶然

---

## 6. 强烈建议补的提升实验

这些实验不一定都必须做，但做出来会明显提升期刊说服力。

## 6.1 更完整的 1D / 2D curriculum 对比

当前已有 task-only 和 difficulty-only，但还可以更系统：

- fixed stage order vs shuffled stage order
- easy-to-hard vs hard-to-easy
- stage-wise curriculum vs mixed-task curriculum
- static difficulty vs dynamic difficulty

### 要回答的问题

- 2D 的收益是否真的来自“两个维度同时建模”
- 横向和纵向 curriculum 是否存在交互效应

---

## 6.2 CRS 深入实验

目前已经证明 CRS 有效，但期刊版可以进一步细拆。

### 可补的 ablation

- code supervision vs chain-of-thought text supervision
- executable code vs non-executable pseudo-code
- full code trace vs abbreviated code trace
- verified code vs unverified code

### 目标

- 证明“代码”这种 supervision 形式比普通 reasoning text 更有效
- 证明“可执行验证”是关键，不只是输出更长

---

## 6.3 Adapter inheritance 深入实验

这是当前最强组件，值得进一步深挖。

### 建议对比

- sequential inheritance
- direct initialization from base
- cumulative multi-stage warm start
- partial inheritance
- only inherit visual stages / only inherit reasoning stages

### 目标

- 解释 inheritance 为什么最重要
- 更清楚地建立“能力递进”的因果关系

---

## 6.4 Inference cost 与部署实验

如果期刊版引入 router 或 unified adapter，这部分必须补。

### 指标

- 推理时延
- GPU 显存
- adapter 数量 / 存储开销
- 路由开销
- 性能-效率 tradeoff

### 目的

- 强化系统实用性
- 符合 TMM 对实际应用价值的偏好

---

## 6.5 Out-of-domain 泛化

建议专门构造一组更远分布测试：

- 学术论文中的图表
- 财经报告图表
- dashboard 截图
- 颜色、布局、标注风格明显不同的图表

### 目标

- 证明不是只学会了 synthetic style
- 对期刊审稿人关于 domain gap 的担忧进行正面回应

---

## 7. 可选扩展实验

如果时间和资源允许，可以继续加分。

## 7.1 多语言实验

- 中文图表问答
- 中英混合图表
- 英文图表 + 中文问题

适合做成一个附加章节，不建议替代主线。

## 7.2 人类评测

对描述质量、解释质量、代码可读性做人工评估。

## 7.3 校准性分析

- confidence vs correctness
- ECE / Brier score

如果方法提升了 reasoning，也可能改善模型置信度与正确性的匹配。

## 7.4 训练过程分析

- 不同 stage 的 loss transfer
- representation similarity
- stage 间知识迁移可视化

更适合作为分析加分项。

---

## 8. 推荐实验优先级

如果按照“投入产出比”排序，我建议：

### 第一优先级：必须完成

1. 更多 benchmark：`ChartBench / OpenCQA`
2. `3 seeds` 统计显著性
3. 数据规模实验
4. 系统化错误分析
5. difficulty 可靠性验证
6. 更完整的 1D vs 2D curriculum 对比

### 第二优先级：强烈建议

1. 新增 `Stage-Aware Router`
2. Router vs single adapter vs oracle adapter
3. inference cost / deployment tradeoff
4. CRS 深入 ablation

### 第三优先级：加分项

1. unified adapter 蒸馏或 merge
2. multilingual setting
3. out-of-domain 手工测试集
4. 人工评测

---

## 9. 期刊版论文结构建议

建议 IEEE TMM 版本按下面结构组织。

## 标题建议

可以保留当前标题风格，但建议略微突出期刊版的新点。例如：

- `Two-Dimensional Curriculum Learning for Chart Understanding`
- `Two-Dimensional Curriculum Learning and Expert Consolidation for Chart Understanding`
- `Curriculum-Guided Chart Understanding with Code-as-Reasoning Supervision`

如果加入 Router / Consolidation，第二种更合适。

## 正文结构建议

### 1. Introduction

比会议版更强调三件事：

- 图表理解的多能力耦合本质
- 现有方法重模型/重数据，轻训练组织
- 期刊版新增点：不仅优化训练，还提升统一推理与部署可用性

### 2. Related Work

建议比会议版更完整，增加：

- chart reasoning
- program supervision for VLM
- curriculum learning in multimodal systems
- model routing / adapter fusion / multi-expert inference

### 3. Preliminary and Problem Definition

期刊版可以加入更正式的问题定义、符号和任务拆解。

### 4. Proposed Method

建议拆成：

- 4.1 Two-Dimensional Curriculum
- 4.2 Code-as-Reasoning Supervision
- 4.3 Difficulty-Aware Sample Scheduling
- 4.4 Stage-Aware Router / Adapter Consolidation
- 4.5 Training and Inference Pipeline

### 5. Experimental Setup

重点补强：

- benchmark 更多
- baselines 更全
- multi-seed protocol
- robustness protocol
- statistical test

### 6. Main Results

### 7. Extended Analysis

这里可以包含：

- difficulty validity
- curriculum order analysis
- error taxonomy
- robustness
- efficiency
- deployment tradeoff

### 8. Discussion

建议新增讨论章节，不要只放 limitations。

可以讨论：

- curriculum 对多模态 reasoning 的一般意义
- 为什么 code supervision 比 text CoT 更适合 chart reasoning
- synthetic-to-real gap

### 9. Conclusion

---

## 10. 预计新增贡献写法

期刊版最终贡献建议写成 4 点左右，不要太散。

### 建议版本

1. 提出一个面向图表理解的增强版二维课程学习框架，在任务复杂度与样本难度两个维度上组织训练。
2. 设计 `CRS` 与改进的 curriculum scheduling / stage-aware inference 机制，使模型既能获得更强推理能力，也更便于部署。
3. 构建更系统的实验协议，覆盖多 benchmark、多模型、多数据规模、多随机种子以及鲁棒性与错误分析。
4. 证明课程学习对图表理解的收益不仅体现在最终准确率，也体现在数据效率、泛化能力、稳定性和部署实用性上。

---

## 11. 具体执行路线

下面给出一个现实可执行的路线，避免项目发散。

## 路线 A：稳妥版，最推荐

### 方法新增

- 增加 `Stage-Aware Router`

### 实验新增

- ChartBench / OpenCQA
- 3 seeds
- data scaling
- robustness
- error taxonomy
- difficulty validity
- router efficiency

### 优点

- 方法新增明确
- 实验闭环完整
- 风险可控
- 最符合“期刊扩展”预期

---

## 路线 B：实验加强版

### 方法不大改

- 保留原方法
- 重点做大规模补实验与深分析

### 风险

- 容易被质疑创新增量不足

### 适用场景

- 时间非常紧
- 不方便重新训练复杂模型

### 结论

可作为保底方案，但不是最佳方案。

---

## 路线 C：统一模型版

### 方法新增

- stage adapter consolidation / distillation

### 优点

- 学术和系统价值都强

### 风险

- 开发和调参成本较高
- 如果 unified 模型性能掉得多，会影响主线

### 结论

适合作为第二阶段增强，不建议一开始就押全部资源。

---

## 12. 最终建议

综合当前稿件状态，我最推荐的 TMM 扩展方案是：

> 保留 `2D-CL + CRS + DACDS` 主体，新增一个轻量但明确的 `Stage-Aware Router`，并系统补强 benchmark、统计显著性、数据规模、鲁棒性、难度可靠性与错误分析。

原因是：

- 它最大程度复用你现有 ACM MM 稿件
- 新增点清晰，不只是“补实验”
- 能正面回应当前 limitation 中最关键的两个问题：
  - 评测广度不足
  - 多 adapter 部署复杂
- 整体工作量与论文收益比较平衡

---

## 13. 建议的最小可发表包

如果希望尽快形成 TMM 初稿，建议至少做到下面这些内容：

- 保留原有全部实验
- 新增 `ChartBench + OpenCQA`
- 新增 `3 seeds`
- 新增 `data scaling`
- 新增 `difficulty validity`
- 新增 `error taxonomy`
- 新增 `Stage-Aware Router`
- 新增 `router vs best single adapter vs oracle`

做到这一步，论文就已经不再是简单的 ACM MM 扩写，而是具备了较明确的期刊增量。

---

## 14. 后续可立即开展的工作清单

### 第一周

- 确定期刊版主线：是否加入 Router
- 确定新增 benchmark 和可获得数据
- 整理现有代码与 adapter 产物

### 第二周

- 跑多 seed 与 data scaling
- 跑 1D/2D 更完整对比
- 开始整理 error taxonomy 标注规则

### 第三周

- 实现并测试 Router
- 跑新 benchmark
- 跑 robustness 实验

### 第四周

- 统一画图和表格
- 重写 introduction、method、experiments、discussion
- 补 discussion 和 limitations

---

## 15. 一句话版本

这篇 ACM MM 工作扩展成 IEEE TMM 的最佳路径，不是单纯“再多做几个实验”，而是：

> 用一个新的统一推理/部署模块把五阶段课程学习串成完整系统，再用更广、更稳、更细的实验把结论做扎实。
