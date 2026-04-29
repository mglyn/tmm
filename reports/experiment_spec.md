# TMM 实验规格冻结

本文件把论文中的实验承诺转成工程侧可执行规格，不修改计划文件本身，只作为后续实现与回填的固定依据。

## 主线

- 方法：`2D-CL + CRS + DACDS + Stage-Aware Router`
- 主模型：`Qwen2.5-VL-7B-Instruct`
- 训练范式：LoRA + 五阶段顺序继承
- 路由范围：`S2/S3/S4/S5`

## 关键数据规格

- synthetic chart 数量目标：`1,973`
- 训练总样本目标：`12,427`
- stage 样本目标：
  - `S1`: `1,775`
  - `S2`: `1,775`
  - `S3`: `1,775`
  - `S4`: `5,327`
  - `S5`: `1,775`
- router holdout 目标：`1,200`

## 关键训练规格

- LoRA：`r=8`, `alpha=16`, `dropout=0.1`
- 有效 batch size：`16`
- 各阶段 epoch：`10`
- 学习率：
  - `S1`: `1e-4`
  - `S2`: `5e-5`
  - `S3`: `3e-5`
  - `S4`: `3e-5`
  - `S5`: `2e-5`

## 最低实验闭环

1. `Standard LoRA SFT`
2. `2D-CL` 五阶段专家
3. `SAR` 路由器
4. `ChartQA` 主结果
5. 消融、路由、数据规模、跨数据集、错误分析、难度可靠性

## TODO-EXP 映射原则

- `TODO-EXP-01 ~ 03`: 摘要与引言总述指标
- `TODO-EXP-04 ~ 21`: 正文主实验、消融、泛化、鲁棒性
- `TODO-EXP-22 ~ 32`: 分析与附录结果

所有自动汇总与论文回填逻辑统一读取 `reports/todo_exp_map.json` 和 `outputs/paper/final_metrics.json`，默认回填目标为 `tmm_paper/main.tex`。
