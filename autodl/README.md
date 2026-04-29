# AutoDL 使用说明

这套环境包默认给 `AutoDL RTX 5090` 使用，思路不是“自己装 NVIDIA 驱动”，而是：

1. 在 AutoDL 选一个已经带好 CUDA 和 PyTorch 的镜像
2. 用本目录脚本补齐项目依赖、缓存目录和首轮 smoke test

## 你需要自己装 NVIDIA 驱动吗

一般不需要。

在 AutoDL 上：

- 宿主机 NVIDIA 驱动通常已经装好
- 容器里你主要关心的是 `CUDA runtime`、`PyTorch`、`transformers`、`flash-attn` 等用户态依赖
- 最稳妥的方式是直接选 AutoDL 的官方 `PyTorch` 镜像，而不是自己手搓驱动

## 推荐镜像选择

优先选这类镜像：

- Ubuntu 22.04
- Python 3.10 或 3.11
- PyTorch 2.5+ 或更新
- CUDA 12.x

如果有多个候选：

- 优先选已经验证过 `torch.cuda.is_available()` 为真的 PyTorch 镜像
- 不建议先选纯 Ubuntu 裸镜像再自己装整套 CUDA/PyTorch

## 首次上机推荐步骤

```bash
cd /root/autodl-tmp
git clone <你的仓库地址> ieee-tmm
cd ieee-tmm
bash autodl/setup_autodl.sh
bash autodl/first_run.sh
```

## 目录说明

- `environment.yml`: conda 环境定义
- `setup_autodl.sh`: 创建环境、安装依赖、准备缓存目录
- `first_run.sh`: 打印 GPU/torch 信息并执行 smoke test

## 常用命令

激活环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tmm-chart
```

数据构建：

```bash
python scripts/build_dataset.py --config configs/base_experiment.yaml
```

标准 SFT：

```bash
python scripts/train_experiment.py --config configs/base_experiment.yaml --mode sft
```

五阶段训练：

```bash
python scripts/train_experiment.py --config configs/base_experiment.yaml --mode curriculum
```

路由器：

```bash
python scripts/train_router.py --config configs/base_experiment.yaml
```

评测：

```bash
python scripts/run_eval.py --config configs/base_experiment.yaml
```

## 论文编译

如果镜像里没有 LaTeX：

```bash
sudo apt-get update
sudo apt-get install -y latexmk texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra
```

然后编译：

```bash
cd tmm_paper
bash -lc "latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex"
```

## 建议

真正上 5090 时，先不要直接跑全量训练。最稳的顺序是：

1. `first_run.sh`
2. `run_smoke_test.py`
3. `train_experiment.py --mode sft --limit 64`
4. 小样本 curriculum
5. 再放大全量训练
