# subtask0: AutoDL 配置

目标是在 AutoDL RTX 5090 上搭好训练环境，并确认显卡、PyTorch 和深度学习依赖正常工作。

注意：合成图表数据不需要上服务器。`1合成数据.md` 优先在本地完成，AutoDL 主要用于后续 LoRA 微调、路由训练和大规模评测。

## 0.1 创建实例

推荐选择：

- GPU：`RTX 5090 32GB`
- 系统：`Ubuntu 22.04`
- 镜像：优先选 AutoDL 官方 `PyTorch + CUDA 12.x` 镜像
- Python：`3.10` 或 `3.11`
- 磁盘：建议至少 `100GB`，如果要缓存多个模型和数据集，建议 `200GB+`

一般不需要自己安装 NVIDIA 驱动。AutoDL 宿主机通常已经配置好驱动，容器里主要安装 Python 依赖。

## 0.2 进入工作目录

```bash
cd /root/autodl-tmp
git clone <你的仓库地址> ieee-tmm
cd ieee-tmm
```

如果暂时不用 git，也可以把整个项目压缩上传到 `/root/autodl-tmp/ieee-tmm`。

## 0.3 安装环境

```bash
bash autodl/setup_autodl.sh
```

这个脚本会做几件事：

- 创建 `tmm-chart` conda 环境
- 安装 `requirements.txt`
- 执行 `pip install -e .`
- 配置 HuggingFace 和 Torch 缓存目录
- 打印 `torch/cuda` 状态

## 0.4 激活环境

以后每次重新登录后执行：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate tmm-chart
```

## 0.5 检查 GPU

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

期望结果：

- `nvidia-smi` 能看到 RTX 5090
- `torch.cuda.is_available()` 输出 `True`

## 0.6 训练环境 smoke test

```bash
bash autodl/first_run.sh
```

如果只是检查 Python 包是否能导入，可以手动执行：

```bash
python - <<'PY'
import torch
import transformers
import peft
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
PY
```

## 0.7 后续任务入口

本任务可以在真正准备训练前再做。数据合成请先在本地完成 `1合成数据.md`，再把项目和 `data/synthetic/` 同步到 AutoDL。
