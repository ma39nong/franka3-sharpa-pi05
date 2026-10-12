# 本机 Sharpa 20 Hz 训练

使用修复后的相机映射：头部 cam0、左腕 cam2、右腕 cam1。
本配置从官方 `pi05_base` 开始微调，使用新实验目录，不恢复旧相机映射的训练。

## 数据与路径

- 配置：`pi05_fr3_sharpa_tier_a_b1_20hz`
- A：`/media/descfly/833BD1AED8D615E8/data/sharpa_lerobot/tier_a_20hz`
  （30 episodes，17,543 frames），采样概率 70%。
- B1：`/media/descfly/833BD1AED8D615E8/data/sharpa_lerobot/tier_b1_20hz`
  （84 episodes，49,772 frames），采样概率 30%。
- 基础权重：`/home/descfly/.cache/openpi/openpi-assets/checkpoints/pi05_base/params`
- 归一化统计：`assets/pi05_fr3_sharpa_tier_a_b1_20hz/fr3_sharpa/tier_a_b1_20hz/norm_stats.json`
- 训练输出：`checkpoints/pi05_fr3_sharpa_tier_a_b1_20hz/sharpa_camfix_a70_b130_20hz_30k/`

以上相对路径均相对于本仓库。数据集权重按整个数据集的概率归一化，再均分到
该数据集各帧，因此 70%/30% 不会因 B1 帧数更多而被改变。

## 独立训练环境

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda .venv/bin/python -c \
  'import jax; print(jax.__version__); print(jax.devices())'
```

使用仓库 `.python-version` 的 Python 3.11 和 `uv.lock` 的固定依赖。
本机已验证另一工程的环境 `/home/descfly/LTT/pi05/.venv/bin/python`：
JAX/jaxlib 0.5.3、Flax 0.10.2、Torch 2.7.1+cu128、NumPy 1.26.4，能够识别
RTX 5090 的 CUDA device。若复用该环境，在下面命令之前执行：

```bash
export SHARPA_TRAIN_PYTHON=/home/descfly/LTT/pi05/.venv/bin/python
```

启动脚本将当前仓库的 `src` 放到 PYTHONPATH 首位，因此复用环境时仍加载本仓库
修复后的代码。独立环境配置后可执行 `unset SHARPA_TRAIN_PYTHON`。
本流程运行 JAX 训练入口，不需要复制 PyTorch 的 Transformers 替换文件。

## 计算统计与开始训练

首次训练先计算匹配这两个数据集和当前动作变换的统计，避免使用仓库根目录的
历史 `norm_stats.json`。本脚本使用训练配置的输入变换，但不做图像归一化训练。

```bash
bash scripts/train_fr3_sharpa_20hz.sh --compute-norm-stats
bash scripts/train_fr3_sharpa_20hz.sh
```

统计保存位置与训练读取位置一致。默认统计脚本遍历拼接数据集，统计由所有帧
汇总得到；70%/30% 的随机采样仅用于训练，不用于完整统计遍历。

若显存不足，降低 batch size 会改变每步样本数：

```bash
bash scripts/train_fr3_sharpa_20hz.sh --batch-size 4 --exp-name sharpa_camfix_a70_b130_20hz_30k_bs4
```

仅恢复同一修正映射实验时使用：

```bash
bash scripts/train_fr3_sharpa_20hz.sh --resume
```

## 当前超参数

| 参数 | 配置 |
|---|---|
| 模型 | π0.5，bfloat16 计算 |
| PaliGemma / Action Expert | gemma_2b_lora / gemma_300m_lora |
| LoRA rank / alpha | Gemma 2B：16 / 16；Action Expert：32 / 32 |
| 可训练参数 | 两个语言主干的 LoRA；视觉编码器、动作投影、时间嵌入等未冻结参数 |
| 状态 / 动作维度 | 58 / 58 |
| 动作表示 | 两臂为相对当前状态的关节位置增量；双手为绝对关节位置；单位 rad |
| 数据 / 动作频率 | 20 Hz / 20 Hz |
| 动作窗口 | 50 步，50 个 0.05 秒 tick；目标偏移 0–2.45 秒 |
| 图像 | 3 路 RGB，模型输入 224×224 |
| 文本长度 | 最多 320 tokens（真实样本超过原 256 上限）；任务来自数据集 task |
| 归一化 | quantile，使用 q01/q99 |
| batch size / workers | 8 / 8 |
| 训练步数 | 30,000 |
| 优化器 | AdamW，β1=0.9，β2=0.95，eps=1e-8 |
| 学习率 | 1,000 步 warmup + cosine；峰值 2.5e-5，末值 2.5e-6 |
| weight decay / 梯度裁剪 | 1e-10 / 全局范数 1.0 |
| EMA / WandB | 关闭 / 关闭 |
| seed / FSDP devices | 42 / 1 |
| 日志 / 保存间隔 | 每 10 步 / 每 2,000 步 |
| 长期保留间隔 | 10,000 步；最终保存遵循 checkpoint 管理逻辑 |

训练日志包括控制台输出、实验目录下的 `train_metrics.csv` 与
`train_summary.json`。32 GB 显存是否足够承载 batch 8，仍需实际训练验证。
