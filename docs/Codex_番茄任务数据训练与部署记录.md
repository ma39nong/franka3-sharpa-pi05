# FR3/Wuji 番茄任务：微调方法与实验记录

本文用于公开说明本仓库如何把 OpenPI `pi05_base` 微调到 FR3 双臂 + Wuji 灵巧手上。原始训练日志、数据集、norm stats 和 checkpoint 体积较大，均被 `.gitignore` 排除；因此本文记录可复现的方法、关键参数和已经确认的结果，但不代表这些训练资产已经随 Git 仓库上传。

任务指令为：

```text
Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.
```

## 1. 方法概览

- 基础模型：OpenPI `pi05_base`
- 微调方式：LoRA，`gemma_2b_lora` + `gemma_300m_lora`
- 硬件：单张 RTX 5090 32 GB
- 输入：头部、左腕、右腕三路 RGB，机器人状态和英文任务指令
- 输出：未来 50 步、每步 54 维的位置指令
- 动作顺序：`[左臂 7, 左手 20, 右臂 7, 右手 20]`
- batch size：8
- 训练步数：20,000
- checkpoint：每 2,000 步保存，保留 10,000 和最终 19,999
- EMA：关闭
- 初始化权重：`PartialCheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`
- 学习率：cosine schedule，warmup 1,000，peak `2.5e-5`，末端 `2.5e-6`

选择 LoRA 是因为这套模型做全量微调时，单卡 32 GB 无法容纳参数、Adam 状态和 EMA；LoRA 配合关闭 EMA 可以在该机器上稳定训练。

实现入口：

- 训练配置：[`src/openpi/training/config.py`](../src/openpi/training/config.py)
- FR3/Wuji 数据变换：[`src/openpi/policies/fr3_wuji_policy.py`](../src/openpi/policies/fr3_wuji_policy.py)
- 数据转换：[`examples/fr3_wuji/convert_dataset.py`](../examples/fr3_wuji/convert_dataset.py)
- 数据校验：[`examples/fr3_wuji/verify_dataset.py`](../examples/fr3_wuji/verify_dataset.py)
- 20 Hz 数据准备：[`scripts/prepare_fr3_wuji_20hz.py`](../scripts/prepare_fr3_wuji_20hz.py)
- 20 Hz 训练入口：[`scripts/train_fr3_wuji_20hz.sh`](../scripts/train_fr3_wuji_20hz.sh)

## 2. 数据约定

### 2.1 原始数据

早期采集数据的主要结构是：

| 项目 | 内容 |
| --- | --- |
| `observation.state` | 108 维，包含位置和速度 |
| `action` | 54 维，原顺序为 `[左臂 7, 右臂 7, 左手 20, 右手 20]` |
| 相机 | `cam0`、`cam1`、`cam2` 三路 RGB |
| 数据 FPS | 30 |
| 未使用字段 | depth、`engaged` |

最早 65 条原始 episode 使用旧 LeRobot 格式，而且三路相机存在各自的时间基准。它们不能只靠修改元数据直接交给官方 loader；转换脚本通过每个 episode 自带的 loader 正确取帧，再写成标准 LeRobot 数据集。

磁盘上的 108 维 state 被在线切片为 54 维：

```text
[state 0:7] + [state 28:48] + [state 14:21] + [state 68:88]
 = 左臂 7       左手 20          右臂 7          右手 20
```

磁盘 action 被在线重排：

```text
[action 0:7] + [action 14:34] + [action 7:14] + [action 34:54]
 = 左臂 7         左手 20           右臂 7          右手 20
```

随后仅对两条机械臂的 14 个维度做 delta action；40 个手部维度保持绝对位置。推理输出会执行反归一化和 inverse delta，所以部署端收到的是绝对位置，不应再叠加一次当前状态。

### 2.2 为什么不使用深度

当前 `pi05` 输入只使用三路 RGB，模型没有深度输入通道。原始数据中的 depth 还以嵌套结构存放在 parquet 中，读取长 episode 时会触发 PyArrow 的 chunked nested-array 限制。因此转换时只保留 RGB、state、action 和 task。LeRobot 本身可以保存深度；这里只是当前模型不使用它。

### 2.3 数据集演进

| 阶段 | 数据 | 配置 / 实验名 | 说明 |
| --- | --- | --- | --- |
| 管线冒烟 | 1 episode | `pi05_fr3_wuji` / `smoke_episode24` | 验证 54 维变换、训练、存盘和推理形状 |
| 第一轮 | 52 episodes | `pi05_fr3_wuji` / `tomato_lora` | 转换中途的完整子集 |
| 第二轮 | 65 episodes，169,282 帧 | `pi05_fr3_wuji` / `tomato_lora_65ep` | 完成旧数据转换后的正式训练 |
| 第三轮 | 65 + 60 = 125 episodes，285,436 帧 | `pi05_fr3_wuji_125ep` / `tomato_lora_125ep_v2` | 物理合并后的联合训练 |
| 第四轮 | 42 + 102 + 125 = 269 episodes | `pi05_fr3_wuji_269ep` / `tomato_lora_269ep` | 三个数据集等权采样 |
| 0918 A/B | A: 88,986 帧；B: 167,484 帧 | 历史 `pi05_fr3_wuji_weighted` / `tomato_lora_0918_a70_b30` | A/B 按 0.7/0.3 采样；该历史配置后来已演进 |
| 最新 20 Hz | 同一 0918 A/B 数据 | `pi05_fr3_wuji_20hz` / `tomato_lora_0918_20hz` | 30 Hz 源数据上按最近帧构造 20 Hz、50 步动作轨迹 |

注意：实验名描述的是训练当时的配置。尤其 `tomato_lora_0918_a70_b30` 使用的是当时的 A/B 0.7/0.3 配置，不能仅凭当前同名配置推断历史 checkpoint 的数据来源。

## 3. Norm stats

Norm stats 必须用“本轮训练实际使用的数据”，在和训练相同的在线 transform 之后计算。不能混用 108 维统计与 54 维模型，也不能简单平均多个数据集各自的 `norm_stats.json`。

普通单数据集配置使用：

```bash
JAX_PLATFORMS=cpu uv run --no-sync scripts/compute_norm_stats.py \
  --config-name pi05_fr3_wuji
```

125-episode 联合数据使用：

```bash
JAX_PLATFORMS=cpu uv run --no-sync examples/fr3_wuji/compute_norm_stats.py \
  --config-name pi05_fr3_wuji_125ep
```

最新 20 Hz 路径由准备脚本计算统计。它扫描两个数据集的全部帧，先做 54 维重排和机械臂 delta，再按 20 Hz 时间网格选取 50 步目标，并写出 `norm_stats.json` 和 `provenance.json`。

```bash
uv run --no-sync scripts/prepare_fr3_wuji_20hz.py
```

统计会被复制到 checkpoint 的 `assets/` 中。部署必须携带与该 checkpoint 对应的 stats，不能只复制 `params/`。

## 4. 标准微调流程

以下命令均在仓库根目录运行。

### 4.1 安装与检查数据

按主 README 创建环境后，先验证转换后的 LeRobot 数据：

```bash
uv run --no-sync python -m examples.fr3_wuji.verify_dataset \
  --repo-id fr3_wuji/tomato \
  --episodes 4
```

校验至少应覆盖：

- 三路 RGB 能解码且时间对齐；
- state 输入为 108 维或已排好顺序的 54 维；
- action 为 54 维；
- transform 后 state/action 都是 54 维；
- action chunk 不跨 episode 边界；
- task prompt 与部署时完全一致。

### 4.2 先跑冒烟训练

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --no-sync scripts/train.py pi05_fr3_wuji \
  --exp-name smoke_new_data \
  --batch-size 2 \
  --num-train-steps 20 \
  --save-interval 10 \
  --no-wandb-enabled \
  --overwrite
```

通过标准是 loss 为有限值、checkpoint 能保存，并且加载 checkpoint 后输出形状为 `(50, 54)`。冒烟测试不能用于评价任务效果。

### 4.3 正式训练

65-episode 配置曾使用：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --no-sync scripts/train.py pi05_fr3_wuji \
  --exp-name tomato_lora_65ep \
  --log-interval 10
```

125-episode 联合训练曾使用：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --no-sync scripts/train.py pi05_fr3_wuji_125ep \
  --exp-name tomato_lora_125ep_v2 \
  --log-interval 10
```

最新 0918 A/B、20 Hz 训练使用统一入口：

```bash
bash scripts/train_fr3_wuji_20hz.sh --log-interval 10
```

该脚本会先执行数据准备，再启动：

```text
scripts/train.py pi05_fr3_wuji_20hz --exp-name tomato_lora_0918_20hz
```

`scripts/prepare_fr3_wuji_20hz.py` 和 `src/openpi/training/config.py` 目前包含本机数据根目录。其他机器复现时，必须先把 0918 A/B 数据放到可访问的位置并修改对应路径；数据集本身没有随 Git 上传。

如果想保留可提交的精简日志，可以将标准输出写到自定义文本文件，再从中摘录配置和指标。`logs/`、`wandb/`、`checkpoints/` 与所有 `*.log` 默认不会被 Git 跟踪。

### 4.4 断点续训

使用同一个 config 和 `exp-name`，增加 `--resume`：

```bash
uv run --no-sync scripts/train.py pi05_fr3_wuji_20hz \
  --exp-name tomato_lora_0918_20hz \
  --resume
```

不要同时使用 `--resume` 和 `--overwrite`。恢复训练依赖完整的 `train_state/`；只有 `params/` 的部署包不能恢复优化器状态。

## 5. 已确认的训练结果

### 5.1 训练 loss

以下数字来自本机保留的原始 stdout 日志。它们是训练集指标，不是成功率。

| 实验 | step 0 | step 10,000 | step 19,900 | 状态 |
| --- | ---: | ---: | ---: | --- |
| 65 episodes / `tomato_lora_65ep` | 1.3447 | 0.0831 | 0.0410 | 完成 20,000 steps |
| 0918 A/B 0.7/0.3 / `tomato_lora_0918_a70_b30` | 1.3743 | 0.0748 | 0.0434 | 完成 20,000 steps |
| 0918 A/B 20 Hz / `tomato_lora_0918_20hz` | 1.3786 | 0.0705 | 0.0436 | 完成 20,000 steps |

20 Hz 训练的源观测与视频仍为 30 Hz。未来动作目标使用最近源帧映射到 20 Hz 网格，50 步覆盖 2.5 秒；没有为此重新编码视频。

### 5.2 125-episode 离线回放

125-episode 模型曾从旧、新数据各选 8 条 episode，每条取约 20%、50%、80% 三个观测窗口，共 48 个窗口。相同采样噪声下，比较 checkpoint 10,000 与 19,999 对未来 50 步示范动作的 MAE：

| 部位 | 10,000 步 MAE | 19,999 步 MAE | 降低 |
| --- | ---: | ---: | ---: |
| 左臂 | 0.0457 | 0.0297 | 35% |
| 左手 | 0.0512 | 0.0381 | 26% |
| 右臂 | 0.0228 | 0.0161 | 30% |
| 右手 | 0.0417 | 0.0303 | 28% |

整体 MAE 从 0.0433 降到 0.0312。该评估来自训练集回放，没有独立测试集，不能替代真机闭环成功率。

## 6. 部署 checkpoint

建议复制完整的最终 step 目录，例如：

```text
checkpoints/pi05_fr3_wuji_20hz/tomato_lora_0918_20hz/19999/
```

纯推理至少需要：

```text
19999/
├── params/
└── assets/
    └── fr3_wuji/
        └── 0918_20hz/
            └── norm_stats.json
```

数据准备阶段生成的 `provenance.json` 位于本地 `assets/pi05_fr3_wuji_20hz/fr3_wuji/0918_20hz/`，当前 checkpoint 保存逻辑只自动复制 `norm_stats.json`。如需完整追溯数据来源，应另行保存 provenance 文件。

`train_state/` 仅在恢复训练时需要。启动最新模型服务可参考：

```bash
uv run --no-sync scripts/serve_policy.py --port 8006 policy:checkpoint \
  --policy.config pi05_fr3_wuji_20hz \
  --policy.dir /path/to/19999
```

推理客户端必须发送：

- `observation/image`
- `observation/left_wrist_image`
- `observation/right_wrist_image`
- `observation/state`（108 维原布局，或已按约定排列的 54 维）
- `prompt`

模型输出已经是 `[左臂 7, 左手 20, 右臂 7, 右手 20]`。真机第一次运行必须限速、限幅并准备急停，重点检查当前状态到首个动作、chunk 之间的衔接和手部关节跳变。

## 7. 复现边界与注意事项

1. Git 仓库只包含代码与本文；训练数据、原始日志、WandB 本地目录、norm stats 和 checkpoint 没有上传。
2. 逻辑 `repo_id` 不等于公开可下载的数据。复现者需要另外取得对应数据，并修正本地路径或 LeRobot cache 映射。
3. `pi05_base` 的 action 维度与本项目不同，必须使用 partial loader；形状不匹配的 action/state projection 会随机初始化。
4. prompt、关节顺序、单位、相机键和 norm stats 必须与训练时一致。
5. 不要将低 train loss 或训练集回放 MAE 当作真机成功率；最终模型仍需留出测试 episode 和安全的真机闭环评估。
6. 如果希望把某次实验日志公开，建议提交脱敏后的摘要或 CSV，而不是强制加入整个 `wandb/` 或 `checkpoints/` 目录。

更早期的数据转换细节和踩坑记录见 [`0906微调.md`](0906微调.md)，分阶段检查清单见 [`fr3_wuji_finetune_todo.md`](fr3_wuji_finetune_todo.md)。
