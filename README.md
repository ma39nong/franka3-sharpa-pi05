# OpenPI FR3 双臂 + 双 Sharpa π0.5 番茄协作任务

更新日期：2026-10-08

本仓库基于 Physical Intelligence 的 OpenPI，面向 **双 FR3 机械臂 + 双 Sharpa 灵巧手** 的番茄双手协作任务：右手抓取并移动到中间，左手摘取番茄，再由左手放入碗中。

当前仓库同时保留原有的 FR3 + Wuji 54 维工程。Wuji 配置和部署功能不变；Sharpa 使用独立的 58 维数据协议、模型配置和离线验收流程。

## 当前状态

- Sharpa 模型配置：`pi05_fr3_sharpa`
- 模型：π0.5，`action_dim=58`，`action_horizon=50`
- 动作顺序：左 FR3 7 + 左 Sharpa 22 + 右 FR3 7 + 右 Sharpa 22
- 输入：三路 RGB、58 维机器人状态、任务文本
- 当前正式训练策略沿用 Wuji 工程：
  - SigLIP 全量训练
  - Gemma 2B 主干冻结 + LoRA
  - Gemma 300M Action Expert 主干冻结 + LoRA
  - Action Head 全量训练
  - Timestep MLP 全量训练
  - AdamW、cosine warmup、关闭 EMA
- 官方 `pi05_base` checkpoint 已在本机完成真实兼容性验证
- 已使用 `episode284` 真实 batch 完成 forward/backward、单次 optimizer step、参数和 optimizer state 保存恢复验证
- 尚未启动 Sharpa 长时间训练，也没有连接或控制硬件

Sharpa 的训练配置位于 [`src/openpi/training/config.py`](src/openpi/training/config.py)，模型 freeze filter 位于 [`src/openpi/models/pi0_config.py`](src/openpi/models/pi0_config.py)。

## Checkpoint 兼容性

本机 checkpoint 路径：

```text
/home/user/checkpoints/pi05_base/params
```

Sharpa 使用 `PartialCheckpointWeightLoader`。官方基础权重的动作投影宽度为 32，Sharpa 为 58，因此只有以下动作投影参数重新初始化：

```text
action_in_proj/kernel

action_out_proj/bias

action_out_proj/kernel
```

其余基础参数严格按名称、形状和 dtype 检查。当前验证结果：

```text
目标参数叶：71
checkpoint 参数叶：51
unexpected：0
非动作投影 shape mismatch：0
合并后参数叶：71
```

LoRA adapter 参数是目标 LoRA 结构新增的参数，由目标模型初始化；这不改变 Gemma 主干冻结策略。

在其他机器上使用时，应把 `src/openpi/training/config.py` 中 Sharpa loader 的本地路径替换为可访问的 checkpoint 路径，或改为官方 GCS 路径。

## 训练策略与参数

Sharpa 继承原 `pi05_fr3_wuji` 的训练策略，不更换 model variant：

| 参数 | 当前 Sharpa 值 |
| --- | --- |
| PaliGemma | `gemma_2b_lora` |
| Action Expert | `gemma_300m_lora` |
| 学习率 | cosine decay，warmup 1000 步 |
| peak / end lr | `2.5e-5` / `2.5e-6` |
| Optimizer | AdamW，`b1=0.9`，`b2=0.95` |
| weight decay | `1e-10` |
| gradient clip | `1.0` |
| batch size | `8` |
| 训练步数 | `20,000` |
| checkpoint | 每 `2,000` 步保存，`keep_period=10,000` |
| EMA | 关闭 |
| WandB | 关闭 |

训练入口仍然是：

```bash
uv run --no-sync scripts/train.py pi05_fr3_sharpa \
  --exp-name <experiment-name> \
  --log-interval 10
```

正式训练前应确认本机 checkpoint、Sharpa 数据目录和 norm stats 路径均可访问。本 README 不代表已经启动过正式训练。

## Sharpa 数据

ROS bag 转换和 LeRobot 导出工具位于 [`examples/fr3_sharpa/`](examples/fr3_sharpa/)：

- [`convert_rosbag.py`](examples/fr3_sharpa/convert_rosbag.py)：读取 ROS bag，按 58 维协议对齐状态、动作和三路图像
- [`export_lerobot.py`](examples/fr3_sharpa/export_lerobot.py)：导出 LeRobot v2 数据集
- [`fr3_sharpa_protocol.py`](src/openpi/policies/fr3_sharpa_protocol.py)：定义关节名称、顺序和 58 维协议
- [`fr3_sharpa_policy.py`](src/openpi/policies/fr3_sharpa_policy.py)：定义 Sharpa 输入输出变换

已验证的 episode284 数据：

```text
30 Hz：/home/user/franka_teleop_data/converted/fr3_sharpa_episode284
20 Hz causal：/home/user/franka_teleop_data/converted/fr3_sharpa_episode284_20hz_causal2
```

30 Hz LeRobot 数据为 1547 帧，nominal FPS 为 30；20 Hz 因果版本为 1031 帧，nominal FPS 为 20。两种版本的时间间隔都没有发现缺帧间隔。

### 30 Hz 与 20 Hz 结论

- 30 Hz 的 50-step action horizon 覆盖约 1.67 秒。
- 20 Hz 的 50-step action horizon 覆盖 2.5 秒。
- 当前 30 Hz 版本使用最近样本同步，可能选到目标时刻之后的观测。
- 20 Hz causal 版本只使用目标时刻之前的最新样本，观测延迟已记录在 `report.json`。
- 30 Hz 也可以采用同样的 causal 同步，但需要重新生成数据并重新计算 norm stats。
- 当前 episode284 没有发现由裁剪或缺帧造成的动作不连续；快速动作峰值仍集中在右手抓取、左手摘取等动作变化附近。

这些结论来自 episode284 的离线测量。抓取、移动、摘取和放碗的语义阶段没有写入 bag 标签，因此动作峰值只能作为阶段变化的代理，不能替代独立任务成功率评估。

## 离线验收

验收内容包括：

- 本地 `pi05_base` 权重初始化
- episode284 真实 batch 的 forward/backward
- SigLIP、Gemma 2B LoRA、Gemma 300M LoRA、Action Head、Timestep MLP 梯度统计
- 单次正式 optimizer step
- 可训练参数更新、冻结参数不更新
- 参数、optimizer state、global step 保存恢复
- 恢复后确定性 forward
- 20 Hz / 30 Hz 数据 FPS、重复率、时间戳年龄和动作峰值对比

本机原始报告位于：

```text
/home/user/franka_teleop_data/converted/fr3_sharpa_episode284/report.json
/home/user/franka_teleop_data/converted/fr3_sharpa_episode284_20hz_causal2/report.json
```

大型数据、视频和 checkpoint 不提交到 GitHub。

## 原 Wuji 工程

原有 Wuji 配置仍为 54 维、30 Hz、50-step action horizon。其训练、服务和硬件执行入口保持不变。原有部署说明见：

- [FR3_Wuji_pi05部署使用说明.docx](FR3_Wuji_pi05部署使用说明.docx)
- [番茄任务数据训练与部署记录](docs/Codex_番茄任务数据训练与部署记录.md)
- [文档索引](docs/README.md)

切换 Wuji 权重时，仍需使用对应的 54 维适配器、norm stats 和部署入口，不要将 Wuji checkpoint 直接当作 Sharpa 58 维 checkpoint 使用。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/openpi/models/` | π0、π0.5 和相关模型实现 |
| `src/openpi/training/` | 训练配置、数据加载、归一化和 checkpoint 逻辑 |
| `src/openpi/policies/` | Wuji、Sharpa 以及其他平台的输入输出变换 |
| `examples/fr3_sharpa/` | Sharpa ROS bag 转换、导出和检查工具 |
| `examples/fr3_wuji/` | 原 Wuji 数据转换和检查工具 |
| `scripts/` | 训练、统计、数据准备和辅助脚本 |
| `deploy/` | 原 Wuji 模型服务和硬件执行入口 |
| `docs/` | 训练记录、部署说明和历史资料 |

## 许可证

项目代码许可证见 [LICENSE](LICENSE)，Gemma 相关条款见 [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)。
