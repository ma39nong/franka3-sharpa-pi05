# FR3 + Sharpa 部署结构

模型训练方式、模型动作协议和机器人执行速度是三个独立问题。

## 职责

| 层 | 当前文件 | 责任 |
| --- | --- | --- |
| 模型加载 | `model_profiles.py`, `model_env.sh`, `model_loader.py`, `serve.py` | checkpoint、模型参数结构、统计、输入输出适配、预热和模型服务 |
| 动作契约 | `config.py`, `protocol.py`, `action_adapter.py` | 58D、关节顺序、rad、绝对位置、30 Hz 和权重/统计身份 |
| 执行策略 | `profiles.py`, `policy_prepare.py` | 慢/中/快时间倍率、prepare、重新观测与 chunk 循环 |
| 公共运行层 | `run.py`, `motion_worker.py`, `ros_observation.py`, `feedback.py` | 观测、IPC、阶段报告和进程清理 |
| 设备层 | `sdk_direct_motion_ros.py`, `sdk_direct.py`, `ros_transport.py` | Sharpa SDK、FR3 网关、反馈与实际下发 |

所有速度使用相同推理服务、prepare 和设备执行代码。无需按 LoRA/full 创建两套机器人部署目录。客户端核对物理动作契约和 checkpoint/统计身份，训练方式仅为服务端模型信息。

当前同一分支注册 `full` 和 `lora`，由共享 shell 入口的 `--model` 选择，默认仍为全量。全量支持已验证的 AB4 normal 4000 步权重；LoRA 预设 30 Hz／58D absolute、VLM rank 16／Expert rank 32，尚无新 checkpoint，真实加载待验证。权重、统计、端口和日志配置分别保存，硬件执行共用。不能仅去掉参数树检查就声称旧 20 Hz/delta 权重可用。LoRA 若已合并为基础模型矩阵，参数结构可以与全量模型相同。无论训练方式，都必须分别确认频率、关节顺序和绝对/增量语义。

## Home 与滚动执行

默认读取指定 poses.yaml 的 powder_weighing 双臂 Home，双手读取 hand_home.json 的固定 Home（由左手实测角按同名关节映射）。模型每轮仍预测 50 帧，执行前 15 帧后重新观测推理，未执行预测不跨轮复用。`--execute-steps` 控制每轮动作数，`--num-chunks` 控制轮数（0 持续运行），不限于前两轮。详细来源和快照同步步骤见 README。

## 当前三档速度

这三档是共用 serial 执行器的**时间倍率预设**，不是旧平台三套规划器的完整移植，也不是经过实机验证的硬件速度上限。

| `--speed` | 模型时间倍率 | 默认前 15 帧主体下发点 | 30 Hz 下发主体时长 |
| --- | ---: | ---: | ---: |
| `slow` | 0.1 | 141 | 约 4.70 秒 |
| `medium` | 0.5 | 29 | 约 0.97 秒 |
| `fast` | 1.0 | 15 | 约 0.50 秒 |

时长不含初始 prepare、每段首帧衔接、观测和推理。prepare 默认仍为 0.03 rad/s，不随速度档位加速。未指定速度时保留历史默认 1.0；也可用 `--execute-speed-scale` 指定倍率，但不能与 `--speed` 同时使用。

只检查配置：

```bash
bash deploy/fr3_sharpa_runtime/runtime.sh --check --speed slow
bash deploy/fr3_sharpa_runtime/runtime.sh --check --speed medium
bash deploy/fr3_sharpa_runtime/runtime.sh --check --speed fast
```

执行端选择速度，例如：

```bash
bash deploy/fr3_sharpa_runtime/runtime.sh --execute --enable-hardware --speed slow --num-chunks 0
```

`fast` 目前仍为串行观测/推理/播放，不代表具备 RTG、非零速度跨段拼接或 Ruckig 平滑；SDK 下发也没有改为 100 Hz。这些能力属于后续 Sharpa 调度/规划实现，不能仅通过改速度档位宣称完成。旧平台的设计保留在 [历史参考](../../docs/reference_wuji_architecture.md)，不将其设备协议、关节切片或实机限值复制到 Sharpa。
