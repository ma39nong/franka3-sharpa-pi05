# FR3 + Sharpa 本机部署

当前模型：π0.5 全量微调、30 Hz、原生 58D、50 帧、全部为绝对关节目标，单位 rad。执行路径是 policy server → runtime → FR3 Safety Gateway / Sharpa SDK direct。LoRA/full 参数差异由模型加载层处理，速度档位共用同一个执行器。

## 全量与 LoRA 统一入口

日常启动说明按模型分别放在 [全量目录](../fr3_sharpa_full/README.md) 和 [LoRA 目录](../fr3_sharpa_lora/README.md)。本目录保留公共实现及硬件工具，原有启动脚本继续有效。

全量和 LoRA 共存在 `chunk-base-test`，共用 Home、SDK、prepare、三档速度和执行循环。`model_profiles.py` 集中注册模型，`model_env.sh` 在导入 Python 配置前选择模型。不传 `--model` 默认全量，现有工作机命令继续有效。模型终端与执行终端需要选择相同模型。

| `--model` | 参数变体 | 时间轴／动作 | 默认端口 | 模型配置 |
| --- | --- | --- | ---: | --- |
| `full` | `gemma_2b` + `gemma_300m` | 30 Hz／58D absolute | 18765 | `.deployment/local.env` 的 `SHARPA_CHECKPOINT`、`SHARPA_NORM_STATS` |
| `lora` | `gemma_2b_lora` + `gemma_300m_lora` | 30 Hz／58D absolute | 18766 | `.deployment/lora.env` 的 `SHARPA_LORA_*` |
| `full-delta` | `gemma_2b` + `gemma_300m` | 30 Hz／双臂 delta、双手 absolute；服务输出 absolute | 18767 | `.deployment/full_delta.env` 的 `SHARPA_FULL_DELTA_*` |

```bash
bash deploy/fr3_sharpa_runtime/serve.sh --list-models
bash deploy/fr3_sharpa_runtime/serve.sh --model full
bash deploy/fr3_sharpa_runtime/runtime.sh --model full --check --speed medium
# 切换 LoRA 时，两端都选择 lora：
bash deploy/fr3_sharpa_runtime/serve.sh --model lora
bash deploy/fr3_sharpa_runtime/runtime.sh --model lora --check --speed slow
```

`--model` 和 `--list-models` 是 shell 入口参数。端口可分别通过 `SHARPA_FULL_PORT`、`SHARPA_LORA_PORT` 设置。LoRA 复用机器配置，但不回退使用全量权重、统计或日志。旧 LoRA 专用脚本保留为兼容别名。

全量 delta 权重已接入独立配置，工作机配置与命令见 [全量 Delta 部署](../fr3_sharpa_full/DELTA.md)；该配置已通过真实 GPU 假观测推理，现场运行待验证。

新的 LoRA checkpoint 尚未提供，目前只完成代码和配置测试，真实加载与工作机实机验证待完成。

权重接入步骤见 [LoRA 部署说明](../fr3_sharpa_lora/README.md)。

## 本机配置

在仓库根目录运行。首次使用可通过已有 OpenPI Python 环境生成本机配置：

```bash
/home/descfly/LTT/franka3-sharpa-pi05/.venv/bin/python \
  deploy/fr3_sharpa_runtime/setup_local.py
```

默认绑定：

| 项目 | 本机路径 |
| --- | --- |
| 权重 | `/home/descfly/桌面/sharpa_tomato_ckpt_4000` |
| 硬件仓库 | `/home/descfly/ZH/gello_upper_body_teleop` |
| Python | 相邻 `franka3-sharpa-pi05/.venv/bin/python` |
| ROS 消息 | 当前仓库 `.deployment/sharpa_ros_messages/install` |
| 执行日志 | 当前用户 `~/.local/state/fr3_sharpa_runtime/execution_traces` |

换机器时用 `setup_local.py --checkpoint ... --gello ... --python ...` 显式设置。生成的 `.deployment/local.env` 只保存本机路径，已被 Git 忽略；启动脚本自动读取，调用者显式设置的环境变量优先。统计文件默认从所选 checkpoint/assets 中唯一的 norm_stats.json 读取；只有确有必要时才设置 `SHARPA_NORM_STATS`，更换 checkpoint 时同时检查该变量。

配置过程不安装、升级或同步共享 Python 环境。`serve.sh` 和 `runtime.sh` 均使用已选解释器，并将当前仓库代码置于 PYTHONPATH 首位，避免导入相邻项目的代码。

`setup_local.py` 还会从硬件仓库现有的 `config/modes/teleop_control.yaml` 生成独立的 `.deployment/teleop_openpi.yaml`：只增加 `openpi` 允许来源，保留原有来源、限位、速度、超时及接触检查设置。Compose 将此文件只读挂载给网关，不修改硬件仓库文件。SDK direct 在连接灵巧手前检查网关及其 `allowed_sources`，来源未获准时终止。

## 从当前状态启动、可选 Home 与每轮执行步数

固定 Home 到位容差为 **0.02 rad**（全部 58 个关节），仍要求相邻反馈变化不超过 0.005 rad 并连续稳定 0.5 秒。反馈新鲜度、原生限位和速度检查继续生效。

sdk_direct 默认 `--initialization current --execute-steps 15`。启动后采集当前双臂、双手实测关节状态和三路图像，直接推理执行，不读取 Home 文件，不发送 Home 或模型首帧 prepare 请求。每段首帧仍由现有执行器从最新实测姿态低速衔接。

需要先回固定位置时，在执行命令后加 `--home`（等效 `--initialization home`）：先把双臂移动到 poses.yaml 的 `tasks.powder_weighing.home`，双手移动到 `.deployment/hand_home.json` 固定目标；到位并稳定后才采集新观测进行任务推理。`--home` 是执行前回位选项，回位后继续任务执行。双臂 Home 使用原文件完整精度。

手部 Home 已于 2026-10-23 14:23（北京时间）采集：左手 22 个实测关节角作为左手固定 Home；右手按同名关节使用相同的 rad 数值，不做符号翻转。`--hand-home` 默认指向 `.deployment/hand_home.json`，与双臂共用一次 58D prepare、到位和稳定检查。只有显式选择 Home 模式才读取这份固定配置；Ctrl-C 不自动回位，下次默认从当前状态推理。该手部配置独立于双臂快照，同步 poses.yaml 不会覆盖它。文件缺失或关节名称/单位错误时，固定 Home 启动失败，不回退为保持实测姿态。

源文件 `/home/descfly/franka_teleop_data/operator_gui/poses.yaml` 是 root 专属读权限，部署读取 `.deployment/operator_poses.yaml` 快照。初次配置或 GUI 更新 Home 后运行以下命令同步；`--via-docker` 只用无网络、只读挂载读取原文件，不启动设备，也不修改权限：

```bash
source deploy/fr3_sharpa_runtime/env.sh
"$SHARPA_PYTHON" -m deploy.fr3_sharpa_runtime.home_pose --via-docker
```

快照配套 source.json 记录原始路径和 SHA256，初始化日志包含该来源信息。不同位置可通过 `--home-poses` 和 `--home-task` 指定。显式 `--initialization policy` 才使用原来的模型首帧 prepare；`--prepare-policy-first` 仍是该旧模式的别名，不能与 `--home` 或显式 `--initialization current` 混用。

每次模型仍预测完整 50×58；执行端只执行索引 0–14，然后丢弃剩余 35 帧、重新观测和推理。模型 horizon 不变。可用 `--execute-steps 1..50` 修改；`--execute-steps 50` 才执行全部预测。日志保留完整预测及计划/实际执行步数，末帧误差对照最后实际执行的动作。

`--num-chunks` 是重新推理/执行的轮数，不是每轮动作数：默认 1；任意正整数执行对应轮数；0 持续执行，没有 chunk1/chunk2 上限。循环中任一观测、推理或执行失败会停止，不会强行继续。

## 预测动作小幅越界裁剪

sdk_direct 默认 `--prediction-clip-rad 0.05`：仅对本轮实际执行前缀内的左右手预测值进行裁剪，每个关节、每帧最多修正 0.05 rad（约 2.86°），实际下发值仍在现有驱动关节范围内。超过此修正量立即停止；设为 0 恢复严格越界拒绝。未执行的预测帧、机械臂目标、固定 Home 和实测反馈不做裁剪。

裁剪后再进行首帧衔接和速度插值，完整下发序列（含当前实测姿态到首帧的过渡）继续检查相邻手部命令变化严格小于 0.10 rad/帧，不因裁剪放宽这一限制。下层驱动的关节限位检查继续生效。

终端出现 `[prediction-clip]` 时会显示修正数量及最大幅度。原始 50 帧预测仍保留在 chunk JSON 中，成功执行报告的 `prediction_clipping` 包含每项修正的动作索引、左右侧、关节名、原值、修正值和幅度；后续执行失败时，已计算的修正审计附在错误详情中。此参数与 Home 到位容差是两个独立设置。

## 不连接设备的验证

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh check
bash deploy/fr3_sharpa_runtime/serve.sh --check-only
bash deploy/fr3_sharpa_runtime/runtime.sh --check --speed slow
bash deploy/fr3_sharpa_runtime/build_messages.sh
bash deploy/fr3_sharpa_runtime/serve.sh --smoke-only \
  --report .deployment/local_model_smoke.json
```

`hardware.sh check` 只解析 Compose；`build_messages.sh` 只编译消息并验证 Python 导入；smoke 使用黑色 RGB 和零状态验证模型计算链路。这些检查不建立设备连接，不代表实时观测、轨迹可执行或任务成功。模型服务正常启动时先预热，再向客户端发送 checkpoint/统计指纹、相机和动作协议。

## 实时观测时间对齐

参考录制入口的就绪检查，持久观测 worker 首次启动要求三路相机连续稳定 3 秒（cam0 至少 18 Hz，cam1/cam2 至少 25 Hz，源时间戳递增，源/接收间隔不超过 150 ms）。首次快照最多等待 15 秒，成功后恢复每次 5 秒。

各路最新消息时间跨度超过 100 ms 时，从有界短时缓存中按共同源时间选择真实历史消息：以最慢一路最新源时间为基准，其余话题取不晚于该时间的最近样本。没有匹配样本时继续等待，不复制、插值或改写时间戳。选帧时原来的 200 ms 源时间/接收时间新鲜度和 100 ms 跨模态跨度限制保持不变；任一最新流过期或未来时间戳仍拒绝。

时间对齐会使用稍早的关节和腕部图像，不能消除头部相机自身的传输延迟。每轮动作后的观测仍要求所有被选消息来自执行结束之后。执行 trace 的 `_alignment` 记录选帧模式、各路年龄和源时间跨度；此缓存仅用于模型观测，执行器实时反馈检查不变。

## 真实数据离线验证

`--episode-parquet` 支持两类输入：已排好顺序的 58D 状态及内嵌图像，或者 LeRobot v2.1 的具名状态及外部 MP4。对服务器原始 state116，根据 meta/info.json 名称提取位置并重排到 `[左臂7, 左手22, 右臂7, 右手22]`；不会截取前 58 维。相机为 cam0 头部、cam2 左腕、cam1 右腕，解码后为 RGB；检查数据及视频为 30 Hz、帧时间戳一致。缺少具名元数据的 116D 输入直接拒绝。

先在终端 1 启动 `serve.sh`，再在终端 2 运行：

```bash
bash deploy/fr3_sharpa_runtime/runtime.sh \
  --episode-parquet /home/descfly/company-remote-2/real/harvest-tomato/data/data-lerobot/sharpa/A/121_430_data_lerobot/data/chunk-000/episode_000000.parquet \
  --sample-frame 100
```

默认是 dry-run：只请求推理和报告，不启动 SDK、不发布运动命令。默认 prompt 与当前服务器 A/B 数据的 tasks.jsonl 一致，其他任务请用 `--prompt` 指定。实时 ROS 输入本来就是 canonical 58D，不重复套用原始数据重排。

## 现场运行

按三个终端顺序复制的完整本机命令（含 Home 同步、预热成功信息、持续执行、停止和日志查询）见 [根 README 的完整启动说明](../../README.md#完整启动持续执行与停止)。下文以全量模型入口 `fr3_sharpa_full` 为例说明各服务职责和执行边界；LoRA 另用对应入口。

以下命令会启动现场硬件依赖或发送运动指令，应在完成现场观测、设备身份和环境检查后使用。本轮本机配置验证没有执行这些命令。

终端一，启动依赖：

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh up
```

此入口仅启动 `franka-control`、采用独立 OpenPI 网关配置的 `teleop-control` 和只运行三路相机的 `openpi-cameras`；无需另一个 `compose.openpi.yaml`，不启动 bag recorder 或旧 sharpa_driver。会复用既有 Compose 项目中的同名服务；已有遥操任务运行时应先结束该任务。不要同时启动旧 sharpa_driver 和 sdk_direct，两者不能共同拥有灵巧手设备。

终端二，模型服务：

```bash
bash deploy/fr3_sharpa_full/serve.sh
```

### 终端三：选择 current 或 home（二选一）

**以下两条命令都会连接设备并发送运动指令，选择其中一条执行，不要同时运行。**两种模式共用终端一和终端二，区别仅是执行前是否先回固定 Home。首次现场验证，将所选命令的 `--num-chunks 0` 改为 `--num-chunks 1`；确认单轮后再持续运行。

**A．current：从当前位置开始，不先回 Home。**适合从当前现场开始或中断后重新推理。采集当前双臂、双手状态及三路图像，再低速衔接到预测首帧；不会读取固定 Home 目标。

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test

bash deploy/fr3_sharpa_full/runtime.sh \
  --execute \
  --enable-hardware \
  --sharpa-backend sdk_direct \
  --initialization current \
  --prepare-speed-rad-s 0.03 \
  --execute-steps 15 \
  --speed slow \
  --num-chunks 0
```

**B．home：先回固定双臂、双手 Home，再开始任务。**双臂使用 `tasks.powder_weighing.home`，双手使用 `.deployment/hand_home.json`；到位并稳定后重新采集观测、推理和执行。缺少 Home 配置时会报错。`--home` 是 `--initialization home` 的别名，以下统一使用完整参数。

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test

bash deploy/fr3_sharpa_full/runtime.sh \
  --execute \
  --enable-hardware \
  --sharpa-backend sdk_direct \
  --initialization home \
  --prepare-speed-rad-s 0.03 \
  --execute-steps 15 \
  --speed slow \
  --num-chunks 0
```

两种模式都每轮预测 50×58，只执行前 15 个动作，然后重新观测、重新推理。`--num-chunks 0` 表示持续循环，不限于 chunk1/chunk2；正整数表示指定轮数，没有自动识别任务完成的逻辑。

中途 Ctrl-C 后，等待失能确认与硬件清理结束；再次执行前先在终端一运行 `bash deploy/fr3_sharpa_runtime/hardware.sh up`，确认终端二的模型服务仍在运行，再重新选择上面的 current 或 home 命令。失能期间是否保持抓取取决于设备实际状态；首帧衔接与 Home prepare 都采用关节空间直线插值，不包含环境碰撞规划。

| 档位 | 时间倍率 | 15 帧主体下发点 | 30 Hz 主体时长 |
| --- | ---: | ---: | ---: |
| slow | 0.1 | 141 | 约 4.70 秒 |
| medium | 0.5 | 29 | 约 0.97 秒 |
| fast | 1.0 | 15 | 约 0.50 秒 |

时长不含 prepare、首帧衔接、观测和推理。未指定速度时默认 1.0；`--speed` 不能与 `--execute-speed-scale` 混用。`--num-chunks 0` 表示持续运行到 Ctrl-C，没有任务完成识别。当前是串行推理/执行，尚未实现 RTG、Ruckig 或跨段非零速度拼接，详见 [架构说明](ARCHITECTURE.md)。

执行终端退出时（包括 Ctrl-C、SIGTERM、正常完成指定轮数或运行失败），`runtime.sh` 的会话管理器先结束 runtime/SDK 子进程，再清理该次执行启动前记录的部署容器。它会停止并删除 OpenPI 的 `openpi-cameras` 与使用 `/etc/openpi/teleop_control.yaml` 的 `teleop-control`；释放网关后停止对应的机械臂控制容器。删除网关容器可避免后续遥操误用它残留的 OpenPI 配置，遥操启动器会重新按自身配置创建网关。

SDK direct 在最终退出时先对已有双手连接请求失能，并通过 SDK `get_enable_state` 读回确认；只接收布尔 False 作为确认，不将“stop 调用未报错”当作成功。左右手并行处理，默认总确认窗口 **120 秒**、每只手最多 **120 次**、重试间隔 **1 秒**。确认成功后断开 SDK，再继续既有的进程和容器清理；某只手始终使能、读回失败或调用超时，则在终端输出该手的 ERROR、尝试次数、最后使能状态和错误，再退出。没有已有连接时不会为检查状态重新连接或使能设备，也不会自动张手、回 Home。

可在第三个执行终端启动前覆盖以下环境变量（模型终端无需设置）：

```bash
export SHARPA_DISABLE_TIMEOUT_S=120
export SHARPA_DISABLE_MAX_ATTEMPTS=120
export SHARPA_DISABLE_RETRY_INTERVAL_S=1
```

以上 120 秒是失能/读回窗口，不包括后续原有容器停止耗时。父进程为确认留有等待余量，并等待 SDK 的专用退出确认消息；即使 Pixi 包装进程先退出也不当作设备已失能。确认期间再次 Ctrl-C 会提示等待，不跳过确认；若 SDK/进程挂起导致最终仍未收到确认，上层会明确输出“未确认设备失能”后按有限超时退出。SIGKILL、断电或设备通信故障无法保证完成失能，不能把程序退出视为设备失能的证据。

终端成功信息形如 `[sharpa-shutdown] left: 已读回使能=False...`，左右手各一条，最终输出 PASS。未成功则输出 `[sharpa-shutdown] ERROR left/right: ...未确认失能...` 和最终 FAIL；如果失能已确认但断连失败，会单独说明断连错误。原来的 watchdog stop、推理服务退出和容器清理顺序/范围不变。

清理按不可变容器 ID、Compose 配置来源和网关挂载路径核对，不按服务名强制关闭后来重建的容器；检测到遥操已经接管时跳过对应控制器。不执行整个项目的 `down`，也不停止其他遥操服务。模型服务是独立终端，需要另行 Ctrl-C 结束。`--check`、帮助、离线 dry-run 不触发硬件清理。

如果只启动了硬件、尚未运行执行端，或强制终止导致自动清理未完成，可显式重试：

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh stop
```

`stop` 使用同样的归属检查，清除匹配的 OpenPI 网关和相机容器；没有遗留部署容器时可重复运行。SIGKILL、断电或 Docker 不可用时不能保证自动清理成功；清理失败会明确报告错误，恢复 Docker 后可重试此命令。

## 日志与验收边界

每轮执行保存 chunk JSON、整次 events.jsonl 和 run.json。`COMPLETE` 表示指令下发完成，不代表全部关节精确到位或番茄任务成功。prepare、观测、推理、执行失败与中断会记录对应阶段。

checkpoint 下载目录未携带完整训练记录；2026-10-25 已在 28 服务器另行找到 AB4 UI 任务、归档配置源码、训练 JSON 与日志，并核对源码及统计指纹一致。实际运行设置为 batch 512、FSDP 8、absolute、horizon 50，数据根目录为四个 A/B normal 批次，详见 [实际 AB4 实验记录](../../docs/SHARPA_HANDOFF.md#已找到的-ab4-实验记录)。原始 checkpoint 旧路径目前不存在，尚未逐项比对模型参数文件，不能仅凭统计一致宣称权重来源已完全闭环。GPU smoke 和真实样本 dry-run 不替代现场相机、左右设备身份及轨迹检查。

单元测试：

```bash
source deploy/fr3_sharpa_runtime/env.sh
"$SHARPA_PYTHON" -m pytest -q deploy/fr3_sharpa_runtime/tests
```

旧 guarded 路径说明见 [LEGACY.md](LEGACY.md)。历史 `--preflight-only` 仍检查 guarded driver，不能用它代替 sdk_direct 的现场验收。

## 可选论文推理实验

`--inference-preset paper-rolling` 提供默认关闭的实验路径，50点模型中消费26点、保留4点，独立选择去噪内部 Inpainting 和三次插值压缩。参考基于轨迹进度，非异步实时 chunking。旧命令仍为50/15；说明及工控机命令见 [推理实验文档](../../docs/pi05_paper_adaptation/README.md)。
