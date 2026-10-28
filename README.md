# FR3 + Sharpa π0.5 番茄任务部署

`chunk-base-test` 已包含可选 RTC 试验：服务端 `--enable-rtc`，执行端用 `--broker-mode serial/rtc` 选择机制。**RTC 启动统一显式使用 `--broker-mode rtc --speed fast`，表示 1.0 倍模型原速；非 RTC 串行保留 slow／medium／fast 三档。**旧命令默认串行。配置、测试状态及现场入口见 [RTC 说明](deploy/fr3_sharpa_full/RTC.md)。串行与 RTC 共存在同一分支，无需切换分支。

双 FR3 机械臂、双 Sharpa 灵巧手的视觉语言动作模型工程。当前部署使用 **π0.5 全量微调、30 Hz、原生 58 维绝对关节目标**。

## 全量与 LoRA 统一入口

日常使用可直接进入 [全量部署目录](deploy/fr3_sharpa_full/README.md) 或 [LoRA 部署目录](deploy/fr3_sharpa_lora/README.md)；底层共用 runtime。

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

全量 delta 权重已接入独立配置，工作机配置与命令见 [全量 Delta 部署](deploy/fr3_sharpa_full/DELTA.md)；该配置已通过真实 GPU 假观测推理，现场运行待验证。

新的 LoRA checkpoint 尚未提供，目前只完成代码和配置测试，真实加载与工作机实机验证待完成。

权重接入步骤见 [LoRA 部署说明](deploy/fr3_sharpa_lora/README.md)。

## Delta 权重：串行与 RTC 完整启动

以下命令全部在**工作机的部署仓库根目录**执行，读取工作机桌面上的 Delta 权重；不是开发机上的路径。Delta 的串行和 RTC 使用同一个 checkpoint、同一份统计、同一套硬件服务，只切换执行方式。不要将下方 Delta 执行端连到后面的 absolute 模型服务。

| 方案 | 模型服务 | 执行入口／选择 | 默认端口 |
| --- | --- | --- | ---: |
| 原 absolute 串行 | `serve.sh` | `runtime.sh --broker-mode serial` | 18765 |
| Delta 串行 | `serve_delta.sh`（可加 `--enable-rtc`） | `runtime_delta.sh --broker-mode serial` | 18767 |
| Delta RTC | `serve_delta.sh --enable-rtc` | `runtime_delta.sh --broker-mode rtc --speed fast` | 18767 |

### 第一次配置或更换 Delta 权重

已配置好且没有更换权重时跳过。将下面示例路径替换成工作机上的实际步骤目录，**这一层下直接包含完整 `params/` 和配套 `assets/`**；部署不需要 `train_state/`。

```bash
source deploy/fr3_sharpa_runtime/env.sh
"$SHARPA_PYTHON" -m deploy.fr3_sharpa_full.setup_delta \
  --checkpoint "$HOME/桌面/实际Delta步骤目录"
```

生成 `.deployment/full_delta.env`。已有配置且明确要更换权重时，在命令末尾加 `--force`。若手动设置过 `SHARPA_FULL_DELTA_NORM_STATS`，更换权重时也要更新或取消该覆盖，避免读取旧统计。不要修改原 absolute 的 `SHARPA_CHECKPOINT` 来切换 Delta。

### 终端一：检查并启动硬件

检查配置的命令不会运动；最后一条启动硬件服务。

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh check
bash deploy/fr3_sharpa_full/serve_delta.sh --check-only
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --check --initialization current --broker-mode serial \
  --speed slow --execute-steps 15 --num-chunks 2
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --check --initialization current --broker-mode rtc \
  --speed fast --execute-steps 15 --num-chunks 2
bash deploy/fr3_sharpa_runtime/hardware.sh up
```

### 终端二：启动同一份 Delta 模型服务

需要对比串行和 RTC 时，统一启动下面这个服务，两种执行方式都能使用：

```bash
bash deploy/fr3_sharpa_full/serve_delta.sh --enable-rtc
```

等待普通 `Warmup OK`、`RTC guided Warmup OK`，以及服务监听 `127.0.0.1:18767`。保持终端运行；切换执行端的 serial／RTC 不需要重新启动模型服务。

如果只使用原来的 Delta 串行流程，仍可运行 `bash deploy/fr3_sharpa_full/serve_delta.sh`，不带 `--enable-rtc`；但这个普通服务不能接受 RTC 执行请求。不要同时启动两个占用 18767 的模型服务。

### 终端三 A：Delta 串行，选择速度

**以下执行命令都会驱动机器人，只运行其中一条。**从相近初始场景比较 slow／medium／fast；每次有限轮数执行结束或 Ctrl-C 后，等待失能确认和清理，再重新 `hardware.sh up` 开始下一次。

从当前状态开始，slow：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --execute --enable-hardware --initialization current \
  --broker-mode serial --speed slow \
  --prepare-speed-rad-s 0.03 --execute-steps 15 --num-chunks 2
```

从当前状态开始，medium：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --execute --enable-hardware --initialization current \
  --broker-mode serial --speed medium \
  --prepare-speed-rad-s 0.03 --execute-steps 15 --num-chunks 2
```

从当前状态开始，fast（1.0 倍原速）：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --execute --enable-hardware --initialization current \
  --broker-mode serial --speed fast \
  --prepare-speed-rad-s 0.03 --execute-steps 15 --num-chunks 2
```

需要先回 Home 再开始串行执行，例如 slow：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --execute --enable-hardware --home \
  --broker-mode serial --speed slow \
  --prepare-speed-rad-s 0.03 --execute-steps 15 --num-chunks 2
```

串行每轮预测 50 帧，执行索引 0–14，再重新观测推理；不带 `--broker-mode` 的旧命令等效于 `serial`。slow／medium／fast 分别为 0.1／0.5／1.0 倍速，设备下发均为 30 Hz。

### 终端三 B：Delta RTC，明确使用 fast 原速

RTC 真机验证仍待完成。RTC 的现场命令统一写 `--speed fast`，即 1.0 倍模型原始时间轴；首次接入仍保留低速衔接。它不是“RTC slow”，也不是提高到超过模型原速。

从当前状态开始，执行两个预测块：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --execute --enable-hardware --initialization current \
  --broker-mode rtc --speed fast \
  --prepare-speed-rad-s 0.03 --execute-steps 15 --num-chunks 2
```

先回固定双臂、双手 Home，再以 RTC 原速执行：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --execute --enable-hardware --home \
  --broker-mode rtc --speed fast \
  --prepare-speed-rad-s 0.03 --execute-steps 15 --num-chunks 2
```

RTC 中 `--execute-steps 15` 是消费 15 个模型节点后请求下一次异步推理，等待结果时旧块继续播放，不等同于串行严格只执行前 15 帧。`--num-chunks 2` 包含首块普通预测和后续 guided 预测；RTC 有限轮数必须至少为 2。

### 连续执行、中断恢复与停止

确认有限轮数正常后，将所选命令的 `--num-chunks 2` 改为 `--num-chunks 0` 持续运行。RTC 保持 `--speed fast`；串行按需要选择三档。`--initialization current` 和 `--home` 二选一，不能同时传。

中途在执行终端 Ctrl-C 后，先等双手失能读回确认及硬件会话清理完成。若模型终端仍在运行，保持不动；若已停止，先重新启动终端二。然后重新启动硬件：

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh up
```

再运行上面对应的完整执行命令。希望从中断后的当前状态继续时选择 current；需要先回位时选择 Home。重启后重新观测推理，不复用上次剩余动作。失能期间是否保持抓取取决于设备实际状态。

完全结束时，先结束执行端并等待清理，再 Ctrl-C 停止模型服务；若清理失败或仅启动硬件未运行执行端，可重试：

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh stop
```

Delta 串行日志默认在 `~/.local/state/fr3_sharpa_full_delta/execution_traces/`；RTC 日志在其 `rtc-<id>/` 子目录。详细配置、记录和验证边界见 [Delta 说明](deploy/fr3_sharpa_full/DELTA.md)及 [RTC 说明](deploy/fr3_sharpa_full/RTC.md)。

## 原 absolute 权重配置

| 项目 | 配置 |
| --- | --- |
| 模型 | `gemma_2b` + `gemma_300m`，无 LoRA |
| 输入 | 三路 RGB、58 维实测关节位置、英文任务文本 |
| 输出 | 每次预测 50 帧 × 58 维，默认执行前 15 帧后重新观测推理 |
| 初始姿态 | 默认 current；显式 Home 才使用 powder_weighing 双臂 Home 和固定双手 Home |
| 关节顺序 | 左臂 7、左手 22、右臂 7、右手 22 |
| 相机 | 头部 cam0、左腕 cam2、右腕 cam1 |
| 模型动作与设备下发 | 30 Hz，名义间隔约 33.33 ms |
| 执行后端 | `sdk_direct`：FR3 Safety Gateway + Sharpa Wave SDK |

本机配置绑定以下权重，代码仓库不包含权重：

```text
/home/descfly/桌面/sharpa_tomato_ckpt_4000
```

归一化统计自动读取该 checkpoint 的 `assets/tomato_sharpa_ab4_normal_rgb_absolute_v1_30hz/norm_stats.json`。更换路径时，在两个终端设置相同的 `SHARPA_CHECKPOINT`；不要替换为其他实验的统计。

<a id="完整启动持续执行与停止"></a>

## 原 absolute 权重：串行完整启动、持续执行与停止

**第一次拉取仓库请先阅读 [新机器部署与研究交接](docs/SHARPA_HANDOFF.md)。**下列命令面向当前已配置的机器，不是从零安装步骤。换机器请替换每段 `cd` 的仓库路径，并先生成自己的 `.deployment/local.env`。权重、SDK、现场配置、双臂 Home 和固定双手 Home 不会随 git clone 下载。

本节为原 absolute 权重的串行启动流程，使用 `serve.sh`／`runtime.sh` 和端口 18765。Delta 的串行与 RTC 请使用上面的完整命令。

本机启动脚本会自动读取 `.deployment/local.env` 并使用选定的 Python，不需要手动激活虚拟环境。请使用新的终端，或确认未遗留覆盖本机配置的 `SHARPA_*` 环境变量。

### 启动前：需要时同步双臂 Home

仅在选择 home 模式且 GUI 修改过双臂 Home 后，需要先同步到部署快照；current 模式不需要这一步。同步不会覆盖 `.deployment/hand_home.json` 的固定双手配置。

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
source deploy/fr3_sharpa_runtime/env.sh
"$SHARPA_PYTHON" -m deploy.fr3_sharpa_runtime.home_pose \
  --source /home/descfly/franka_teleop_data/operator_gui/poses.yaml \
  --task powder_weighing \
  --via-docker
```

这里的 Docker 仅用于读取 root 专属权限的 poses.yaml，采用无网络、只读挂载，不启动设备。双手配置需要单独交接或标定；当前固定 Home 模式缺少手部配置会报错，不会自动改为保持启动时姿态。

### 终端一：检查配置并启动硬件依赖

先结束已有遥操任务、相机采集和旧 `sharpa_driver`，避免同时占用设备。以下检查命令不连接设备；`hardware.sh check` 成功时可能没有输出。

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test

bash deploy/fr3_sharpa_runtime/hardware.sh check
bash deploy/fr3_sharpa_full/serve.sh --check-only
# 分别显示 current 与 home 配置；这两条都不会执行运动
bash deploy/fr3_sharpa_full/runtime.sh \
  --check --initialization current --execute-steps 15 --speed slow --num-chunks 0
bash deploy/fr3_sharpa_full/runtime.sh \
  --check --initialization home --execute-steps 15 --speed slow --num-chunks 0

# 以下命令启动现场硬件服务
bash deploy/fr3_sharpa_runtime/hardware.sh up

docker ps --format 'table {{.Names}}\t{{.Status}}'
```

`up` 启动机械臂控制、使用 OpenPI 配置的 Safety Gateway 和三路相机，不启动旧 sharpa_driver 或 bag recorder。容器显示 Running 只说明进程状态，不能代替设备及实时观测检查。

### 终端二：启动模型服务

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
bash deploy/fr3_sharpa_full/serve.sh
```

等待出现以下预热成功信息，以及服务监听 `127.0.0.1:18765` 的信息，再运行执行端。保持终端二运行。

```text
Warmup OK: actions=(50, 58), full, absolute, 30 Hz
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

共有参数说明：

- `--speed slow` 对应时间倍率 0.1，15 个动作的主体为 141 个 30 Hz 下发点，约 4.70 秒，不含首帧衔接、观测及推理。medium=0.5、fast=1.0；不指定速度时默认 fast。
- `--prepare-speed-rad-s 0.03` 同时用于首帧衔接和 home 模式的回位；采用关节空间插值，不包含环境碰撞规划。current 模式也保留首帧低速衔接。
- 手部预测的小幅越界裁剪默认 `--prediction-clip-rad 0.05`，范围与审计含义见 [runtime 说明](deploy/fr3_sharpa_runtime/README.md#预测动作小幅越界裁剪)。

### 中途停止后重新启动

在终端三按 Ctrl-C 后，等待双手失能读回确认和硬件会话清理结束。终端二的推理服务若仍在运行，可保持不动；若已停止，先重新启动终端二。

在终端一重新启动硬件依赖：

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
bash deploy/fr3_sharpa_runtime/hardware.sh up
```

然后在终端三重新选择上面的 **A（current）** 或 **B（home）** 完整命令。从中途现场继续时选择 current；需要先回固定初始位置时选择 home。新一轮读取当前实测状态和图像，不恢复上次剩余的旧动作。失能期间是否保持抓取取决于设备实际状态。

### 停止顺序

1. 先在终端三按 Ctrl-C。当前会话管理器会结束 runtime/SDK 子进程，并按容器身份清理本次部署所属的相机、OpenPI 网关和对应机械臂控制器；不自动回 Home。
2. 再在终端二按 Ctrl-C，停止独立的模型服务。
3. 若只启动了硬件而未启动执行端，或退出后仍有部署容器遗留，在终端一显式重试清理：

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
bash deploy/fr3_sharpa_runtime/hardware.sh stop
```

`stop` 可重复运行，使用归属核验，不对整个硬件项目执行 down。SIGKILL、断电或 Docker 不可用时不能保证自动清理完成；遇到清理错误应确认实际容器状态。详细归属检查和遥操接管行为见 [runtime 说明](deploy/fr3_sharpa_runtime/README.md#现场运行)。

### 运行记录

本机默认目录：`/home/descfly/.local/state/fr3_sharpa_runtime/execution_traces/`，可用 `--execution-trace-dir` 或 `SHARPA_EXECUTION_TRACE_DIR` 覆盖。包含 chunk JSON、events.jsonl 和 run.json。

```bash
ls -lt /home/descfly/.local/state/fr3_sharpa_runtime/execution_traces/
```

`COMPLETE` 表示指令下发完成，不表示任务成功。模型加载、单元测试和离线 dry-run 的通过记录不能代替本次现场运动验收。启动及运行限制详见 [runtime 部署说明](deploy/fr3_sharpa_runtime/README.md)，模型与执行层结构见 [架构说明](deploy/fr3_sharpa_runtime/ARCHITECTURE.md)。

## 检查与验证

以下检查不连接机器人：

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
source deploy/fr3_sharpa_runtime/env.sh
"$SHARPA_PYTHON" -m pytest deploy/fr3_sharpa_runtime/tests -q
bash deploy/fr3_sharpa_runtime/hardware.sh check
bash deploy/fr3_sharpa_full/serve.sh --check-only
bash deploy/fr3_sharpa_full/serve.sh --smoke-only
```

当前 4000 步权重已通过真实 GPU 加载及假观测推理，输出有限的 `(50, 58)`。这证明模型加载与动作接口可用，不代表现场任务成功。执行记录默认位于 `~/.local/state/fr3_sharpa_runtime/execution_traces/`。

原始转换器数据为 state116/action58；训练时选择实测位置并重排为 canonical 58D。实时部署输入已经是 canonical 58D，不能再套原始数据重排或重复添加当前关节位置。数据 normal/speedup 的区别、训练代码对应关系及尚未确认的实验信息见 [runtime 说明](deploy/fr3_sharpa_runtime/README.md)。

## 代码导航

| 目录 | 用途 |
| --- | --- |
| `deploy/fr3_sharpa_full/` | 全量模型启动入口与使用说明 |
| `deploy/fr3_sharpa_lora/` | LoRA 模型启动入口、权重配置与说明 |
| `deploy/fr3_sharpa_runtime/` | 两种模型共用的加载、观测、prepare、SDK 执行和运行记录 |
| `src/openpi/policies/fr3_sharpa_policy.py` | 58D 输入输出变换 |
| `src/openpi/policies/fr3_sharpa_protocol.py` | Sharpa 关节名称和协议 |
| `examples/fr3_sharpa/` | Sharpa 离线数据转换、导出与比较工具 |
| `src/openpi/training/` | 通用训练框架及保留的 Sharpa 历史训练配置 |
| `src/openpi/models/` | OpenPI 模型实现 |
| `scripts/` | 通用训练、统计和模型服务入口 |
| `packages/openpi-client/` | WebSocket 客户端 |
| `docs/` | 通用 OpenPI 参考资料与文档索引 |

## 训练来源、当前加载配置与历史版本

本仓库负责部署。当前权重由 `deploy/fr3_sharpa_runtime/model_loader.py::build_config()` 加载：`gemma_2b` + `gemma_300m`、无 LoRA、`action_dim=58`、`action_horizon=50`、`max_token_len=256`、30 Hz、`extra_delta_transform=False`。推理配置名为 `pi05_fr3_sharpa_ab4_normal_rgb_fullft_absolute_30hz_deploy`，它不是恢复出来的原始训练实验名。

训练参考工程是 [fruit-train/tomato-openpi](https://gitlab.litchibot.top/fruit-train/tomato-openpi)，本次文档核对版本为 `68440ed8b39693b5fd206403843901187f453c1c`。本机路径 `/home/descfly/LTT/tomato-openpi` 仅是该工程的一个 clone，不是其他人必须创建的目录。具体来源如下：

| 参考工程中的文件 | 用途 |
| --- | --- |
| `training_configs/experiments/tomato_sharpa/baseline/tomato_fr3_sharpa_fullft_v1.py` | Sharpa 全参 baseline，定义任务文本、数据身份和实验配置名 |
| `training_configs/recipes/pi05_fullft_v1.py` | 选择无 LoRA 的 Gemma 变体，全参数训练，baseline 全局 batch=64 |
| `training_configs/recipes/common.py` | 模型窗口、优化器、学习率、随机种子和 checkpoint 保存规则 |
| `training_configs/embodiments/fr3_sharpa.py` | 58D、rad、absolute、30 Hz 和三路相机角色 |

**2026-10-25 已在 28 服务器找到实际 AB4 UI 任务和当次训练归档。**任务 ID 为 `2200a091b1ee48239ad11b203a4bf934`，训练源码记录为 `42f887916d22cd58c022e95a000e2cb241a8a459`；实际记录是四个 A/B normal 批次、训练全局 batch **512**、FSDP **8**、absolute、horizon **50**、计划 30,000 步、每 2,000 步保存、`epoch_shuffle`。不能将上述参考 baseline 的 batch 64，或早期对话建议的 batch 32，写成该实验的实际设置。

实际 asset_id 为 `tomato_sharpa_ab4_normal_rgb_absolute_v1_30hz`。服务器统计文件与本机 4000 步 checkpoint 自带文件的 SHA256 完全一致；归档 `.experiment.py` 的摘要也与 UI 记录一致。源码快照、UI 覆盖参数、四个数据路径及两份摘要见 [已找到的 AB4 实验记录](docs/SHARPA_HANDOFF.md#已找到的-ab4-实验记录)。原始 checkpoint 旧路径目前不存在，仍未完成模型参数文件逐项比对；训练主进程环境变量和数据文件完整性也不能仅由这两份摘要推定。

本仓库 `src/openpi/training/config.py` 中的 `pi05_fr3_sharpa`、`pi05_fr3_sharpa_a04_b05_c01_30hz` 是保留的历史 LoRA 配置，包含 LoRA 变体和双臂 delta 变换。当前 `serve.sh` 使用上述独立 model_loader，不选择这些历史配置；不要只把 checkpoint 路径替换到旧配置里加载。

旧 Wuji 部署和 `experiments/weight_motion_eval` 在 `c60b4c2612020d641ed3003a82cd8ab5d143f4fc` 清理。清理前版本是 `bbe7f697e17a220ca91fae10b1812322b0741fa2`，可在不切换当前分支的情况下查阅：

```bash
git show bbe7f697e17a220ca91fae10b1812322b0741fa2:deploy/fr3_wuji_fast/README.md
git ls-tree -r --name-only bbe7f697e17a220ca91fae10b1812322b0741fa2 -- deploy/ experiments/
```

旧实现包括 Wuji 设备协议、54D/其他模型映射和 RTG/Ruckig 方案，不是当前 Sharpa 的可运行入口。当前功能与历史设计的差异见 [架构说明](deploy/fr3_sharpa_runtime/ARCHITECTURE.md)和 [Wuji 历史参考](docs/reference_wuji_architecture.md)。

权重、训练数据、运行日志和本机构建产物不提交 Git。许可证见 [LICENSE](LICENSE) 和 [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)。

## 可选论文推理实验

新增默认关闭的滚动参考、去噪内部 Soft Inpainting、可选相关性传播，以及 26→20 三次插值压缩。支持现有 full/full-delta，保留默认 50/15 和全部原硬件执行路径。本轮只做 CPU 离线测试；接入方式、独立开关、时间轴限制和验证边界见 [推理实验说明](docs/pi05_paper_adaptation/README.md)。

`soft-inpainting-test` 在原项目目录中使用；新增 `deploy/fr3_sharpa_full/serve_soft.sh` 和 `runtime_soft.sh`，默认串行 Soft、无压缩。旧入口及 RTC 入口继续可用，`main` 不含本实验改动。切换方式时先停止当前执行端和模型服务，再按对应命令启动；不要共用同一端口启动两套服务。
