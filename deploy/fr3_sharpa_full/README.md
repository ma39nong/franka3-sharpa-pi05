# Sharpa 全量微调部署

`chunk-base-test` 已包含可选 RTC 试验：服务端 `--enable-rtc`，执行端用 `--broker-mode serial/rtc` 选择机制。**RTC 启动统一显式使用 `--broker-mode rtc --speed fast`，表示 1.0 倍模型原速；非 RTC 串行保留 slow／medium／fast 三档。**旧命令默认串行。配置、测试状态及现场入口见 [RTC 说明](RTC.md)。串行与 RTC 共存在同一分支，无需切换分支。

这是全量模型的日常入口，与 `../fr3_sharpa_lora/` 对称。两者共存于 `chunk-base-test`，相机、Home/prepare、三档速度、SDK 和执行循环共用 `../fr3_sharpa_runtime/` 的实现。

## Absolute 与 Delta

本页 `serve.sh`／`runtime.sh` 使用 absolute。新提供的 30 Hz 全量 delta 权重使用本目录的 `serve_delta.sh`／`runtime_delta.sh`，独立读取 `.deployment/full_delta.env`，端口默认 18767。双臂 delta 在模型服务中还原为 absolute，双手保持 absolute；执行流程共用。配置和完整命令见 [DELTA.md](DELTA.md)。

## 配置

沿用工作机已有 `.deployment/local.env`，不需要迁移或重新生成。权重使用 `SHARPA_CHECKPOINT`，统计使用 checkpoint 自带的唯一 `assets/**/norm_stats.json` 或显式 `SHARPA_NORM_STATS`；端口默认 18765，可用 `SHARPA_FULL_PORT` 设置。日志沿用 `SHARPA_EXECUTION_TRACE_DIR`，默认 `~/.local/state/fr3_sharpa_runtime/execution_traces/`。

当前模型：30 Hz、58D absolute、horizon 50，`gemma_2b` + `gemma_300m`。每轮默认执行前 15 帧，再重新观测推理。日常启动明确选择 `--initialization current` 或 `--initialization home`，具体命令见下方终端三。省略该参数时默认 current。

## 离线检查

从仓库根目录运行；`--check` 不需要机器人或模型权重：

```bash
bash deploy/fr3_sharpa_full/runtime.sh --check --initialization current --execute-steps 15 --speed slow --num-chunks 1
```

以下两条需要工作机真实 checkpoint；第二条需要 GPU，仅使用假观测：

```bash
bash deploy/fr3_sharpa_full/serve.sh --check-only
bash deploy/fr3_sharpa_full/serve.sh --smoke-only --report .deployment/full_smoke.json
```

## 三个终端启动（全量模型）

以下命令用于当前已配置的工作机，自动读取 `.deployment/local.env`，不需要手动激活 Python 环境。新机器先按 [交接说明](../../docs/SHARPA_HANDOFF.md)配置。Home 来源、快照同步及其他现场细节见 [根 README](../../README.md#完整启动持续执行与停止)。

### 终端一：检查配置并启动硬件依赖

先结束已有遥操任务、相机采集和旧 sharpa_driver，避免争用设备。先运行检查，确认无误后再运行硬件启动命令。

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test

bash deploy/fr3_sharpa_runtime/hardware.sh check
bash deploy/fr3_sharpa_full/serve.sh --check-only
bash deploy/fr3_sharpa_full/runtime.sh \
  --check --initialization current --execute-steps 15 --speed slow --num-chunks 0
bash deploy/fr3_sharpa_full/runtime.sh \
  --check --initialization home --execute-steps 15 --speed slow --num-chunks 0

# 以下命令启动现场硬件服务
bash deploy/fr3_sharpa_runtime/hardware.sh up
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

`hardware.sh check` 成功时可能没有输出；配置检查通过不代表实时设备已就绪。

### 终端二：启动全量模型推理服务

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
bash deploy/fr3_sharpa_full/serve.sh
```

等待 `Warmup OK: actions=(50, 58), full, absolute, 30 Hz` 和监听 `127.0.0.1:18765` 的信息，保持此终端运行，再启动执行端。

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

### 中途停止后重新启动

在终端三按 Ctrl-C 后，等待双手失能读回确认和硬件会话清理结束。终端二的推理服务若仍在运行，可保持不动；若已停止，先重新启动终端二。

在终端一重新启动硬件依赖：

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
bash deploy/fr3_sharpa_runtime/hardware.sh up
```

然后在终端三重新选择上面的 **A（current）** 或 **B（home）** 完整命令。从中途现场继续时选择 current；需要先回固定初始位置时选择 home。新一轮读取当前实测状态和图像，不恢复上次剩余的旧动作。失能期间是否保持抓取取决于设备实际状态。

### 停止

先在终端三按 Ctrl-C，等待双手失能读回确认（默认最多 120 秒、每手 120 次）和所属硬件会话清理。确认失败会明确显示 ERROR/FAIL。再在终端二按 Ctrl-C 关闭模型服务。

若只启动了硬件，或清理失败留下部署容器，在终端一运行：

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
bash deploy/fr3_sharpa_runtime/hardware.sh stop
```

该命令不是双手 SDK 失能命令，不能用容器已停止代替双手失能确认。参数及异常情况见 [公共 runtime 退出说明](../fr3_sharpa_runtime/README.md)。

## 目录与兼容性

- `serve.sh`：固定选择 full 的模型服务入口。
- `runtime.sh`：固定选择 full 的执行入口。
- `env.sh`：供手动 source 的全量环境入口。
- `serve_delta.sh`／`runtime_delta.sh`：全量 delta 模型入口。
- `setup_delta.py`／`delta.env.example`：独立 delta 权重配置。
- `../fr3_sharpa_runtime/`：共享实现与硬件工具；旧的启动命令继续有效。

本目录脚本转发到共享入口的 `--model full`，不复制模型加载或机器人执行代码，也不修改现有工作机配置。完整现场配置、观测及退出行为见 [公共 runtime 说明](../fr3_sharpa_runtime/README.md)。LoRA 权重接入见 [LoRA 说明](../fr3_sharpa_lora/README.md)。
