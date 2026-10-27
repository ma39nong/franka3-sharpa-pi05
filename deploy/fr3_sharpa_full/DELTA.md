# Sharpa 30 Hz 全量 Delta 部署

`chunk-base-test` 已包含可选 RTC 试验：服务端 `--enable-rtc`，执行端用 `--broker-mode serial/rtc` 选择机制。**RTC 启动统一显式使用 `--broker-mode rtc --speed fast`，表示 1.0 倍模型原速；非 RTC 串行保留 slow／medium／fast 三档。**旧命令默认串行。配置、测试状态及现场入口见 [RTC 说明](RTC.md)。串行与 RTC 共存在同一分支，无需切换分支。

本配置对应 28 服务器提供的 4000 步全量权重。入口放在 `fr3_sharpa_full/` 中，与 absolute 版本共享运行层。`full` 仍为默认 absolute；`full-delta` 是独立加载配置，默认端口 18767。

## 动作定义

模型顺序为 `[左臂7, 左手22, 右臂7, 右手22]`，30 Hz、58D、horizon 50、token 256、bfloat16，VLM `gemma_2b`、Expert `gemma_300m`，无 LoRA。

- 双臂维度 `[0:7, 29:36]`：训练目标为 `每帧绝对目标 - chunk 起始实测关节角`。
- 双手维度 `[7:29, 36:58]`：始终为绝对目标。
- 模型输出先反归一化，再仅对双臂加回本次观测的实测关节角；同一 chunk 的 50 帧共用该起始状态。不是逐帧累加，不是速度动作。
- 执行端收到的仍是 rad 单位的 58D absolute，不再叠加状态。实时输入已经 canonical，不重复应用训练时的原始 116D 状态提取或 action58 重排。

`--home` 控制启动姿态，与 delta 语义独立。默认从当前状态观测、推理，执行前 15 帧后重新观测。需要先回 Home 时显式添加 `--home`。

## 工作机接入

先将 checkpoint 的完整 `params/` 和 `assets/` 放到工作机同一个步骤目录。部署不需要 `train_state/`；不能只复制 manifest 或元数据。`sftp://28/...` 是传输地址，启动参数要使用工作机上的绝对文件路径。

沿用已有 `.deployment/local.env` 的 Python、SDK、相机和硬件设置，在仓库根目录生成独立模型配置：

```bash
source deploy/fr3_sharpa_runtime/env.sh
"$SHARPA_PYTHON" -m deploy.fr3_sharpa_full.setup_delta \
  --checkpoint /工作机上/这份delta权重/4000
```

它生成 `.deployment/full_delta.env`，使用 `SHARPA_FULL_DELTA_CHECKPOINT`、可选 `SHARPA_FULL_DELTA_NORM_STATS`、`SHARPA_FULL_DELTA_PORT`。文件已存在时需显式 `--force` 才覆盖；也可参考 [delta.env.example](delta.env.example)。`SHARPA_FULL_DELTA_ENV_FILE` 可指定另一份配置。权重和统计均不回退使用 absolute 或 LoRA 的配置。

检查和 GPU 假观测验证：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh --check --initialization current --speed slow
bash deploy/fr3_sharpa_full/serve_delta.sh --check-only
bash deploy/fr3_sharpa_full/serve_delta.sh --smoke-only --report .deployment/full_delta_smoke.json
```

先按现有方式 `bash deploy/fr3_sharpa_runtime/hardware.sh up` 启动工作机硬件服务，然后分别在模型终端和执行终端运行以下命令。

模型终端：

```bash
bash deploy/fr3_sharpa_full/serve_delta.sh
```

执行终端，从当前状态执行一轮：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --execute --enable-hardware --initialization current --speed slow --num-chunks 1
```

确认现场效果后，设 `--num-chunks 0` 连续重推。需要先回 Home 再执行时：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --execute --enable-hardware --speed slow --num-chunks 1 --home
```

两套脚本等效于公共入口的 `--model full-delta`。两端必须选择同一模型、checkpoint 和统计。原 `serve.sh`／`runtime.sh` 仍选择 full absolute。Delta 日志默认位于 `~/.local/state/fr3_sharpa_full_delta/execution_traces/`，可用 `SHARPA_FULL_DELTA_EXECUTION_TRACE_DIR` 覆盖。

Ctrl-C 后仍执行双手失能确认与硬件清理。等清理结束，重新 `hardware.sh up`，再使用上述 current 命令恢复；每次重启重新观测，不复用旧 chunk。失能期间是否保持抓取由设备实际状态决定。

## 权重来源与验证

2026-10-26 通过 SSH 别名 `28` 只读核对的 checkpoint：

```text
/home/tairan/workspace/code/openpi/checkpoints/
  pi05_fr3_sharpa_1016_2_multimodal_rgb_fullft_v1_delta_30hz/
    tairan_tairan_tomato_sharpa_1016_2_multimodal_fullft_delta_s42_1024_1119_1024_113613/4000
```

对应训练记录位于该工程 `logs/pi05_fr3_sharpa_1016_2_multimodal_rgb_fullft_v1_delta_30hz/` 下同名实验目录，文件前缀 `20261024_113732_812096_1601829`，包含 `.json`、`.experiment.py`、`.log`。归档 JSON 记录训练提交 `42f887916d22cd58c022e95a000e2cb241a8a459`、`extra_delta_transform=True`、训练语义 `delta_arm_joint_target_absolute_hand`、batch 512、FSDP 8、seed 42，四个 A/B normal 批次共 245415 帧、251 个 episode。归档源码摘要与训练记录一致。

任务 prompt 与现有部署一致：`Pick up a tomato truss with the right hand, then pick a cherry tomato with the left hand and place it in the left basket.` 相机为 cam0 头部、cam2 左腕、cam1 右腕。

该 checkpoint 自带统计 asset_id 为 `ui_tairan_tairan_tomato_sharpa_1016_2_multimodal_fullft_delta_s42_1024_104610_delta`。统计 SHA256 与训练记录及实际 GPU 服务读取结果一致：

```text
norm_stats.json:
4f760f029a44ea0c0e25d4001ccd74d7aeb0a925403d0a60e48bb03ed3a78551
params/manifest.ocdbt:
d709ee7969f534ba0a17534cc7e6602ba095c3aa64ecdc976d59f095b900f461
归档 .experiment.py:
1842e40bf333c954ea1babd27ffc8845a9d46f350a3ba4076e49941f452546a3
```

已在 28 的单张 GPU 上用当前部署代码完成真实参数恢复及黑色 RGB／零状态假观测推理，结果为有限 `(50,58)`，输出契约为 absolute。另有非零状态单元测试验证双臂还原、双手不叠加及归一化顺序。GPU smoke 不代表真实相机、机器人运动或番茄任务成功；工作机现场测试仍待完成。

本机原始审计记录和 GPU 报告保存于 `.deployment/full_delta_4000_audit/`（Git 忽略）。没有下载完整权重到本机，也未将 checkpoint 放入 Git。资产目录的 `norm_metadata.json` 在当前远程账号下不可读，因此此处使用训练归档、训练版本源码、checkpoint 自带统计与实际加载结果交叉核对，没有重新审计原始数据集。
