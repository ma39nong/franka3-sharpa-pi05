# Sharpa 全量微调部署

这是全量模型的日常入口，与 `../fr3_sharpa_lora/` 对称。两者共存于 `chunk-base-test`，相机、Home/prepare、三档速度、SDK 和执行循环共用 `../fr3_sharpa_runtime/` 的实现。

## 配置

沿用工作机已有 `.deployment/local.env`，不需要迁移或重新生成。权重使用 `SHARPA_CHECKPOINT`，统计使用 checkpoint 自带的唯一 `assets/**/norm_stats.json` 或显式 `SHARPA_NORM_STATS`；端口默认 18765，可用 `SHARPA_FULL_PORT` 设置。日志沿用 `SHARPA_EXECUTION_TRACE_DIR`，默认 `~/.local/state/fr3_sharpa_runtime/execution_traces/`。

当前模型：30 Hz、58D absolute、horizon 50，`gemma_2b` + `gemma_300m`。每轮默认执行前 15 帧，再重新观测推理。初始动作先到双臂 powder_weighing Home 和固定双手 Home。

## 离线检查

从仓库根目录运行；`--check` 不需要机器人或模型权重：

```bash
bash deploy/fr3_sharpa_full/runtime.sh --check --speed slow --num-chunks 1
```

以下两条需要工作机真实 checkpoint；第二条需要 GPU，仅使用假观测：

```bash
bash deploy/fr3_sharpa_full/serve.sh --check-only
bash deploy/fr3_sharpa_full/serve.sh --smoke-only --report .deployment/full_smoke.json
```

## 工作机启动

先按现有流程检查并启动共享硬件：

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh check
bash deploy/fr3_sharpa_runtime/hardware.sh up
```

模型终端：

```bash
bash deploy/fr3_sharpa_full/serve.sh
```

执行终端，以下命令会驱动机器人：

```bash
bash deploy/fr3_sharpa_full/runtime.sh \
  --execute --enable-hardware --sharpa-backend sdk_direct \
  --initialization home --home-task powder_weighing \
  --prepare-speed-rad-s 0.03 --execute-steps 15 --speed slow --num-chunks 1
```

确认单轮后，按任务需要设置 `--num-chunks 0` 连续运行。执行终端 Ctrl-C 会清理其所属硬件会话；模型服务需在模型终端单独停止。

## 目录与兼容性

- `serve.sh`：固定选择 full 的模型服务入口。
- `runtime.sh`：固定选择 full 的执行入口。
- `env.sh`：供手动 source 的全量环境入口。
- `../fr3_sharpa_runtime/`：共享实现与硬件工具；旧的启动命令继续有效。

本目录脚本转发到共享入口的 `--model full`，不复制模型加载或机器人执行代码，也不修改现有工作机配置。完整现场配置、观测及退出行为见 [公共 runtime 说明](../fr3_sharpa_runtime/README.md)。LoRA 权重接入见 [LoRA 说明](../fr3_sharpa_lora/README.md)。
