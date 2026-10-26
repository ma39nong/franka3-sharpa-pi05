# Sharpa 30 Hz LoRA 部署版本

全量与 LoRA 已在 `chunk-base-test` 共存，使用同一组入口，通过 `--model full/lora` 选择。当前没有新的 LoRA checkpoint；本目录准备模型加载配置和部署入口，尚未完成真实 LoRA 权重加载或实机验证。

全量版本的对称入口与命令见 [fr3_sharpa_full](../fr3_sharpa_full/README.md)。

## 与全量部署的关系

| 项目 | 全量入口（默认） | LoRA 入口 |
| --- | --- | --- |
| 模型服务 | `serve.sh --model full` | `serve.sh --model lora` |
| 执行端 | `runtime.sh --model full` | `runtime.sh --model lora` |
| 机器／硬件配置 | `.deployment/local.env` | 复用同一文件，不覆盖 |
| 模型配置 | 原 `SHARPA_CHECKPOINT` 等配置 | `.deployment/lora.env` 的 `SHARPA_LORA_*` 配置 |
| 参数变体 | `gemma_2b` + `gemma_300m` | `gemma_2b_lora` + `gemma_300m_lora` |
| 服务端口 | 18765 | 默认 18766，可单独设置 |
| 日志 | `~/.local/state/fr3_sharpa_runtime/execution_traces/` | `~/.local/state/fr3_sharpa_lora/execution_traces/` |

表中脚本均位于 `deploy/fr3_sharpa_runtime/`，`--list-models` 输出集中注册的配置。本目录的 `serve.sh`／`runtime.sh` 仅保留为固定选择 LoRA 的兼容别名。

没有复制机器人执行代码。Home、固定双手配置、SDK 后端、30 Hz 下发、预测裁剪、默认预测 50／执行 15 帧，以及退出清理，均复用当前已部署版本。两个模型服务可以分别启动，但同一套机器人一次只由一个执行端控制。

LoRA 入口不使用全量 `SHARPA_CHECKPOINT`／`SHARPA_NORM_STATS`／日志配置作为后备。新权重缺失时，配置检查会显示 `checkpoint: null`，模型启动或真机执行会明确失败；不会把现有全量模型当作 LoRA 加载。

## 当前 LoRA recipe

- 30 Hz、原生 58D、horizon 50、token 256、bfloat16。
- 模型和实时状态顺序：左臂7、左手22、右臂7、右手22。
- 学习绝对关节目标，`extra_delta_transform=False`；相机 cam0 头部、cam2 左腕、cam1 右腕。
- VLM LoRA rank/alpha = 16/16，Action Expert rank/alpha = 32/32，沿用训练工程 `training_configs/recipes/pi05_lora_v1.py`。
- 配套完整 Orbax checkpoint，包含基础参数及 VLM、Expert 的完整 LoRA 分支，使用该模型自己的 `assets/.../norm_stats.json`。不支持直接提供只包含 adapter 的文件。

参数树检查验证 adapter 分支完整性；rank/alpha 是所选 recipe 的假设，不是从参数名推断的事实。全局参数形状在实际恢复模型时严格核对；alpha 和动作语义仍需由实际训练配置确认。若未来训练使用不同 rank/alpha、频率或 delta 监督，需要先匹配加载配置。

## 还没有 checkpoint 时

以下命令只检查接口和配置，不需要机器人或 LoRA 权重：

```bash
cd /实际的/部署仓库
bash deploy/fr3_sharpa_runtime/serve.sh --model lora --help
bash deploy/fr3_sharpa_runtime/runtime.sh --model lora \
  --check --initialization current --execute-steps 15 --speed slow --num-chunks 1
```

此时 `checkpoint: null` 是预期状态，不表示模型已就绪。`serve.sh --check-only` 需要真实 checkpoint，尚不能通过。

## 权重训练完成后

工作机沿用已经配置好的 `.deployment/local.env`、双臂和双手 Home、网关与相机配置。无需为了切换模型重新生成或覆盖这些文件。如果使用独立 clone，需要按 [工作机交接说明](../../docs/SHARPA_HANDOFF.md)准备其本机路径和 Home 文件。

从部署仓库根目录，用已安装 OpenPI 的 Python 生成 LoRA 独立配置：

```bash
source deploy/fr3_sharpa_runtime/env.sh
"$SHARPA_PYTHON" -m deploy.fr3_sharpa_lora.setup_local \
  --checkpoint /工作机上/新LoRA权重的完整步骤目录
```

这只生成 `.deployment/lora.env`，不修改全量模型、网关或 Home 文件。文件已存在时拒绝覆盖；明确要更新 LoRA 配置时添加 `--force`。若统计文件在 checkpoint 外部，可添加 `--norm-stats /配套的/norm_stats.json`。

也可参考 [local.env.example](local.env.example) 手动创建 `.deployment/lora.env`，或使用 `SHARPA_LORA_ENV_FILE` 指向另一份 LoRA 配置。不要复制示例占位路径后直接执行。

### 权重检查与 GPU 假观测验证

```bash
bash deploy/fr3_sharpa_runtime/serve.sh --model lora --check-only
bash deploy/fr3_sharpa_runtime/serve.sh --model lora --smoke-only --report .deployment/lora_smoke.json
```

应显示 `model_finetune_mode: lora`、两种 `_lora` 变体、30 Hz 和 58D；预热后输出 `(50,58)`。合成测试元数据不能替代这一步真实参数加载。

### 工作机现场运行

终端一，按现有流程检查并启动共享硬件服务：

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh check
bash deploy/fr3_sharpa_runtime/hardware.sh up
```

终端二，启动 LoRA 模型服务并保持运行：

```bash
bash deploy/fr3_sharpa_runtime/serve.sh --model lora
```

终端三，以下命令驱动机器人，首次使用有限轮数：

```bash
bash deploy/fr3_sharpa_runtime/runtime.sh --model lora \
  --execute --enable-hardware --sharpa-backend sdk_direct \
  --prepare-speed-rad-s 0.03 --execute-steps 15 --speed slow --num-chunks 1
```

从当前状态持续执行：

```bash
bash deploy/fr3_sharpa_lora/runtime.sh \
  --execute --enable-hardware --speed slow --num-chunks 0
```

先回 Home，再持续执行：

```bash
bash deploy/fr3_sharpa_lora/runtime.sh \
  --execute --enable-hardware --speed slow --num-chunks 0 --home
```

中途 Ctrl-C 后，等待失能确认与硬件清理结束；保持模型服务运行，再按顺序恢复：

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh up
bash deploy/fr3_sharpa_lora/runtime.sh \
  --execute --enable-hardware --speed slow --num-chunks 0
```

若模型服务也已停止，先重新启动模型终端。

默认执行命令从当前状态开始。需要先回固定双臂／双手 Home 时，在该执行命令末尾加 `--home`（等效 `--initialization home`）；Home 后会继续观测、推理和执行。原命令若还带 `--initialization home`，请删除后再用于中途恢复。

Ctrl-C 仍按现有流程失能确认并清理硬件；再次执行前先运行 `bash deploy/fr3_sharpa_runtime/hardware.sh up`。重新启动读取此时的状态和图像，不恢复上次未执行的动作，也不保证失能期间保持抓取。

缺少或误选权重会在进入硬件会话管理器之前失败。完成真实单轮验证后，才按实际任务需要调整为 `--num-chunks 0` 持续重推。停止和硬件清理行为与全量版本相同，见 [共享 runtime 说明](../fr3_sharpa_runtime/README.md)。
