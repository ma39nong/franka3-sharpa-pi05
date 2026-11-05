# FR3/Sharpa Mamba-3 部署

代码适配已接入；**完整训练权重验收尚未完成**。2026-11-05 本机只找到
`diagnostic_only=true` 的训练测试产物，生产入口会拒绝它们。合成模型测试不代表任务训练完成、
完整 Gemma-2B checkpoint 推理成功或实机验收。

## 版本与改动范围

- 本地分支：`mamba3-jax-deployment`。
- 部署基线：`train-rtc` / `e04dade1707af6d6f574e8b4890c54705c855980`。
- 训练源码：`LTT-BIT/tomato-openpi` / `0117bb5bdc26874082cb462764131cfad037df7a`。
- 当前实现保留为工作区修改，未创建实施 commit、未推送；HEAD 仍为部署基线。
- 源文件 SHA256、依赖版本及许可证见 `third_party/mamba3/deployment_source.json`、
  `third_party/mamba3/NOTICE` 和两份 Apache-2.0 许可证。
- `gemma.py`、`siglip.py`、`pi0.py`、原有模型配置及训练器未被覆盖。
  Mamba 的标准高斯 helper 与训练 `Pi0.noise` 等价；Paper 直接保留训练字段及 `z @ L.T`。
  部署配置去除了训练专用 hooks；RTC 字段留在 Mamba 子类中。
- 不改 RTCStream、收据、时钟、动作适配器、机器人 SDK/ROS/相机、急停或 Home 流程。
  JAX 0.5.3 / Flax 0.10.2 锁文件未变。验证环境仅补装了原客户端已声明的 `websockets`。

## Profile 与动作语义

| Profile | 训练动作 / 归一化 | 部署输出 | 默认端口 | 环境变量前缀 |
|---|---|---|---|---|
| `full` | 原有 Absolute / quantile | Absolute | 18765 | `SHARPA` |
| `lora` | 原有 LoRA Absolute / quantile | Absolute | 18766 | `SHARPA_LORA` |
| `full-delta` | 原有 Delta / Paper 等原路径 | Absolute | 18767 | `SHARPA_FULL_DELTA` |
| `mamba` | Absolute / ordinary mean/std | Absolute | 18768 | `SHARPA_MAMBA` |
| `mamba-delta` | 双臂 Delta、双手 Absolute / Paper | Absolute | 18769 | `SHARPA_MAMBA_DELTA` |

所有输出均为物理弧度 `(50,58)`、30Hz；顺序为 LA7/LH22/RA7/RH22。
图像 slot 为 base/left/right，分别来自 cam0/cam2/cam1。Mamba 仍使用 Pi05 的文本/离散 state
Tokenizer。`action_expert_variant=gemma_300m` 仅为训练 dataclass 的兼容字段，
不会构造 Gemma Action Expert；实际专家由嵌套 `mamba` 和深度等字段定义。

Mamba 两个 profile 无默认权重，不读取旧 `SHARPA_CHECKPOINT`。
各前缀可设置 `_CHECKPOINT`、`_NORM_STATS`、`_PORT`、`_EXECUTION_TRACE_DIR`。
默认文件分别为 `.deployment/mamba.env`、`.deployment/mamba_delta.env`；
可用 `SHARPA_MAMBA_ENV_FILE` / `SHARPA_MAMBA_DELTA_ENV_FILE` 指定。
调用方显式 Mamba 设置优先于该 profile 文件；`--model` 优先于继承的 profile。
本次检查时 18768/18769 未监听；实际服务仍需成功绑定端口。

## Checkpoint 契约

生产目录必须完整保留：

```text
step/
  params/manifest.ocdbt
  params/_METADATA
  params/...                 # 实际 Orbax 数据文件，不能只有 metadata
  assets/<asset_id>/norm_stats.json
  assets/<asset_id>/mamba_architecture.json
  assets/<asset_id>/delta_paper.json       # Paper 必须
  assets/<asset_id>/delta_paper.npz        # Paper 必须
  assets/<asset_id>/norm_metadata.json     # Paper 必须
```

`--check-only` 只读磁盘和 NumPy，不导入 JAX、ROS 或 SDK，也不宣称实际权重可恢复。
完整数据读取、所有参数 shape/dtype/有限值检查发生在真实恢复时。

`mamba_architecture.json` 使用训练端原生 schema v1，顶层严格为
`schema_version, model, upstream, norm_stats_sha256`，Paper 另有 `paper_files_sha256`。
`model` 的完整字段集见 `third_party/mamba3/config_schema.json`；该文件中的值是训练默认值，
实际模型配置始终来自 checkpoint。未知/缺失字段、错误 stage、非 FullFT、非 H50/D58/token256、
错误 dtype 或 mixer 配置会被拒绝。生产 VLM 必须 `gemma_2b`，Pi05/discrete state 必须开启。

受支持的 upstream 精确为：

```json
{
  "mamba3_jax": "ffa55848fc513ca91aa330bda4c33c86dab74827",
  "MambaAct": "97414cfffc82225c6be34abf486f06c7eb11d22d"
}
```

仅允许唯一 checkpoint 自有资产目录。普通统计 state/actions 均须有限 58D；Paper 动作统计须
50×58，并验证 bundle、manifest、audit 三文件摘要、内部算法及噪声 factor。
Paper 的归档 `paper_asset_dir` 只有在内容校验后才定位到当前 checkpoint 的 assets。
普通 Mamba 禁止混入 Paper 资产。

参数树只能包含 VLM-only `PaliGemma`、`context_compressor`、`action_expert`，以及必要的
`paper_noise_cholesky`。磁盘检查不把 Orbax `write_shape` 当成全局 shape。
真实恢复不裁剪多余参数、不把 FP32 存储强转 BF16，完整检查 shape/dtype/有限值；
Paper 恢复的噪声 factor 必须逐值匹配已校验 bundle。推理只恢复 params，不使用 optimizer，
也不使用训练初始化器；EMA 是否被导出为 params 由训练 checkpoint 决定。

握手额外绑定 `model_architecture`、`mamba_manifest_sha256`、`mamba_stage`、`mamba_config_hash`、
`mamba_normalization`、`mamba_provenance` 和 checkpoint metadata 摘要。
缓存检查 manifest/统计/Paper 文件身份与资产清单，增加重复资产、改文件或移除文件都会重新校验。
服务在模型恢复和预热后再次核对资产身份。旧 Transformer 握手不强制新增字段。

这些 hash 用于检测身份冲突/文件替换，并不是训练来源的数字签名。
训练 schema 未包含训练提交、训练步数、数据频率或任务质量的独立证明；
`mamba_provenance.supported_source_commit` 表示本实现支持的源码版本，不能解释为权重的已签名来源。
仍须另行核对实际训练记录、30Hz 数据及评估结果。

## RTC 模式矩阵

| 权重 | Serial | 软 RTC / Paper RTC | 训练式硬前缀 |
|---|---|---|---|
| 旧 Transformer 模式 | 原功能 | 原合法组合 | 原合法组合 |
| Mamba Plain，delay=0 | 支持 | 拒绝 | 拒绝 |
| Mamba Plain，delay>0 | 支持 | 拒绝 | `--enable-train-rtc` |
| Mamba Paper，delay=0 | 支持 | 拒绝 | 拒绝 |
| Mamba Paper，delay>0 | 支持 | 拒绝 | `--enable-train-paper-rtc` |

Mamba 的 delay 只读取已校验架构 manifest 中 `rtc_max_delay`，范围 0..49。
旧 `SHARPA_TRAIN_RTC_MAX_DELAY`、手工确认及旧训练 JSON 不构成 Mamba RTC 证据。
UI 从 manifest 识别 profile 和支持模式，保留原有用户执行授权步骤。

前缀始终从物理 Absolute canonical 58D 转换，按**新观测 state 和新时间索引**重新编码。
`prefix_length` 为静态 JIT 参数。前缀在第一个积分步前插入，并在每次 Euler 更新后硬保持；
反归一化后复制已承诺物理前缀，消除浮点往返误差，与源离线 Policy 一致。
condition/FiLM 只在本次 query 内缓存，每个去噪步重新扫描 SSM，不跨请求保留 carry。
服务启动逐个预热 d=0..max_delay，Paper 同时检查 K=50/35/1，完成后才宣布 ready。
超出训练最大延迟时使用既有拒绝/停止协议。

## 启动与无硬件验证

先选择已安装项目依赖的 Python，不需要升级锁文件。

```bash
export SHARPA_PYTHON=/path/to/openpi-environment/bin/python
export SHARPA_MAMBA_CHECKPOINT=/path/to/trained_mamba/step
# 只有唯一自有统计时可以省略 NORM_STATS
export SHARPA_MAMBA_NORM_STATS="$SHARPA_MAMBA_CHECKPOINT/assets/<asset_id>/norm_stats.json"
bash deploy/fr3_sharpa_runtime/serve.sh --list-models
bash deploy/fr3_sharpa_runtime/serve.sh --model mamba --check-only
bash deploy/fr3_sharpa_runtime/runtime.sh --model mamba --check
bash deploy/fr3_sharpa_runtime/serve.sh --model mamba --smoke-only --report /tmp/mamba-smoke.json
# 完成 smoke 后启动 WebSocket 服务（不连接机器人）
bash deploy/fr3_sharpa_runtime/serve.sh --model mamba
```

Paper / 训练式 RTC：

```bash
export SHARPA_MAMBA_DELTA_CHECKPOINT=/path/to/trained_paper_mamba/step
bash deploy/fr3_sharpa_runtime/serve.sh --model mamba-delta --check-only --enable-train-paper-rtc
bash deploy/fr3_sharpa_runtime/serve.sh --model mamba-delta --smoke-only --enable-train-paper-rtc
bash deploy/fr3_sharpa_runtime/serve.sh --model mamba-delta --enable-train-paper-rtc
# 对应客户端选项：--model mamba-delta --broker-mode rtc --train-paper-rtc
# Plain 训练式 RTC 对应 --enable-train-rtc / --train-rtc。
```

服务默认采样 NFE=10，来自训练模型的 `sample_actions` 默认值，不由执行频率推导。
直接调用 Policy 的显式 noise 只接受有限 `(50,58)` 或 `(1,50,58)`；未给 noise 时按请求推进 RNG。
真实机器人启动仍使用原有准备、现场授权、preflight、设备与停止流程。
此任务未运行任何硬件命令。

## 复现本次测试

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PWD"
export JAX_PLATFORMS=cpu
# 原 SDK 单元测试需要现有机器的真实 schema 路径，无需导入或连接设备。
export SHARPA_SCHEMA_PATH=/path/to/teleop/config/litchibot/sharpa_schema.json
export SHARPA_MAMBA_SOURCE=/path/to/tomato-openpi  # HEAD 须为上述训练 SHA
"$SHARPA_PYTHON" -m pytest deploy/fr3_sharpa_runtime/tests -q --disable-warnings
# GPU 数值检查：合成输入，含生产尺寸 4×512 专家，不含完整生产 VLM。
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$SHARPA_PYTHON" -m pytest deploy/fr3_sharpa_runtime/tests/test_mamba_numerics.py -q
```

没有 `SHARPA_MAMBA_SOURCE` 时，跨仓库同权重对照会明确 SKIP。
该测试在独立训练进程导出同一权重/输入，再在部署端对比 hidden、compressor、完整采样和
Plain 物理动作（FP32，NFE=2，合成小 VLM，容差 rtol/atol=1e-4；hidden/compressor 为 1e-5）。
同样覆盖静态前缀 0/1/3。它不替代完整已训练 checkpoint 的数值对照。

最终完整测试 **534 项通过**（基线 472 项）；独立 GPU 核心测试 **10 项通过**。
验证汇总和 CPU/GPU 数值误差保存在 `docs/mamba3_validation/`。
硬件：RTX 4070 Ti SUPER 16GB。合成 16D Mixer 在 GPU 上对独立 Torch 参考的 FP32 relative L2
约 1.84e-7；具体 dtype/梯度误差见 JSON。没有测量完整模型端到端 p50/p95、冷启动或峰值显存，
不作速度优于 Transformer 的结论。

## 验收状态

| ID | 状态 | 证据 / 限制 |
|---|---|---|
| C01 | PASS | 独立本地分支，base=HEAD=e04dade；修改未提交，旧分支未写入 |
| C02 | PASS | 生产配置 shape tree、无旧 expert/LoRA、严格完整恢复单测 |
| C03 | PASS | 固定训练 SHA、source hashes、NOTICE、许可证；锁文件未改 |
| C04 | PASS | CPU/GPU 独立 SSD/Torch 数值、H50 JIT、跨调用隔离 |
| C05 | BLOCKED | 合成同权重全前向/Plain 物理对照通过；缺完整已训练权重 |
| C06 | BLOCKED | 合成同权重 VLM hidden 和三相机顺序通过；完整 VLM 待实权重 |
| C07 | PASS | 58D mean/std 数据桥、单次逆变换、错资产拒绝 |
| C08 | PASS | Paper hash、时间统计、新 anchor、beta>0 噪声及恢复绑定单测 |
| C09 | PASS | 配置重建、握手 hash/stage、旧模式兼容测试 |
| C10 | PASS | manifest/统计/Paper/树篡改及缓存失效测试 |
| C11 | PASS | 未改基线472项通过；修改后完整回归通过，registry 测试仅扩展新增 profile |
| C12 | BLOCKED | 合成 Mamba 实际 TCP/WebSocket/Msgpack→既有 ActionAdapter 通过；实权重待验 |
| C13 | PASS | 检查模式、客户端和 UI 均拒绝 Mamba 软 RTC |
| C14 | BLOCKED | 合成 d=0..3、逐步硬固定/收据通过；实际训练最大 delay 编译待验 |
| C15 | BLOCKED | 合成 Paper K50/35/1、新 state/索引、物理前缀通过；实权重待验 |
| C16 | PASS | 未修改协议/队列；旧 RTC stale/receipt/延迟拒绝回归 |
| C17 | PASS | Mamba↔Full 切换、错误 profile、旧默认与环境隔离测试 |
| C18 | BLOCKED | 缺完整 checkpoint；未测生产 GPU 延迟/编译/显存 |
| C19 | PASS | 硬件实现未改；Mamba 失败在 ROS/SDK 会话之前的入口测试 |
| C20 | PASS | 本说明、来源、测试指令、数值记录及未完成项明确列出 |

上线前仍需提供完整已训练 Plain/Paper checkpoint，验证真实 RGB/prompt/state/noise 下的
训练离线 Policy ↔ 部署 Policy 物理动作对照，逐 delay 的编译和时延/显存，
再走原有设备准备和现场验证。本代码变更不能据此标记为已完成实机部署。
