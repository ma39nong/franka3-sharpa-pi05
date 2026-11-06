# Mamba-3 部署端 Soft RTC

本轮在 `mamba3-jax-deployment` 的 `35071da7c3b0c5e996c936ec4775ef31488c4200`
上增量修改，保留为本地未提交内容，不提交、不推送、不修改训练仓库。
训练分支已复核，仍固定 `0117bb5bdc26874082cb462764131cfad037df7a`。

用户说明正式训练尚未开始，因此真实 checkpoint 适配/验收和生产性能测量本轮暂缓。
本次结果覆盖软件契约、合成模型数值、真实 TCP/WebSocket 及虚拟设备测试，**不代表实机可用**。
原架构/资产校验不放宽，也不会把 diagnostic 权重放入生产服务。

## 模式

| Checkpoint | Serial | 部署端软引导 | 训练式硬前缀 |
|---|---|---|---|
| Plain，rtc_max_delay=0 | `mamba` | `rtc` | 拒绝 |
| Paper，rtc_max_delay=0 | `mamba-delta` | `paper-rtc` | 拒绝 |
| Plain，rtc_max_delay>0 | `mamba` | 拒绝 | `train-rtc` |
| Paper，rtc_max_delay>0 | `mamba-delta` | 拒绝 | `train-paper-rtc` |

软引导不保证前缀逐值相等，不执行末端前缀粘贴。`soft-serial` / `soft-async` 是原有的压缩与
rolling 算法，本轮仍不向 Mamba 开放。旧 Transformer/LoRA 模式维持原有合法组合。

服务、客户端 `--check`、Shell 启动前检查和 UI planner 共用已校验的模型训练/Paper身份来决定模式。
不增加 checkpoint 字段、wire 字段或协议版本。模型 identity / 模式变化仍使原 UI 核对失效。

完整 T3 **配置**必须同时满足 `paper_mode=timestamp_zscore_arms`、`paper_beta>0`、
`rtc_max_delay>0`。`train-paper-rtc` 这个名称本身不足以判断 T3，更不能证明训练已完成。

## 计算路径

部署侧 `MambaPi0.sample_actions()` 可选接受：

```python
rtc_actions=None                  # (B,50,58)，当前模型的归一化动作空间
rtc_weights=None                  # (B,50,1)
rtc_max_guidance_weight=5.0        # 标量
```

无 `rtc_actions` 时保留原速度调用和 Euler 算术，不创建 VJP；普通/训练式硬前缀采样语义不变。
未修改 `compute_loss()`、模型构造或任何 Param/Variable。Plain/Paper 噪声先验沿用原实现。

有软引导时直接复用 Transformer `_rtc_guided_velocity`：

```text
x0_hat = x_t - t * v(x_t, t, condition)
error = (target - x0_hat) * weights
g = VJP_x(x0_hat)(error)
v_guided = v - clip(nan_to_num((t²+(1-t)²)/(t*(1-t))), 0, cap) * g
x_next = x_t + (-1/NFE) * v_guided
```

helper 使用标量时间；Mamba 专家单独接收 `(B,)` 时间，避免错误广播到 `(B,50,58)`。
condition/FiLM 每个 query 计算一次，VJP 只对当前动作输入求导；aux 复用速度结果。
SSM 每次速度计算重新扫描，未增加跨步或跨请求 carry，没有 `stop_gradient` 或近似 VJP。

Host Policy 在 JIT 前校验协议、有限值、动作/权重 shape 和 guidance cap。
低层采样器也检查结构和可用的非 tracer 数值；JIT 后在 host 拒绝非有限物理动作。
wire cap 仍为 `(0,100]`；低层零 cap/零权重可用于无引导等价测试。

Plain 使用原 ordinary mean/std。Paper 复用 `PaperRTCPolicy` / `PaperRTCPrefixTransform`：
以本次新 state 重锚双臂 `0:7,29:36`，补齐 H50，按新窗口索引归一化，再截回 K。
手部保持 Absolute，最后按原 Paper bridge 逆变换一次。Soft Policy 不复制物理承诺前缀；
只有原训练式 Hard Policy 保留精确物理前缀的往返误差修正。

## 预热和服务

新增 `warmup_soft_rtc()` 实际运行非零权重的 JIT/VJP 请求、同步转换为 host 数组并验证 receipt。
Plain 运行两次完整参考，Paper 另检查 K35/K1。所有检查成功后才返回 `rtc_warmed_up=True`，
Paper 同时返回原 adapter 和短前缀能力。Serial 请求仍无 RTC receipt。

`rtc_soft_warmup` 记录每次同步 policy wall time、参考重合 MAE/P95/Max、首帧和速度差。
首次时间包含编译；这些 dummy observation 的少量预热数字不是生产 p50/p95、显存或 deadline 验收。
`performance_admission=NOT_MEASURED` 明确标识该限制。软引导不保证任意模型/轨迹上的误差都下降。

```bash
export SHARPA_PYTHON=/path/to/openpi-environment/bin/python
export SHARPA_MAMBA_CHECKPOINT=/path/to/future_plain_mamba/step
bash deploy/fr3_sharpa_runtime/serve.sh --model mamba --enable-rtc --check-only
bash deploy/fr3_sharpa_runtime/runtime.sh --model mamba --check \
  --broker-mode rtc --speed fast --execute-steps 15 --num-chunks 2
# 有真实完整权重后才能执行下列 GPU 离线检查；不会连接设备
bash deploy/fr3_sharpa_runtime/serve.sh --model mamba --enable-rtc --smoke-only \
  --report /tmp/mamba_soft_rtc_smoke.json
```

Paper 的对应选项：

```bash
export SHARPA_MAMBA_DELTA_CHECKPOINT=/path/to/future_paper_mamba/step
bash deploy/fr3_sharpa_runtime/serve.sh --model mamba-delta --enable-paper-rtc --check-only
bash deploy/fr3_sharpa_runtime/runtime.sh --model mamba-delta --check \
  --broker-mode rtc --paper-rtc --speed fast --execute-steps 15 --num-chunks 2
bash deploy/fr3_sharpa_runtime/serve.sh --model mamba-delta --enable-paper-rtc --smoke-only \
  --report /tmp/mamba_paper_soft_rtc_smoke.json
```

有 clean-prefix 训练证据时仍使用原 `--enable-train-rtc` / `--enable-train-paper-rtc`，不能叠加 Soft。
profile、默认端口、默认 NFE=10、归一化资产和训练模型的参数树均未改变。

## 验证与复现

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PWD"
export SHARPA_SCHEMA_PATH=/path/to/teleop/config/litchibot/sharpa_schema.json
export SHARPA_MAMBA_SOURCE=/path/to/tomato-openpi
JAX_PLATFORMS=cpu "$SHARPA_PYTHON" -m pytest deploy/fr3_sharpa_runtime/tests -q --disable-warnings
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$SHARPA_PYTHON" -m pytest \
  deploy/fr3_sharpa_runtime/tests/test_mamba_soft_rtc_math.py \
  deploy/fr3_sharpa_runtime/tests/test_mamba_soft_rtc_policy.py -q --disable-warnings
```

改动前全量基线为 **534 PASS**；本轮 CPU 全量 **593 PASS**，GPU Soft 数学/Policy **30 PASS**。
报告、实际命令及数值记录见 `docs/mamba3_soft_rtc_validation/`。
硬件为 RTX 4070 Ti SUPER 16GB，JAX 0.5.3 / Flax 0.10.2，未升级依赖或锁文件。

- 数学：可解析线性场、反向符号、端点/上限、JVP/VJP adjoint 和独立有限差分。
- 模型：B1/B2 H50 D58、零权重等价、查询隔离、每次只计算一次 condition、输入拒绝。
- 无引导：直接读取修改前提交中的采样实现，同权重/固定噪声比较 Plain/Paper，结果逐值相等，且不调用引导 helper。
- 结构：完整 NNX 状态键/shape/dtype 和值在引导前后不变，无新增可训练或非训练状态。
- Policy：Plain mean/std、Paper 新 state/新时间轴 K1/35/50、receipt、Soft/Hard 互斥和非有限输出拒绝。
- 服务：两个 profile 的 `--check-only` 不导入 JAX/ROS；真实 WebSocket 普通/引导请求，小模型 JIT/VJP 预热。
- Timeline：真实小 Mamba Policy + 假 worker/虚拟时间接受五块，等待时继续消费旧轨迹；迟到、300ms 旧观测、错误收据和耗尽均停止。
- 旧回归：原 Full/LoRA/Delta/Paper/Train RTC/Train Paper RTC、协议和 UI 检查全部保留。

FP32 数值容差说明：解析线性夹具显式使用 HIGHEST dot precision，避免 GPU 默认低精度
矩阵乘法污染数学参考。零权重结果容差 1e-6；eager 与 fused VJP 对照使用 rtol=1e-5、
atol=3e-5（CPU 已观测最大约 2.15e-5）。GPU 不同 batch 尺寸会改变归约顺序，
完整 guided sampler 的 batch/逐样本对照同时要求 relative L2<1e-5、max abs<1e-4；
重复相同 compiled query 要求逐值相等。实际误差写入数值 JSON，不将此差异当作状态串扰。

## 尚未进行的验收

| 项目 | 状态 | 原因 |
|---|---|---|
| M01–M08、P03–P08、S01/S02/S04–S07 软件/合成测试 | 见验证汇总 | 不涉及真实训练权重 |
| P01/P02 真实 checkpoint Serial 对照 | BLOCKED | 用户暂缓；正式训练尚未开始，合成同权重检查已做 |
| S03 生产 GPU `serve --smoke-only` | BLOCKED | 缺真实完整权重，合成 JIT/VJP 预热不替代该项 |
| S08 实权重 WebSocket→物理动作→虚拟安装 | BLOCKED | 同上 |
| 生产编译时间、p50/p95、GPU 峰值/增量显存 | BLOCKED | 不使用合成小网络代替完整模型测量 |
| 重合/位置/速度 Gap 与真实 RTC deadline 满足率 | BLOCKED | 需同权重、同输入、同 NFE、目标机器连续 query |
| 机器人执行 | NOT_RUN | 本次未授权、不在本轮范围 |

真实模型必须在原时间预算内完成观测、推理、网络及安装，并扣除原 50ms 余量，
再按原设备 preflight 和单独现场授权进行验证；不能因 VJP 变慢而放宽截止时间。
本轮没有改动 RTC wire schema、timeline、coordinator、runtime client、ROS/SDK、相机、录制、
关节限位、30Hz 时钟、200ms 观测阈值或失能规则；UI 模式可选不等于实机准入已通过。
