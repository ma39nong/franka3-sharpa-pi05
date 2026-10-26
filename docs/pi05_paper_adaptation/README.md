# π0.5 滚动参考、Soft Inpainting 与动作压缩

这些是默认关闭的实验功能，基于 `chunk-base-test@64f8f66`。支持 `full` 和 `full-delta`，复用原来的硬件、Home、SDK、安全网关、速度档和退出清理。旧命令仍预测 50、执行 15，默认从当前状态启动；本功能不新增可训练参数、不改变已有 checkpoint 或 norm stats。

## 算法与限制

参考 [论文 §7.1–7.2](https://arxiv.org/abs/2512.06951v2) 和[作者实现 ca556f7](https://github.com/IliaLarchenko/behavior-1k-solution/tree/ca556f74a455cef7987a2be4537b5ac85cc56dd7)。这里复用既有权重，始终使用训练时的独立高斯噪声；相关性传播是推理消融，不是完整复现作者相关噪声训练。

- 网络仍生成 `50×58`；实验消费 `[0:26]`，保存 `[26:30]` 四点，丢弃 `[30:50]`。
- 缓存保存在执行端，只有成功消费完整 26 点才提交。新一轮先用最新实测状态重建参考：Delta 模型仅双臂减去状态，44 维手部仍为 absolute，然后使用模型自己的 quantile normalization。服务输出仍为 rad 的绝对位置，执行端不再加状态。
- `soft` 在 Euler 更新后，按更新后的 `t_new` 约束前四点到 `(1-t_new)×reference+t_new×fixed_noise`，仅在 `t_new>0.3` 应用；浮点比较留 `1e-6` 容差。默认 10 次更新中作用于 0.9…0.4 六次，后续自由生成，不在输出末端粘贴旧动作。
- `correlation` 额外使用当前模型的完整 2900×2900 归一化动作协方差。O 为前 `4×58` 维，U 为其余 `46×58` 维。使用 `0.5Σ+0.5I` 和线性求解；传播幅度超过归一化值 1 时跳过本次 U 修正并计数，O 约束不变。统计资产校验身份、维度、有限性、对称性和半正定性（允许 `1e-6` 数值容差），不提供其他任务的默认资产。
- `cubic` 对绝对输出前 26 点，沿索引 `[0,25]` 均匀重采样成 20 点，首尾不变，所有关节共用时间轴。任一手指在当前反馈和原 26 点中的范围超过 `0.05 rad`，整段取消压缩；这是初始实验阈值，不是已标定接触判据。设备 worker 会再用其最新反馈核对一次。
- 压缩结果严格检查 native 范围及完整衔接/插值下发序列；不使用预测裁剪额度修补样条过冲。失败则退回原 26 点，再按旧路径检查和预测裁剪；原序列仍失败时停止。取消压缩仍消费 26 点、保存 `[26:30]`，与作者取消压缩后消费 20 点的代码不同。

**本版本按轨迹进度关联参考，不是严格物理时间对齐的实时 chunking。** 20 个压缩点表示消费 26 个源点；slow/medium 的时间伸缩、首帧衔接和串行推理停顿分别记录，不把 26 点解释为真实过去了 26/30 秒。不包含异步推理、RTG、Ruckig 或任务完成判断。

## 工控机使用

以下全部在工控机仓库根目录执行。此开发轮只完成本地 CPU 离线验证；这些命令提供现场接入方式，**没有在本轮执行硬件启动或机器人运动**。

先按原部署流程确认权重、现场配置及设备身份。硬件终端沿用：

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh check
bash deploy/fr3_sharpa_runtime/hardware.sh up
```

模型终端，以 Delta 为例：

```bash
bash deploy/fr3_sharpa_full/serve_delta.sh --enable-inference-experiments
```

等待原 `Warmup OK`、新增 `Inference experiments ready: paper-rolling-v1` 以及服务监听信息。服务会在监听前预热实际使用的有参考 sampler；启动实验模型会额外消费随机数，旧服务默认启动流程不变。

执行前可先做不连接设备的参数检查：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh --check \
  --inference-preset paper-rolling --inpainting soft --compression cubic --speed slow
```

执行终端，以下会发送机器人运动指令，首次只跑一轮：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --execute --enable-hardware --sharpa-backend sdk_direct \
  --initialization current --speed slow --num-chunks 1 \
  --inference-preset paper-rolling --inpainting soft --compression cubic
```

第一轮没有旧参考，验证跨轮效果需要后续现场选择有限多轮。确认后才考虑 `--num-chunks 0`。需要启动前回 Home 时使用 `--home` 替换 `--initialization current`。全量 absolute 对照使用 `serve.sh` / `runtime.sh` 替换上述 `_delta.sh` 入口，两端模型须一致。

独立消融方式：

| 模式 | 执行端参数 |
|---|---|
| 旧基线 50/15 | 不传实验参数 |
| 50→30→26+4，无约束 | `--inference-preset paper-rolling` |
| Soft，无压缩 | 再加 `--inpainting soft` |
| 仅压缩 | 再加 `--compression cubic` |
| Soft + 压缩 | 同时加以上两个参数 |
| 相关传播 | 用 `--inpainting correlation`，服务端必须有匹配协方差资产 |

preset 下 `--execute-steps` 只能省略或设 26。`--reference-max-age-s` 默认 1.0，`--hand-variation-rad` 默认 0.05。实验模式仅允许 current/Home 初始化、sdk_direct，以及 check/execute；合成评估使用下方独立工具。

缓存年龄从上一轮命令完成到下一次开始推理计算。开始前超时则清空参考，进行无参考推理；参考相关响应返回时已过期则停止，不下发迟到结果。无参考的实验请求同样有请求起算的 1 秒响应期限。现有执行入口更严格的 **200ms 观测年龄检查仍保留**，调整参考有效期不会放宽它。

每次正常推理新建 WebSocket 不清空参考。握手的服务启动 UUID、模型或统计身份变化会清空参考；请求失败、中断、部分执行、Home 或进程重启不会携带旧缓存继续。执行端退出后的双手失能确认及硬件清理沿用原实现；模型服务仍需单独停止。

## 协方差资产

输入必须是显式准备的 NPZ：`absolute_actions: (N,50,58)`、`states: (N,58)`，N≥2，canonical 顺序，rad，50 点均针对同一观测起点；states 为各 chunk 起点实测状态。由数据准备者保证它们来自当前任务和训练数据来源，不能拿留出测试集拟合统计。工具不猜测原始 116D 或 action58 的排列，也不修改原来的 norm stats。

```bash
source deploy/fr3_sharpa_runtime/model_env.sh --model full-delta
"$SHARPA_PYTHON" -m deploy.fr3_sharpa_runtime.correlation \
  --input /工控机上/已准备的训练动作窗口.npz \
  --output /工控机上/独立资产/sharpa_delta_covariance.npz

bash deploy/fr3_sharpa_full/serve_delta.sh --enable-inference-experiments \
  --covariance-asset /工控机上/独立资产/sharpa_delta_covariance.npz
```

输出拒绝覆盖已有文件；包含来源文件 SHA256、样本量、模型/统计身份、训练动作语义、频率、顺序、完整 horizon 和 flatten 顺序。估计过程复用现有 quantile normalization，Delta 仅双臂减 chunk 起点状态，协方差按 N−1 计算。当前工具在内存中计算完整矩阵和样本数组，适用于已准备好的窗口集，应预留 RAM。本轮没有真实任务协方差资产。

## 离线验证与日志

已有 OpenPI 环境：

```bash
source deploy/fr3_sharpa_runtime/env.sh
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 "$SHARPA_PYTHON" -m pytest deploy/fr3_sharpa_runtime/tests -q

JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 "$SHARPA_PYTHON" -m deploy.fr3_sharpa_runtime.paper_replay \
  --seed 42 --rounds 5 --report /tmp/paper-fast.json
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 "$SHARPA_PYTHON" -m deploy.fr3_sharpa_runtime.paper_replay \
  --seed 42 --rounds 5 --speed-scale 0.1 --report /tmp/paper-slow.json
```

`paper_replay` 使用玩具向量场、合成观测和解析合成协方差，**不加载真实权重，也不是机器人动力学仿真**。报告按四组关节输出目标跳变、玩具目标 RMSE、实际合成时间轴上的指令速度/加速度、衔接点数和耗时。CPU 计算耗时含测试框架/JIT，不能解释为真实模型推理速度。旧 15 点基线与 26 点实验重规划间隔不同；效果归因应优先比较相同 26 点模式。

新增 chunk 日志保留原始 50 点预测、请求参考、服务端是否实际使用参考、采样修正次数、协方差资产身份、候选执行目标、压缩及回退原因。成功的 worker 报告还包含最终执行目标和全部衔接/时间伸缩后的指令点，消费模型点数与实际下发数量分开保存。仍沿用事件和 run 日志；`commands_sent`/`COMPLETE` 不能解释为全部关节到位或番茄采摘成功。

审计与本轮结果见 [WP0_AUDIT](WP0_AUDIT.md) 和 [合成结果](SYNTHETIC_RESULTS.md)。
