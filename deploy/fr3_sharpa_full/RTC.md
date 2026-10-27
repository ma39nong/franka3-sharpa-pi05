# Sharpa 全量模型 RTC

RTC 与串行部署统一保存在 `chunk-base-test`，工作目录 `/home/user/sharpa-pi05`，通过启动参数选择，不需要切换分支。部署前先将 `chunk-base-test` 的最新代码同步到工作机，继续使用工作机自己的 `.deployment/local.env`、`full_delta.env`、Home 和相机配置。

## 两个独立选项

- `--broker-mode serial`：默认值；保留每轮预测 50、执行前 15、停下来重新观测推理的流程。
- `--broker-mode rtc`：首次普通推理后持续播放已接纳的块；播放过程中观测并异步请求 guided 推理，接纳新块时跳过已过期节点。
- **RTC 现场命令统一使用 `--broker-mode rtc --speed fast`**：fast 表示 1.0 倍模型原始时间轴，不是额外加速；模型节点与设备下发均为 30 Hz。首次受控接近仍保留低速衔接。
- 非 RTC 串行模式仍可用 `--speed slow/medium/fast` 选择 0.1／0.5／1.0 倍速。速度与 broker 是两个参数，本页 RTC 命令均显式写出 fast。

RTC 下 `--execute-steps 15` 表示每接纳一块后，消费 15 个模型节点就请求下一块。等待结果时旧块继续播放，因此不是严格“只发前 15 帧”。有限 `--num-chunks` 计数接纳的预测块，必须至少为 2，或者设 0 持续执行；最后一块消费指定间隔后结束。RTC 不包含 RTG cubic 拼接，也没有复用 Wuji 的设备参数或取消现有硬件边界。

支持 full/full-delta、原生 58D、50 帧、JAX、sdk_direct。LoRA、legacy 后端和 policy-first 初始化暂不开放 RTC。current 与显式 Home 均可用。

## 同一模型服务，切换执行方式

在工作机部署仓库根目录操作。首先停止原普通模型服务，再启动允许 RTC 的服务：

```bash
bash deploy/fr3_sharpa_full/serve_delta.sh --enable-rtc
```

服务在监听前完成普通和 guided 预热，并宣告协议、58D 维度和预热状态。RTC 执行端会在创建 SDK worker 前核对能力；普通服务不能被误当作 RTC 服务。这个服务也接受不带 RTC 上下文的串行请求，不需要为两种执行方式切换 checkpoint 或重新训练。

只检查配置，不连接设备：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --check --broker-mode rtc --initialization current \
  --speed fast --execute-steps 15 --num-chunks 2
```

工作机 GPU 离线验证，假观测，不连接机器人：

```bash
bash deploy/fr3_sharpa_full/serve_delta.sh \
  --enable-rtc --smoke-only --report .deployment/rtc_smoke.json
```

以下命令是后续现场试验入口，会驱动机器人。现场试验尚未完成，不应将本次离线验证视为部署验收。硬件先按原流程启动：

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh up
```

非 RTC，保持原 slow 行为：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --execute --enable-hardware --broker-mode serial \
  --initialization current --speed slow --execute-steps 15 --num-chunks 2
```

RTC，按模型原速从当前状态执行两个预测块：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --execute --enable-hardware --broker-mode rtc \
  --initialization current --speed fast --execute-steps 15 --num-chunks 2
```

需要先回 Home、再按原速运行 RTC 时，使用：

```bash
bash deploy/fr3_sharpa_full/runtime_delta.sh \
  --execute --enable-hardware --broker-mode rtc \
  --home --speed fast --execute-steps 15 --num-chunks 2
```

确认有限轮数正常后，将 `--num-chunks 2` 改为 `--num-chunks 0` 持续执行，仍保留 `--speed fast`。单次试验结束或 Ctrl-C 后仍使用原失能确认和硬件清理；下一次先重新 `hardware.sh up`。模型服务保持独立运行。同一套机器人一次只启动一个执行端。

RTC 参数默认值：`--rtc-initial-delay-steps 5`、`--rtc-delay-buffer-size 10`、`--rtc-prefix-attention-schedule exp`、`--rtc-max-guidance-weight 5`。延迟历史以 5 个模型节点为种子，之后记录成功安装的实际跳步，取历史最大值作为下次模型条件。硬件 30 Hz deadline、新鲜反馈、关节范围、实际下发帧 slew、网关和停止/失能确认仍生效。旧块耗尽、回执不匹配、实际延迟超过引导窗口均停止，不无限保持旧块，也不推测新动作。

## Delta 与记录

前缀取自设备 worker 已接纳、经预测限位裁剪后的绝对轨迹，并按当前播放时刻采样；不直接拿未接纳的原始预测冒充实际旧前缀。每次 guided 请求将该 canonical 58D 前缀复制后送入现有数据变换：仅双臂减去新观测的实测状态，然后使用本 checkpoint 的统计归一化。输出反归一化后加回同一新观测状态，双手始终 absolute。

RTC 日志放在现有执行日志目录下独立的 `rtc-<id>/`：

- `requests.jsonl`：初始预测、guided 请求前缀与实测状态、响应回执及安装结果。
- `control.jsonl`：实际逐帧指令和反馈、源块编号、实际节点相位、限位修正及换块跳步。
- `run.json`：完成、失败或中断状态、已接纳块数和 RTC 参数。

RTC 不能保证任意场景的模型行为正确，也不增加环境碰撞规划。主线程不再对 guided 结果使用串行路径的“推理后观测年龄 <=200ms”条件；改用发送前的新鲜观测检查，加上 worker 的实际延迟/引导窗口验证和持续的新鲜反馈。首块普通推理仍保留原有时效检查。

## 来源与验证记录

参考 `harvesting/pi05` 的 `pi05-deploy` 分支提交 `7c3d36500624e7706eaa7ae7d82dbdb1537f45fe`：模型端 soft prefix guidance、策略前缀变换、请求/回执协议。来源文件为 `src/openpi/models/pi0.py`、`src/openpi/policies/policy.py`、`policy_config.py`、`src/openpi/serving/websocket_policy_server.py`、`packages/openpi-client/src/openpi_client/rtc.py`。保留 OpenPI 的 Apache-2.0 许可。协议维度已从 Wuji 54 改为 Sharpa 58；执行层是本项目独立实现，没有复制 Wuji 设备控制代码。

2026-10-27 已用昨天的 251-episode Delta step 4000 在 28 单 GPU 运行当前代码的普通和 guided 假观测预热，输出均为有限 `(50,58)`，guided 预热后模型采样约 72.8ms。权重实验名为 `tairan_tairan_tomato_sharpa_1016_2_multimodal_fullft_delta_s42_1024_1119_1024_113613`，没有改用新 311-episode 权重。该时间不包括相机、网络、安装和硬件控制，不能推定工作机实时性能。报告位于`.deployment/rtc_audit/rtc_smoke.json`，Git 忽略。

已覆盖非零新状态的 Delta 前缀重锚定、三档延迟换算、异步五块模拟、晚结果拒绝、回执与服务能力检查，以及原串行回归。ROS/SDK 真机时序、运动跟踪和任务效果仍需工作机测试。

本次离线回归共 283 项通过（`deploy/fr3_sharpa_runtime/tests`、客户端 RTC 协议测试、`pi0_test.py`），并完成运行层静态检查和示例 shell 语法检查。模型测试中的 JAX/Flax 弃用告警来自现有依赖，不影响本次通过结果。
