# 异步 Soft Inpainting：独立第四种实验模式

本实验只在 `soft-inpainting-test` 分支新增文件。原串行、原异步 RTC、串行 Soft 的启动脚本、调度器、采样器及硬件 worker 不修改，main 保持原提交。已有权重、归一化统计、设备配置继续使用，不需要重新训练。

这里复用 RTC 的异步执行思路，用 Soft 作为独立去噪引导；**不是把原 RTC 的 VJP 前缀引导与 Soft 同时叠加**。不使用相关性传播，不使用三次样条压缩，不声称完整复现论文。

## 四种方式

| 方式 | 服务端 | 执行端 | 状态 |
| --- | --- | --- | --- |
| 原串行 | 原 `serve.sh` / `serve_delta.sh` | 原 `runtime.sh` / `runtime_delta.sh` | 保持原样，默认50/15 |
| 原 RTC | 原服务加 `--enable-rtc` | 原执行端加 `--broker-mode rtc` 和原参数 | 保持原样 |
| 串行 Soft | `serve_soft.sh` | `runtime_soft.sh` | 保持原样，默认26+4、无压缩 |
| 新异步 Soft | `serve_async_soft.sh` | `runtime_async_soft.sh` | 独立入口，默认50步、15步后重规划 |

上述脚本均位于 `deploy/fr3_sharpa_full/`。新入口两端使用相同的 `--model full` 或 `--model full-delta`。切换前先停止旧执行端和模型服务，硬件准备与退出失能确认沿用原流程。不能同时启动两套机器人执行进程。

新服务默认使用原模型端口加100：full 为18865、full-delta 为18867，与旧服务端口分开。修改服务端 `--port` 时，执行端同时指定相同的 `--policy-url ws://127.0.0.1:端口`。只支持本机回环连接，不支持跨主机比较单调时钟。默认日志根目录沿用所选模型，但新增 `async-soft-<ID>` 子目录。

## 时间与动作契约

1. 第一段从当前观测推理，不带旧参考；初次结果仍检查200ms观测年龄。进入流之前保留原低速首帧衔接。
2. worker 持续以30Hz下发旧轨迹。消费15个源模型节点后，协调器在后台获取新观测和推理，worker 不等待模型结果。
3. 新预测第0点始终对应**观测采集时刻**。worker 根据该时刻在旧轨迹中的进度，补计采集到快照的时间，选择预计接入处连续4个旧轨迹点。不是固定旧 `[26:30]`，也不是把快照时刻当作观测时刻。
4. 新采样器仅在这4个未来位置进行 `t_new>0.3` 的 Soft 约束；默认10步去噪作用6次，后期释放。旧绝对参考用**这次观测的实测状态**重建双臂Delta，再用所选权重的统计归一化。手部保持Absolute，输出仍是58D绝对关节目标。
5. 结果提前到达时继续播放旧轨迹，等待计划衔接时刻。允许在这4个参考节点覆盖的时间区间接入；超出窗口或旧轨迹耗尽即失败，不能靠重新启用旧缓存继续。
6. 接入时使用从观测到实际安装时刻的**小数源节点进度**跳过过期预测，不向上取整造成额外时间跳跃。已发送动作不回改，实际衔接继续执行原来的限位、反馈新鲜度、slew检查和SDK/网关保护。Soft允许偏离参考，不保证衔接必然通过。
7. slow/medium/fast 复用原 RTC 的播放时间定义：`node_hz=30/ceil(1/speed_scale)`，分别3/15/30源节点每秒；设备下发仍30Hz。这是沿用现有播放约定的迁移实验，真实跟踪和纠偏质量待验证。

`--soft-initial-delay-steps` 默认5，允许1到20个源节点；`--soft-delay-buffer-size` 默认10，使用近期推理延迟的最大值。4点衔接窗口宽度为3个源节点周期；延迟超出窗口会停止。观测采集也受旧轨迹剩余时长约束，仍保留200ms新鲜度门限。不要靠放宽现有安全阈值让测试通过。

## 工程机命令

先拉到本分支最新提交，并确认工程机已有该模型的权重和机器配置。在仓库根目录执行。

只检查参数，不加载模型、不连接设备：

```bash
bash deploy/fr3_sharpa_full/runtime_async_soft.sh \
  --model full-delta --check --speed slow --num-chunks 3
```

模型终端先检查资产，再做实际GPU假观测smoke；两条都不连接设备，但后者加载真实权重：

```bash
bash deploy/fr3_sharpa_full/serve_async_soft.sh --model full-delta --check-only
bash deploy/fr3_sharpa_full/serve_async_soft.sh --model full-delta --smoke-only
```

确认smoke后启动独立服务，等待 `Async Soft Warmup OK`：

```bash
bash deploy/fr3_sharpa_full/serve_async_soft.sh --model full-delta
```

硬件按原流程启动。以下会连接设备并发送运动，第一轮检查执行和退出链路；首轮不含跨轮Soft：

```bash
bash deploy/fr3_sharpa_full/runtime_async_soft.sh --model full-delta \
  --execute --enable-hardware --sharpa-backend sdk_direct \
  --initialization current --speed slow --num-chunks 1
```

确认首轮后，将 `--num-chunks 1` 改为 `3` 才能验证异步更新；不要首次直接无限循环。Absolute模型两端都改为 `--model full`。Home可显式选择 `--initialization home`，沿用原 `.deployment` 中的Home配置。旧的 `--broker-mode`、`--inpainting`、`--compression` 参数不属于这个新入口，误传会拒绝。

## 实现边界和验证

- `async_soft_stream.py`：独立协议、观测时间对齐、窗口检查，继承原 timeline 的只读播放数学。
- `async_soft_sampler.py` / `async_soft_policy.py`：独立采样与转换；仅在新服务实例绑定，不替换原 `Pi0.sample_actions` 或原 Policy。
- `async_soft_client.py` / `async_soft_serve.py`：独立能力握手、模型/统计/服务身份与回执检查，拒绝RTC与串行Soft混用。
- `async_soft_runtime.py` / `async_soft_run.py`：独立后台推理与执行协调，不改原调度器。
- `async_soft_motion_ros.py`：从 `sdk_direct_motion_ros.py@246cccc` 冻结复制，仅替换timeline与快照时间参数；保留原SDK、网关、反馈与退出失能确认。专门测试会检查该边界，后续升级原worker时需要单独评估迁移。
- `async_soft_session.py`：调用原硬件归属管理与退出清理，不改它们。

本地CPU完整回归通过365项，其中新增33项。新增测试覆盖三档时间映射、延迟位置Soft及后期释放、模型默认采样等价性、真实转换/独立policy、真实msgpack假socket、四轮协调、提前等待、迟到/耗尽/过期/中断拒绝、真实新worker循环配合完全模拟的ROS/SDK以及部分失败时的失能确认。它们不是实际Gemma权重加载、动力学仿真或实机验证。

复现：

```bash
PYTHONPATH=src:packages/openpi-client/src:. \
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m pytest deploy/fr3_sharpa_runtime/tests \
  src/openpi/models/pi0_test.py packages/openpi-client/src/openpi_client/rtc_test.py -q
```

真实GPU预热、真实留出观测回放、工程机时延、双臂/手部接触阶段、跨轮跟踪误差及采摘表现均待验证。`COMPLETE` 和 `commands_sent` 仅表示命令链路完成，不表示关节全部到位或任务成功。
