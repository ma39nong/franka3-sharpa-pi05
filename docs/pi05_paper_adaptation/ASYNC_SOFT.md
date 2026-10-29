# 异步 Soft：50 → 30 → 26＋4 → 20

本实验在 `soft-inpainting-test` 部署分支维护，训练仓库不参与这次改动。原串行、原 RTC、串行 Soft 的脚本、调度器、采样器及 worker 保持原样，main 不变。使用原 full/full-delta 权重和配套统计，无需重新训练；不兼容训练侧新论文归一化资产。

## 执行契约（rolling-v2）

- 模型每次预测50个58维绝对动作。立即舍弃索引30..49；它们不进入执行、参考、限位检查或轨迹摘要。
- 索引0..25共26点，经三次插值重采样为20个执行目标，保留首尾。索引26..29共4点只作下一段Soft参考，不单独追加执行。
- 新段开始后立即后台获取新观测并推理，当前段由独立worker继续播放。旧4点用新观测实测状态重建双臂Delta并归一化，约束新预测索引0..3，10步去噪在 `t_new>0.3` 时作用6次，后期释放。双手仍Absolute。
- **按轨迹阶段重定时**：新预测索引0对应下一段开头，参考为固定旧 `[26:30]`。不是“新预测0点对应观测时刻”的RTC时间线，因此不按推理耗时跳过新预测前缀。新计划提前完成只排队，必须等当前20目标的固定边界才激活，整个26点前缀始终压成20点。第一段没有Soft参考。
- 设备仍30Hz下发。执行目标推进率为 `30/ceil(1/speed_scale)`，slow/medium/fast分别3/15/30目标每秒；它们现在是**压缩后目标**，不再是源模型节点。相邻目标间插值。20个目标对应20个目标周期：前19个间隔播放到末目标，最后一个周期保持末目标，然后切到下一段；整段分别约6.67/1.33/0.667秒。源轨迹进度按目标0..19映射到源0..25，不能再按旧15源节点估算重规划时刻。
- 结果必须在固定边界之前到达worker并排队；不能提前切换，也不能用旧后20点或旧4参考延长播放。超时、过期观测、错误回执、服务重启、限位或衔接slew失败均停止并进入既有失能确认。Soft是软约束，不能保证衔接总通过。
- 采用严格26→20：压缩产生越界就失败，不静默退回26点。保留原预测手部纠偏上限；压缩目标另做严格限位检查，排队前检查旧段末目标到新段首目标的未来衔接，实际每个30Hz命令仍检查新鲜反馈和slew。
- `--num-chunks N` 表示完整播放N段，每段20目标；`0`表示持续运行。最后一段完整结束后停止，不再以15点阈值提前结束。

本设计是异步、固定裁剪压缩和Soft的组合实验，不叠加RTC的VJP引导、不使用相关性传播，也不声称完整复现论文。观测在上一段播放期间采集，新计划为下一阶段重定时，实机需评估接触阶段误差和工程机推理是否能赶上边界。

协议为 `sharpa-async-soft-rolling-v2`，旧异步服务与新执行端会互相拒绝，更新后须重启模型服务。默认full端口18865、full-delta端口18867。仅允许回环连接。日志为原日志根目录的 `async-soft-<ID>`，记录前30点、参考、回执、排队/激活边界及实际控制帧。配套归一化、模型及服务身份检查继续有效。

## 三个终端直接启动

先在工程机拉取 `soft-inpainting-test` 最新提交。三个终端都进入自己的部署仓库根目录。以下示例使用5999的full-delta；请保持权重目录与本机实际位置一致。

终端一：硬件依赖。

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh up
```

终端二：新异步Soft服务。

```bash
export SHARPA_FULL_DELTA_CHECKPOINT=/home/descfly/桌面/sharpa_1016_2_fullft_delta_30hz/5999
bash deploy/fr3_sharpa_full/serve_async_soft.sh --model full-delta
```

等待 `Async Soft Warmup OK: 50x58, 50->30->26+4->20, phase-aligned Soft, no RTC guidance` 和监听信息。

终端三：从当前实测姿态开始持续执行（会驱动机器人）。

```bash
export SHARPA_FULL_DELTA_CHECKPOINT=/home/descfly/桌面/sharpa_1016_2_fullft_delta_30hz/5999
bash deploy/fr3_sharpa_full/runtime_async_soft.sh --model full-delta \
  --execute --enable-hardware --sharpa-backend sdk_direct \
  --initialization current --prepare-speed-rad-s 0.03 \
  --speed slow --num-chunks 0
```

默认已固定26→20，无需另开压缩。`--execute-steps`只接受26，旧的15会明确报错；旧 `--soft-initial-delay-steps` / `--soft-delay-buffer-size`、`--broker-mode`、`--compression` 参数不属于此入口。若要有限段执行，将 `--num-chunks 0` 改为3。full模型两端一起改为 `--model full`。Home仍由原配置读取双臂与双手Home。

停止时先在终端三按一次Ctrl+C，等待双手失能读回确认和硬件清理，再停止模型服务。再次执行前按原流程 `hardware.sh up`。

## 可选离线检查与验证范围

下列检查不连接设备；不是新增的启动前交互确认。

```bash
bash deploy/fr3_sharpa_full/runtime_async_soft.sh --model full-delta --check --speed slow --num-chunks 3
bash deploy/fr3_sharpa_full/serve_async_soft.sh --model full-delta --check-only
# 加载真实GPU权重并假观测推理，不连接设备：
bash deploy/fr3_sharpa_full/serve_async_soft.sh --model full-delta --smoke-only
```

`--check`输出应包含 `execute_steps=26`、`effective_horizon=30`、`execution_target_count=20`、`soft_reference_count=4`、`compression=cubic`、`alignment=trajectory_phase`、`replan_trigger=segment_start`。服务实际启动自带预热。

实现仅位于 `async_soft_*.py`；timeline已独立实现，不继承RTC的50点时间线。模型保持原50×58采样，仅独立服务实例绑定Soft采样器，未改原Pi0/Policy。硬件worker继续使用原SDK、网关、反馈和退出失能确认，但单独实现裁剪、严格压缩、结果排队及边界切换。session继续沿用原硬件归属管理。

本轮本地CPU回归382项通过（其中异步专项50项）。离线测试覆盖：三档速度的完整跨段播放、压缩端点和时间映射、固定4点参考、舍弃尾部不影响输出、提前排队而不切换、迟到/过期/错误身份拒绝、真实JIT Soft采样与Delta转换、真实msgpack假socket、四段后台协调，以及真实worker循环搭配模拟ROS/SDK的边界切换、压缩越界/衔接失败、失能读回确认。运行：

```bash
PYTHONPATH=src:packages/openpi-client/src:. \
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m pytest deploy/fr3_sharpa_runtime/tests \
  src/openpi/models/pi0_test.py packages/openpi-client/src/openpi_client/rtc_test.py -q
```

未在本轮加载工程机5999真实权重或驱动硬件；离线通过不代表跟踪、接触或采摘效果已验证。`COMPLETE` / `commands_sent`只表示命令链路完成，不表示关节全部到位或任务成功。
