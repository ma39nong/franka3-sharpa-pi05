# FR3 + Wuji 中速部署（再次提速版）

中速现已在上一版基础上再提高一倍。默认 slider 模式下，同一条双臂播放轨迹的
时长是上一版中速的一半、原默认慢速的四分之一。接近阶段也同步提速。
慢速、原有快速入口和它们的代码未修改。

连续模式现在默认在实际动作开始约 25 秒后切换到原慢速轨迹参数。
切换只发生在下一段推理轨迹的边界；当前段会完整执行，双臂和双手仍使用同一条时间轴。
如果 25 秒落在当前段内部，控制进程会在段末选择已提前规划的慢速版本，
并将实际切换轮次、时间和计划哈希写入运行事件与 `continuous-report.json`。
切换后保持慢速，直到本次运行结束。慢速轨迹使用原慢速的 2.5 最低时间倍率、
双臂及双手规划约束、接近阶段和 0.5 秒稳定等待；slider 手的 SDK 逐帧限速同时降至 1 rad/s。
设备会话仍由中速入口持有，100 Hz 下发、20 ms 帧期限和现有停止检查不变。

要回退到原来的**全程中速**，在原命令末尾加 `--slow-after-seconds 0`。
只想调整切换时间，可指定其他正数，例如 `--slow-after-seconds 30`。
单次非连续执行没有下一段边界，保持原中速。

机械臂跟踪误差阈值已统一设为 **0.2 rad**（播放器、设备发送端及轮间预取条件），
并写入每次运行的 `runtime.json` 的 `arm_tracking_rad` 字段。

## 本次参数变化

| 参数 | 上一版中速 | 当前中速 |
| --- | ---: | ---: |
| 默认 minimum-time-scale | 1.25 | 0.625 |
| 双臂播放速度上限 rad/s | 0.84 | 1.68 |
| 双臂播放加速度上限 rad/s² | 5.76 | 23.04 |
| 双臂播放 jerk 上限 rad/s³ | 138.24 | 1105.92 |
| 双臂接近速度上限 rad/s | 0.18 | 0.36 |
| 双臂接近加速度上限 rad/s² | 0.432 | 1.728 |
| 双臂接近 jerk 上限 rad/s³ | 3.456 | 27.648 |
| 设备反馈、命令、网关速度上限 rad/s | 1.0 | 2.0 |
| slider 双手 SDK 位置变化率上限 rad/s | 1.0 | 2.0 |
| strict 双手速度上限 | 45°/s | 90°/s |
| 首次接近最短时间 s | 1.0 | 0.5 |
| 轮间接近最短时间 s | 0.05 | 0.025 |

strict 手部规划的加速度、jerk 上限也分别乘以 4、8。
时间缩短一半时，同一路径的速度、加速度、jerk 分别乘以 2、4、8。
slider 接近阶段的手部距离耗时按距离 / 2 rad/s 计算。

100 Hz 下发、20 ms 控制帧期限、跟踪检查、关节位置边界、电流、接触保护和固件故障处理保留。
中速连续执行的单次周期超时容差从 2 ms 提高到 5 ms：只在提交已确认、反馈健康且帧未过期时
记录警告并调整后续调度，不补发追赶。连续 3 次超时或 1 秒内超过 5 次超时仍停止。
实际帧过期或丢失反馈仍停止；此容差不是将帧有效期延长到 25 ms。
轮间继续采用到位后预取下一轮推理，与 0.15 秒稳定等待重叠；最终稳定等待为 0.5 秒。
仍在段末减速到零，不是非零速度跨段拼接。推理耗时、稳定等待和实际跟踪不按比例缩短，
所以整轮墙钟耗时不保证减半。

## 用户当前 checkpoint 的启动命令

终端一，启动推理服务（如果已经使用此 checkpoint 在 8001 服务，无需重启模型）：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji/run.sh \
  -m experiments.weight_motion_eval.oneshot.policy_server \
  --checkpoint /home/user/lpy/Pi05/checkpoints/pi05_fr3_wuji_weighted/tomato_lora_0918_a70_b30/19999 \
  --port 8001
```

等服务提示 ready 后，在终端二启动中速：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji_medium/run.sh \
  --model 54 \
  --checkpoint /home/user/lpy/Pi05/checkpoints/pi05_fr3_wuji_weighted/tomato_lora_0918_a70_b30/19999 \
  --uri ws://127.0.0.1:8001 \
  --execute --supervised-trial \
  --continuous --rounds 50 --replan-steps 20 \
  --minimum-time-scale 0.625 \
  --slow-after-seconds 25 \
  --start-cameras --finish-policy hold
```

**旧命令中的 `--minimum-time-scale 1.25` 要改为 `0.625`，或删除该参数使用新默认值。**
已经运行的中速进程不会热更新，退出后重新启动才使用新实现。
本次更新未启动、停止或接管真实机器人。

只生成配置并检查文件，不连接模型或设备：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji_medium/run.sh \
  --model 54 \
  --checkpoint /home/user/lpy/Pi05/checkpoints/pi05_fr3_wuji_weighted/tomato_lora_0918_a70_b30/19999 \
  --check --continuous
```

64 维适配现在使用明确的名称：

| `--model` | 原名称 | 默认端口 |
| --- | --- | --- |
| `64-lora-30hz` | `30000` / `64` | 8002 |
| `64-full-30hz` | `30000v2` | 8003 |
| `64-full-15hz-ab` | `25000` | 8004 |
| `64-full-15hz-a` | `25000-single` | 8005 |

25000 有单独的全量微调服务、状态输入顺序和 15 Hz 节点时间轴，见
`/home/user/lpy/Pi05/deploy/fr3_wuji_25000/README.md`。
25000-single 使用相同的输入输出映射，但有自己的 `tomato_A_15hz` 统计和服务身份，见
`/home/user/lpy/Pi05/deploy/fr3_wuji_25000_single/README.md`。
默认模型为 `64-lora-30hz`（原 30000 适配），因此以上 54 维 checkpoint 的命令需保留 `--model 54`。
`54` 表示原生 54 维、30 Hz 适配；原生 54 维、20 Hz 仍用 `--model 20hz`。
所有旧名称继续兼容，其中 `64` 仅代表 `64-lora-30hz`，不代表全部 64 维模型。
权重路径、服务契约、输入输出映射、手部参数和安全阈值不变。
不传 `--continuous` 时执行单次 50 步预测。
默认日志目录为 `/home/user/lpy/Pi05/logs/weight_motion_eval/medium-*`。
默认结束后保持，Ctrl-C 停止并清理；`--finish-policy disable` 可在末端到位后释放。

## 隔离实现

规划器、播放器、连续调度、配置和入口均在此目录。
本次新增 `runtime/`，独立维护速度相关的反馈/命令检查、IPC、设备桥、手部 SDK 发送、
ROS 网关及执行资格参数，避免旧边界的 1 rad/s、1.2 rad/s 检查拦截新配置。
没有 monkey-patch 原模块，也没有改动原工作站配置或控制器二进制。
观测、模型服务、诊断、时间期限编码和接触处理等不涉及本次速度调整的工具仍通过接口只读复用。
三种部署共用设备占用锁，不能同时占用机器人。

## 离线验证

```bash
cd /home/user/lpy/Pi05
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest deploy/fr3_wuji_medium \
  experiments/weight_motion_eval/oneshot/test_continuous.py \
  deploy/fr3_wuji_fast/test_arm_speed.py -q -p no:cacheprovider
```

99 项测试通过，覆盖提速后的实际 SDK 发送限幅、网关数学检查、反馈及命令限幅、IPC 模式匹配、
跨进程两段执行、预取、异常停止，以及原慢速连续执行和原快速速度参数回归。
新增检查覆盖切换点落在预取与真正段边界之间、已规划轨迹的选择、跨进程切换、
slider 手限速切换，以及禁用切换后的原中速路径。
10 段历史记录重新规划后，接近及播放时间均是上一版中速的一半。
用户提供的 checkpoint 配置检查通过。离线验证不代表完成了真实机器人跟踪验证。
