# Ctrl+C、双手失能与录制

2026-10-27，使用 `runtime_delta.sh ... 2>&1 | tee ...` 的现场试验在 Ctrl+C 后没有失能确认，用户反馈仍有主动夹持力。现有日志不能完整还原退出故障；代码检查确认日志输出异常能在发送失能前中断原退出流程。

修复只作用于之后新启动的进程，不能解除已经退出进程留下的夹持力。当前设备未确认失能时，先使用设备已有硬件失能／急停流程。不要靠重新启动策略卸力，不要强掰手指。

## 退出保证的边界

- 失能提示、信号提示写入失败不再打断双手 stop/读回；一只手失败仍继续尝试另一只。
- 停止 SDK 反馈线程后，优先尝试双手失能，再做 ROS 和轨迹日志收尾。
- 双手仍采用已有失能 API，读回 `enabled=False` 才算确认。不重新连接、不发送张手／回 Home 目标、不放宽重试上限。
- SDK worker 在启动时写入 `PENDING`，结束时独立保存 `shutdown-<pid>.json` 到本轮 `--execution-trace-dir`。文件先写临时文件、flush/fsync 后替换，不依赖 stdout、stderr 或 IPC 管道。
- `PASS` 还需检查左右手 `confirmed_disabled=true`。`PENDING`、`FAIL`、`NOT_CONNECTED`、缺少文件均不能当作双手已失能。记录写盘失败也不能据此认定失能。
- 父会话对重复 Ctrl+C 不重复转发中断；容器清理输出不再继承已断开的控制台管道。
- 程序失能读回不是机械张手或零接触力证明。SDK 阻塞、断电、强杀进程等情况仍可能无法确认失能。

## 录制执行端：替代 tee

硬件启动并稳定后，在终端一用一条命令录制原有三路图像、双臂双手反馈和双臂命令：

```bash
bash deploy/fr3_sharpa_full/record_rosbag.sh 5999_rtc_02
```

默认保存在 `~/桌面/sharpa_deploy_records/5999_rtc_02/rosbag`，也可传完整绝对目录。该脚本只录制、不启动硬件；已有 bag 时拒绝覆盖。`--check 5999_rtc_02` 只检查路径和配置。终端三必须使用同一个试验目录。

三路相机和关节 ROS bag 仍按原录制命令启动。模型服务仍需先加载 5999 并使用 `--enable-rtc`。执行端改用：

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
export SHARPA_FULL_DELTA_CHECKPOINT=/home/descfly/桌面/sharpa_1016_2_fullft_delta_30hz/5999

bash deploy/fr3_sharpa_full/record_runtime_delta.sh \
  /home/descfly/桌面/sharpa_deploy_records/5999_rtc_02 \
  --execute --enable-hardware --home \
  --broker-mode rtc --speed fast \
  --prepare-speed-rad-s 0.03 --execute-steps 15 --num-chunks 0
```

必须用新的录制目录；已有 `runtime_console.log` 会拒绝覆盖。程序输出直接写普通文件，屏幕显示只是独立镜像，显示端断开不会关闭执行端的日志文件。Ctrl+C 只转发一次，并等待原执行端的有时限失能和硬件清理流程。不要再外接 `| tee`，不要直接关闭窗口。

退出后查看本轮记录：

```bash
ls -lt /home/descfly/桌面/sharpa_deploy_records/5999_rtc_02/execution_traces/shutdown-*.json
cat /home/descfly/桌面/sharpa_deploy_records/5999_rtc_02/execution_traces/shutdown-*.json
```

本轮所有 SDK worker 的报告都要核对，不能拿之前试验的 PASS 作为本轮确认。正常终端退出码不替代失能读回报告。

## 验证

离线测试覆盖真实关闭的 stderr 管道、日志回调抛错、双手失能读回、失败不冒充 PASS、录制镜像管道断开后 Ctrl+C 的日志完整性，以及旧录制不被覆盖。假设备不连接机器人；离线通过不等于现场卸力已验收。
