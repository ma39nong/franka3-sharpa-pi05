# cam1 源时间戳异常：现场排查结果

2026-10-27，右腕相机 `CV2756100018` 保持约 30 FPS 传图，但设备时间多次复位、global 时间出现倒退和未来时间，导致部署的 200 ms 新鲜度、100 ms 同步检查拒绝观测。相机单独运行即可复现，不依赖 RTC 或模型。用户确认只有 USB，没有外部同步／触发线。

## 对照结果与限制

测试没有改变分辨率、YUYV 格式、帧率或观测阈值：

- 只关闭主机校时，保留 global 时间：60 秒内 cam1 倒退 270 次，1322 帧超龄，未解决。
- 恢复原配置，通过 SDK 关闭 timestamp reset 后重启相机：60 秒内 cam1 1801 帧、零倒退、零超龄。
- 重新开启 timestamp reset 后重启：45 秒内也正常。
- 再次通过 SDK 关闭该选项、使用原相机配置启动：120 秒 cam1 3603 帧、30.02 FPS、零倒退、零超龄，源时间年龄 25.34–27.92 ms。cam0/cam2 也零倒退、零超龄。
- **停止相机并重新打开 SDK 后，该选项读回为 true。不能证明关闭设置跨设备会话保持有效。** 因此撤回试验中的自动启动逻辑，未启用相机时钟自动改写。

这些结果说明当前传图和时间戳恢复，不能证明 timestamp reset 开关是唯一根因，也不能区分配置切换、设备重开及内部状态恢复的作用。暂不据此更换相机或刷固件。未改硬件仓库、未改 header.stamp、未放宽部署检查；试验没有控制机械臂或灵巧手。

最后不再调用 SDK 设置、直接按原配置重启的 60 秒复查：cam1 1801 帧、30.02 FPS、零倒退、零超龄，年龄 25.40–28.19 ms；cam2 也正常。cam0 启动初期出现 12 帧超龄和约 3.73 秒接收间隔，之后超龄计数未再增加。这次不满足三路全程通过，不能直接视作整套部署验收。报告为 `original_restart_report.json`。

## 只读复查

`camera_clock_probe.py` 只订阅三路 Image，预热 10 秒后统计帧率、源时间年龄、倒退及接收间隔，不发布控制命令、不保存图像。需要先启动原来的相机服务。在仓库根目录运行：

```bash
source deploy/fr3_sharpa_runtime/env.sh
export ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI="file://$SHARPA_GELLO_REPO/config/cyclonedds.xml"
pixi run --locked --manifest-path "$SHARPA_GELLO_REPO/portable_deps/litchi_hardware/pixi.toml" -e ros2 \
  python "$SHARPA_ROOT/deploy/fr3_sharpa_runtime/camera_clock_probe.py" \
  --duration 120 --output "$SHARPA_ROOT/.deployment/camera_clock_report.json"
```

每路应有正常帧率，`source_backwards` 和 `bad_age_frames` 为 0。此测试只验证各路相机，不能替代包含手臂／手状态和 100 ms 跨模态同步的完整观测检查。

## 继续现场验证

先使用原相机配置重启并复查时间戳，再做有限两块的 serial/slow，然后有限两块 RTC。通过后才延长连续执行。若再次出现时间倒退，先停止动作，保留当次报告及相机驱动日志；停止相机服务后再进行相机 USB 断电重连、线材／端口交换对照，观察异常是否跟随同一序列号。不要在控制运行中拔线。

`.deployment/camera_clock_trials/` 保留实验程序、CSV 和报告，不进入 Git：`no_host_sync_report.json`、`no_reset_report.json`、`reset_reenabled_report.json`、`guard_original_profile_report.json`。其中 guard 文件名仅标识当次实验，不能解释为生产保护已生效。

相机测试正常不代表异常退出时重复 rclpy shutdown 等清理问题已经修复，也不代表完整机器人部署通过验收。
