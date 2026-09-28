# Pi05 独立 tomato Home

这个入口只负责把双臂、双手移动到 Pi05 自己保存的 `tomato` Home。位姿在
`poses.yaml` 中，初始值复制自 2026-09-28 遥操配置的 `assembly` Home；运行时不再读取或修改
遥操的 Home 数据，也不启动遥操。底层仍复用 Pi05 已有且经过哈希校验的控制器、安全网关和设备边界。

归位严格分两阶段：双臂以最高 0.20 rad/s 移动并由实测反馈确认稳定，然后双手以
0.50 rad/s 的目标时间线移动并确认到位。完成后禁用双手、停止本次机械臂命令并退出最小控制栈。

只检查配置和启动命令，不连接设备：

```bash
cd /home/user/lpy/Pi05
bash deploy/fr3_wuji_home/run.sh --check
```

现场有人值守时执行真实归位：

```bash
cd /home/user/lpy/Pi05
bash deploy/fr3_wuji_home/home.sh
```

该命令不启动模型、相机、GELLO、MANUS 或 UI。遥操、其他 Pi05 部署或任何手部 SDK/FCI
拥有者正在运行时会在运动前拒绝启动；不要并行运行。每次结果保存在
`logs/weight_motion_eval/home-*`，包含目标、规划、实测反馈、设备身份和停止报告。

以后增加 Home 时，在 `poses.yaml` 的 `homes` 下加入一个新名字，并用 `--home NAME` 选择；
不需要恢复遥操中的 `--task` 概念。
