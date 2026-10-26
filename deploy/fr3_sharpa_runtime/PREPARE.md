# 当前状态启动与可选 Prepare

SDK direct 默认 `--initialization current`：采集当前双臂、双手实测关节状态和三路图像，推理 50 帧，低速衔接到首帧并执行前 15 帧，再重新观测推理。启动时不读取固定 Home，不先发送 Home 或独立首帧 prepare。

```bash
bash deploy/fr3_sharpa_runtime/runtime.sh --execute --enable-hardware \
  --prepare-speed-rad-s 0.03 --speed slow --num-chunks 0
```

需要先回固定位置时，在同一执行命令末尾添加 `--home`。它等效于 `--initialization home`：双臂和双手一起低速到达固定 Home，停稳后重新观测并继续任务。Home 配置和到位判据见 [README](README.md)。

显式 `--initialization policy` 或旧 `--prepare-policy-first` 保留模型首帧 prepare：先推理出首帧目标，低速到位，再重新观测推理执行。该模式与 `--home`、显式 `--initialization current` 互斥。

Ctrl-C 后仍执行双手失能确认和硬件清理，不自动回位。恢复任务时先 `bash deploy/fr3_sharpa_runtime/hardware.sh up`，再使用不带 Home／policy prepare 选项的执行命令，重新采集当时的状态与图像。失能期间是否仍握住物体由设备实际状态决定。

三种启动模式都保留动作范围、slew、反馈时效、watchdog 和 FR3 网关检查。首帧衔接和可选 prepare 都是关节空间插值。模型动作保持 absolute，不重复叠加当前状态。`--execute-steps` 默认 15，`--num-chunks 0` 持续到 Ctrl-C。
