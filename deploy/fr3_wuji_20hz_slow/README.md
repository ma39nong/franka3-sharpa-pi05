# 20 Hz 权重慢速部署

这是 `tomato_lora_0918_20hz/19999` 的独立慢速入口。模型每次仍输出
`50×54`，但轨迹结点按训练时的 **20 Hz** 时间轴规划，再应用默认的
**2.5 倍**慢放和原有速度、加速度、jerk、关节范围及设备停止检查。
现有 `fr3_wuji_slow` 的 30 Hz 配置不变。

只检查文件和启动配置，不连接模型或设备：

```bash
cd /home/user/lpy/Pi05
bash deploy/fr3_wuji_20hz_slow/run.sh --check
```

终端 1 启动模型服务：

```bash
bash deploy/fr3_wuji_20hz_slow/serve.sh
```

终端 2 慢速执行 100 轮，每轮使用完整 50 步：

```bash
bash deploy/fr3_wuji_20hz_slow/run.sh \
  --execute --supervised-trial \
  --continuous --rounds 100 --replan-steps 50 \
  --start-cameras --finish-policy disable
```

默认 checkpoint 为：

```text
/home/user/lpy/Pi05/checkpoints/tomato_lora_0918_20hz/19999
```

执行端会验证 checkpoint 的唯一归一化文件确实位于
`assets/fr3_wuji/0918_20hz/norm_stats.json`。因此误传常规 30 Hz 权重会在
启动设备前失败。模型端与执行端仍会核对 checkpoint manifest 和归一化哈希。

完整 50 步的原始跨度为 `49/20 = 2.45` 秒；默认 2.5 倍慢放后的播放阶段
至少为 6.125 秒。速度、加速度或 jerk 约束可能继续延长它。初始接近和停稳
时间另计。执行前必须退出遥操或其他设备拥有者；已有相机发布时省略
`--start-cameras`。
