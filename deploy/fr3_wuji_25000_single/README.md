# 25000-single 并列部署

本目录使用 `/home/user/lpy/Pi05/checkpoints/25000-single`、自带的
`assets/tomato_A_15hz/norm_stats.json` 和独立模型服务端口 8005。
它与 25000 的 `tomato_AB_15hz` 权重分开校验，不会交换归一化统计或复用服务身份。

两份权重都为 64 维全量微调、无 LoRA 参数；状态按左臂、左手、右臂、右手排列，
只需在末尾补 10 维。动作按左臂、右臂、左手、右手、补零 10 维排列，
反归一化后重排并去掉补零，得到机器人接口的 50×54 绝对关节位置。
中速规划仅在选择 `25000-single` 时使用该模型契约与 15 Hz 节点时间轴，
以 100 Hz 向设备下发。15 Hz 依据归一化资产目录名，尚无随 checkpoint
保存的训练配置来独立证明；真机试用前应核实。

`25000-single` 的中速执行现在仅将**左手 20 个关节的 Kp 设为 5**；右手仍为 8，
两侧 Kd 均为 0.1，原电流与速度限制不变。配置在手部使能前写入并读回核对；
运行报告的 `hand-operating-parameters.json` 记录两侧实际读回值。

离线核查和假观测推理都不连接机器人：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji_25000_single/run.sh --check-only
bash /home/user/lpy/Pi05/deploy/fr3_wuji_25000_single/medium.sh --check --continuous --slow-after-seconds 0
bash /home/user/lpy/Pi05/deploy/fr3_wuji_25000_single/run.sh --smoke-only
```

真机试用时，终端一启动模型服务：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji_25000_single/run.sh
```

服务显示 `Warmup OK: actions=(50, 54)` 后，在终端二先执行一轮全程中速：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji_25000_single/medium.sh \
  --execute --supervised-trial \
  --continuous --rounds 1 --replan-steps 20 \
  --slow-after-seconds 0 \
  --start-cameras --finish-policy hold
```

第二条命令会驱动真实机器人。此前右手 NID=8 曾报告 `4099` 温度超限；
重试前需确认温度和固件故障已恢复。确认单轮表现后，可提高 `--rounds`。
如果不传 `--slow-after-seconds 0`，中速入口默认约 25 秒后
在下一段推理轨迹边界切为慢速。
