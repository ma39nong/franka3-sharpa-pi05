# 25000 独立模型服务与中速入口

本目录只适配 `/home/user/lpy/Pi05/checkpoints/25000`，服务默认监听 8004。
它与 19999（8001）、30000（8002）、30000v2（8003）并列；不改这些模型的服务、权重或入口。

25000 是 64 维全量微调权重，末尾 10 维为补零。与 30000 不同，它的**状态输入**
已经按机器人顺序排列：左臂 7、左手 20、右臂 7、右手 20；服务仅补 10 个零。
**动作输出**仍按左臂 7、右臂 7、左手 20、右手 20、补零 10 排列；服务反归一化后
交换右臂与左手，丢弃末 10 维，得到 50×54 的绝对关节位置。
服务对权重、统计文件、参数树、输入输出顺序和适配器版本建立独立契约；中速入口会逐项核对服务元数据。

统计目录名 `tomato_AB_15hz` 是目前可见的动作频率线索。25000 中速规划据此按 15 Hz
解释预测节点，仍以 100 Hz 下发。训练配置没有随 checkpoint 保存，所以 15 Hz
尚缺独立训练记录佐证；在真机使用前应核实这一点。其余三个模型保持原 30 Hz 时间轴。
连续执行沿用中速入口的约 25 秒后、在下一段推理边界切慢速轨迹的逻辑；切换后
仍使用 25000 的 15 Hz 原始节点时间轴。

只核查权重、统计与接口，不连接模型服务或设备：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji_25000/run.sh --check-only
bash /home/user/lpy/Pi05/deploy/fr3_wuji_25000/medium.sh --check --continuous
```

用假观测加载权重、验证动作输出，不连接机器人：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji_25000/run.sh --smoke-only
```

确认模型服务验证通过后，终端一启动 25000 服务：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji_25000/run.sh
```

服务显示 `Warmup OK: actions=(50, 54)` 和 `Serving 25000 at ws://127.0.0.1:8004`
后，终端二运行中速执行入口：

```bash
bash /home/user/lpy/Pi05/deploy/fr3_wuji_25000/medium.sh \
  --execute --supervised-trial \
  --continuous --rounds 50 --replan-steps 20 \
  --start-cameras --finish-policy hold
```

第二条命令会连接并驱动真实机器人。此前 8daf43f594 运行中，右手 NID=8 曾触发
`4099` 温度超限；应确认固件故障已清除、温度恢复且关节无卡阻后再执行。
如果要全程中速，可在第二条命令末尾加 `--slow-after-seconds 0`。

离线测试：

```bash
cd /home/user/lpy/Pi05
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m pytest deploy/fr3_wuji_25000 deploy/fr3_wuji_medium -q -p no:cacheprovider
```
