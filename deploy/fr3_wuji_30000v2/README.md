# 30000v2 full 微调慢速部署

本目录为 `checkpoints/30000v2` 的独立入口。模型内部为64维，真机接口为54维；状态/动作映射、绝对关节位置语义和末10维丢弃规则沿用已验证的30000适配。参数树没有LoRA分支，使用 `gemma_2b` + `gemma_300m` 原始参数结构加载 full 微调权重。

默认模型服务为 `ws://127.0.0.1:8003`，不会占用30000的8002。慢速执行复用当前30000专用规划配置：双臂首帧接近上限1.5rad，机械臂平滑偏差上限0.2rad；旧部署文件、权重、设备IP、硬限位、速度、接触与超时检查不变。

检查、假观测加载和启动服务：

```bash
bash deploy/fr3_wuji_30000v2/run.sh --check-only
bash deploy/fr3_wuji_30000v2/run.sh --smoke-only
bash deploy/fr3_wuji_30000v2/run.sh
```

服务显示 `Warmup OK: actions=(50, 54)` 后，在另一终端启动慢速真机执行：

```bash
bash deploy/fr3_wuji_30000v2/execute.sh \
  --execute --supervised-trial --hand-control slider \
  --continuous --replan-steps 30 --rounds 50 \
  --minimum-time-scale 2.5 \
  --start-cameras --finish-policy disable
```

`execute.sh` 已默认选择30000v2与8003，不需要重复传入。执行命令会启动真实硬件；模型服务和 `--check-only`/`--smoke-only` 不连接机器人。

已验证：参数树无LoRA分支；64维 state/actions 统计与30000逐项一致且末10维全零；7项CPU适配回归通过；GPU真实加载与假观测推理通过，输出 `Warmup OK: actions=(50, 54)`；慢速执行入口的 `--check` 通过，设备地址仍为左/右臂 `172.16.0.2` / `172.16.1.2`、左/右手 `192.168.1.110:7447` / `192.168.2.111:7447`。未运行真机动作。
