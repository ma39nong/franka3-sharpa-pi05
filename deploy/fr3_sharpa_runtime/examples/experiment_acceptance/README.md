# 合成实验归档与自动报告示例

**这里全部是确定性synthetic样例，不是真实相机、GPU或机器人实验。**

- [中文示例报告](report/report.md)
- [完整统计JSON](report/summary.json)
- [逐请求CSV](report/switches.csv)
- [帧级绘图CSV](report/frames.csv)
- [时钟与来源清单](logs/analysis-manifest.json)

`logs/`仿照9f3f3c2的真实日志字段，包含六次在线切换、七阶段生效、旧Future丢弃、worker交接及终态快照。所有主机/boot域和时钟值均为虚构；不能复制为实机证据。输入只保留各次切换附近的帧，故报告中的5处frame_gap是刻意保留的片段间缺口，分析不会跨这些缺口求导。

预期软件结果：6/6确认接入；每次点击→登记5ms、登记→观测24ms、新Prompt推理60ms、安排→计划桥接开始9ms、桥接200ms、登记→实际接入300ms、点击→实际接入305ms。全部为构造值，不是性能实测或可靠性认证。右手误差构造为0.003rad，双臂为0.001rad。抓握、接触力及采摘成功始终缺失，即使合成终态为COMPLETED。

可以只重算报告：

```bash
.venv/bin/python -m deploy.fr3_sharpa_runtime.experiment_analysis \
  deploy/fr3_sharpa_runtime/examples/experiment_acceptance/logs \
  --out .deployment/acceptance-example-report
```

生成另一份新样例（不会运行协调器或设备）：

```bash
.venv/bin/python -m deploy.fr3_sharpa_runtime.experiment_example \
  --out .deployment/acceptance-example-new
```

输入日志使用SHA-256校验；报告生成时间和分析源绝对路径随运行环境改变。完整规则见[工具文档](../../EXPERIMENT_ANALYSIS.md)，实机前提见[验收规程](../../HARDWARE_ACCEPTANCE_ZH.md)。
