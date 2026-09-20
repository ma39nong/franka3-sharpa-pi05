# FR3 + Wuji 慢速部署入口

与 `deploy/fr3_wuji_medium/`、`deploy/fr3_wuji_fast/` 并列的慢速入口。

从仓库根目录运行：

```bash
bash deploy/fr3_wuji_slow/run.sh --check --checkpoint /path/to/19999
```

`--check` 生成并检查部署配置，不启动设备。`--execute` 会驱动真机，继续使用原慢速入口的资格与现场值守要求；具体参数见 [`experiments/weight_motion_eval/oneshot/README.md`](../../experiments/weight_motion_eval/oneshot/README.md)。模型服务仍单独启动，服务端与执行端必须使用同一个 checkpoint。

本目录只提供新入口。慢速的规划、设备接口和安全边界仍由 `experiments.weight_motion_eval.oneshot` 实现；原 `experiments/weight_motion_eval/oneshot/run.sh` 和 Python 导入路径保持可用。新入口继续接受原 `PI05_ONESHOT_HOST_CPUSET` 环境变量，也可用 `PI05_SLOW_HOST_CPUSET` 设置主机 CPU 集。没有在这里复制一份设备逻辑。
