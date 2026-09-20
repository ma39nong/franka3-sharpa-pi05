# FR3 + Wuji 慢速部署

从仓库根目录运行：

```bash
bash deploy/fr3_wuji_slow/run.sh --check --checkpoint /path/to/19999
```

慢速规划、执行、连续调度和配置现位于本目录：`planner.py`、`core.py`、`runner.py`、`continuous.py`、`live.py`、`config.yaml`。公共设备启动、占用检查和 IPC 位于 `deploy/fr3_wuji_runtime/`。旧 `experiments.weight_motion_eval.oneshot` 导入与 `run.sh` 仍可用，用于现有脚本和测试的兼容。

`--check` 生成配置，不启动设备；`--execute` 会驱动真机，须配合资格记录或现场值守模式。模型服务独立启动，服务端与执行端必须使用同一 checkpoint。所有运行参数、模型默认值与目录关系见 [`FR3_WUJI_ARCHITECTURE.md`](../FR3_WUJI_ARCHITECTURE.md)。原 `PI05_ONESHOT_HOST_CPUSET` 仍受支持，`PI05_SLOW_HOST_CPUSET` 优先。
