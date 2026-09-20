# FR3 + Wuji 模型适配

此目录只处理权重身份、模型服务、归一化与 54/64 维映射；执行速度由 `deploy/fr3_wuji_slow/`、`deploy/fr3_wuji_medium/`、`deploy/fr3_wuji_fast/` 选择。

- `registry.py`：模型名、默认 checkpoint、服务端口、模型节点频率与契约模块。
- `model_19999/`：原生 54 维模型服务。
- `model_30000/`、`model_30000v2/`：64 维模型服务及契约。
- `model_25000/`、`model_25000_single/`：15 Hz 的 64 维模型服务及契约。

原 `deploy/fr3_wuji_30000*`、`deploy/fr3_wuji_25000*` 和 `experiments.weight_motion_eval.oneshot.policy_server` 的导入保留兼容；旧目录的执行包装和 shell 命令也保留。新增权重时，先在本目录实现服务与契约，再向 `registry.py` 添加身份和默认值，最后由速度入口显式选择。三种速度的参数总表见 [`FR3_WUJI_ARCHITECTURE.md`](../FR3_WUJI_ARCHITECTURE.md)。
