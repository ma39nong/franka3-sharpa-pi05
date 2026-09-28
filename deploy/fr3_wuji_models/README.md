# FR3 + Wuji 模型适配

此目录只处理权重身份、模型服务、归一化与 54/64 维映射；执行速度由 `deploy/fr3_wuji_slow/`、`deploy/fr3_wuji_medium/`、`deploy/fr3_wuji_fast/` 选择。

中速命令使用 `--model 54` 选择原 19999 适配。64 维使用
`64-lora-30hz`（原 30000）、`64-full-30hz`（原 30000v2）、
`64-full-15hz-ab`（原 25000）、`64-full-15hz-a`（原 25000-single）。
旧名称继续兼容，`64` 也保留为 `64-lora-30hz` 的别名。
内部服务身份和权重目录不改名，原生 54 维 20 Hz 权重仍使用 `20hz`。

- `registry.py`：模型名、默认 checkpoint、服务端口、模型节点频率与契约模块。
- `model_19999/`：原生 54 维模型服务。
- `model_30000/`、`model_30000v2/`：64 维模型服务及契约。
- `model_25000/`、`model_25000_single/`：15 Hz 的 64 维模型服务及契约。
- `model_20hz/`：0918 A/B 数据训练的原生 54 维、50 步、20 Hz 模型服务，默认端口 8006。

原 `deploy/fr3_wuji_30000*`、`deploy/fr3_wuji_25000*` 和 `experiments.weight_motion_eval.oneshot.policy_server` 的导入保留兼容；旧目录的执行包装和 shell 命令也保留。新增权重时，先在本目录实现服务与契约，再向 `registry.py` 添加身份和默认值，最后由速度入口显式选择。三种速度的参数总表见 [`FR3_WUJI_ARCHITECTURE.md`](../FR3_WUJI_ARCHITECTURE.md)。
