# 文档索引

当前 Sharpa 实机部署以[根 README](../README.md)和 [runtime 部署说明](../deploy/fr3_sharpa_runtime/README.md)为准。

| 文档 | 用途 |
| --- | --- |
| [新机器部署与研究交接](SHARPA_HANDOFF.md) | 仓库来源与版本、环境安装、外部资产、训练参考及历史代码查询 |
| [Sharpa 部署](../deploy/fr3_sharpa_runtime/README.md) | 全量／LoRA 统一入口、30 Hz、SDK 与日志 |
| [全量部署版本](../deploy/fr3_sharpa_full/README.md) | 全量模型独立入口、工作机配置与启动 |
| [LoRA 部署版本](../deploy/fr3_sharpa_lora/README.md) | 30 Hz LoRA 配置隔离、权重接入与工作机运行 |
| [Sharpa 部署结构](../deploy/fr3_sharpa_runtime/ARCHITECTURE.md) | 模型层、公共执行层和三档速度 |
| [保留的 Wuji 设计参考](reference_wuji_architecture.md) | 历史职责划分和轨迹方案，非当前运行入口 |
| [Prepare 流程](../deploy/fr3_sharpa_runtime/PREPARE.md) | 初始到位、重新观测、完整 chunk 循环 |
| [Sharpa 数据转换](../examples/fr3_sharpa/README.md) | 离线 bag 转换和 LeRobot 导出 |
| [归一化统计](norm_stats.md) | OpenPI 通用归一化说明 |
| [Docker](docker.md) | OpenPI 通用容器参考 |
| [远程推理](remote_inference.md) | OpenPI 通用服务与客户端参考 |
| [上游 README 归档](README_原版_2026-09-29.md) | 通用模型安装与示例，非当前实机启动入口 |
| [贡献指南](CONTRIBUTING_2026-09-29.md) | 上游开发约定 |

旧平台专用文档随对应代码移除，需要查阅时使用 Git 历史。Sharpa legacy 后端仅供历史兼容，见 [LEGACY.md](../deploy/fr3_sharpa_runtime/LEGACY.md)；当前执行入口使用 `sdk_direct`。
