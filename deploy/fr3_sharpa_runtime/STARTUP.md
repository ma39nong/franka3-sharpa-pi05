# Startup reference

当前推荐入口和完整命令见 [README.md](README.md)。当前真机部署使用两个终端：

1. `./serve.sh`：加载 π0.5 policy。
2. `./runtime.sh --execute --enable-hardware --sharpa-backend sdk_direct ...`：从当前状态观测、推理并执行 chunk；添加 `--home` 才先回位。

旧 permit、handoff、acquisition、guarded driver 说明仅适用于 legacy fallback，见 [LEGACY.md](LEGACY.md)。
