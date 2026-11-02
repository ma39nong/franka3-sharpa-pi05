# 部署相机 SDK 2.10.6

部署的三路相机共用 Orbbec SDK。独立镜像 `franka-sharpa-cameras:orbbec-sdk-2.10.6` 在原工作机镜像之上安装官方 SDK 2.10.6（含匹配的 extensions 和许可证），启动时在 ROS overlay 初始化后设置库搜索路径，并检查实际 SDK 版本。保留原有 ROS 驱动、序列号映射、IP 直连、global 时间戳和诊断改动。

只影响本部署的 `openpi-cameras` 服务。采集仓库、共享基础镜像、机械臂服务及相机固件不变。升级不代表相机时序或完整机器人执行已通过验收，也不会放宽 200 ms 新鲜度、100 ms 对齐及频率检查。

## 构建与启用

从 [官方发布页](https://github.com/orbbec/OrbbecSDK_v2/releases/tag/v2.10.6) 获取 `OrbbecSDK_v2.10.6_202609291004_575347c_linux_x86_64.tar.gz`。构建脚本固定核对官方 SHA-256 `23ec9f7023881cc9061bf0b4ba65735fb16c10e168fd5d62b1548c847e7ddb46`：

```bash
bash deploy/fr3_sharpa_runtime/build_camera_sdk.sh /path/to/OrbbecSDK_v2.10.6_202609291004_575347c_linux_x86_64.tar.gz
```

在本机 `.deployment/local.env` 中配置：

```bash
export SHARPA_CAMERA_IMAGE=franka-sharpa-cameras:orbbec-sdk-2.10.6
```

之后按原 GUI/命令启动，会新建使用该镜像的相机容器。入口日志应有 `[camera-sdk] Orbbec SDK 2.10.6`。现有驱动通过 `RTLD_NOW` 加载检查后才启用本机配置；这不替代真实出图检查。

## 回退

停止当前部署并等清理结束后，将同一配置项改为：

```bash
export SHARPA_CAMERA_IMAGE=franka-upper-body-teleop:latest
```

下次启动恢复原镜像和 SDK；无需重刷设备。不要在执行中替换库或重启相机。

2026-10-31 本机升级记录和相机检查日志保存在 `.deployment/sdk-2.10.6-check-*`，升级前的本机环境备份为 `.deployment/local.env.before-sdk-2.10.6-*`。
