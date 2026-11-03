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

## 相机订阅与时效检查

部署端图像订阅使用 Reliable / KeepLast(1)，源时间对齐所需的已接收历史仍由应用缓冲保存。相机监控在每次检查前先处理待收回调：首次等待最多 50 ms，随后最多 16 次非阻塞处理，追加处理预算为 10 ms，避免只处理了腕部回调就用旧的头部记录判断失效。

这项处理不修改时间戳，也不延长 200 ms 新鲜度或 100 ms 对齐上限。真实的设备、网络或调度延迟仍会触发停止；应结合日志中的源时间年龄和接收年龄检查。软件回归测试不能替代三路相机在真机运行负载下的验证。

## 启动时间同步收敛

2026-11-02 的独立相机测试发现，右腕相机可保持约 33 ms 的接收间隔，却在启动阶段发送接近零或落后约 11.6 秒的源时间戳。一次短暂的正常出图窗口不足以证明 SDK 的 global 时间戳已经稳定；原来的连续 3 秒出图检查可能过早放行。

部署现在额外要求：相机验证进程至少观察 30 秒，每一路源时间戳连续 10 秒新鲜且严格递增，接收间隔不超过 150 ms，再允许启动执行端。发生过期、未来、倒退时间戳或长接收间隔时，该路的稳定计时重新开始。序列号、频率、200 ms 新鲜度和 100 ms 对齐检查仍需同时通过；启动超时则拒绝执行，不修改时间戳或切换时间域。

这会增加部署启动等待时间。运行期间不适用启动等待策略；发现失效仍会停止动作并等待双手失能确认。`camera_alignment_failed` 现在同时记录各路源时间年龄和接收年龄，便于区分持续收到旧时间戳与实际断流。

仅延长启动等待并未解决右腕的问题：19:03 的两阶段启动中，右腕仍反复出现不合格帧。后续对照发现，右腕单独指定了 YUYV，而同型号、同固件的左腕使用 MJPG。部署可用独立的只读参数挂载为右腕恢复 MJPG 传输，避免修改共享遥操仓库。生成配置时只改变 `color_format`，其余字段从当前工作站配置复制：

```bash
source deploy/fr3_sharpa_runtime/env.sh
"${SHARPA_PYTHON:-.venv/bin/python}" -m deploy.fr3_sharpa_runtime.camera_profile --gello "$SHARPA_GELLO_REPO"
```

本机 `.deployment/local.env` 中启用生成的绝对路径，例如：

```bash
export SHARPA_CAMERA_CAM1_PARAMS=/home/descfly/LTT/franka3-sharpa-pi05/.deployment/camera_cam1_mjpg_params.yaml
```

`HardwareSession` 仅在此变量存在时添加 `compose.camera-profile.yaml`。路径必须是已存在的绝对文件路径，否则在 Docker 调用前失败。右腕仍为 640×480、30 Hz、global 源时间戳；不会改成主机接收时间戳。共享相机配置和固件不变。回退时从本机环境中删除该变量，再正常启动新的相机服务。

相机内部验证上限为 90 秒，外层进程等待为 105 秒，为初始化、30 秒观察与连续 10 秒稳定检查留出预算。超时仍拒绝执行；延长时间本身不代表修复时间戳。源时钟日志同时列出 `stale_source`、`future_source`、`nonadvancing_source`、`receive_gap` 的累计次数及最近异常数值。

## 头部 RGB-only 配置

头部原配置还启用 20 Hz 深度流和帧同步，而本部署的模型观测与录制只使用 RGB。独立对照中，关闭未使用的深度流及帧同步能降低头部 RGB 的源时间年龄。生成独立配置：

```bash
source deploy/fr3_sharpa_runtime/env.sh
"${SHARPA_PYTHON:-.venv/bin/python}" -m deploy.fr3_sharpa_runtime.camera_profile --gello "$SHARPA_GELLO_REPO" --camera cam0
```

生成器只设置 `enable_depth: false` 和 `enable_frame_sync: false`，其余参数从当前工作站的 `camera_primary_params.yaml` 复制。RGB 保留 640×400、20 Hz、原曝光设置和 global 源时间戳。通过 `.deployment/local.env` 启用部署专用只读挂载：

```bash
export SHARPA_CAMERA_CAM0_PARAMS=/home/descfly/LTT/franka3-sharpa-pi05/.deployment/camera_cam0_rgb_params.yaml
```

启用该配置后，本部署不提供头部深度流；如果将来要给模型增加深度输入，需重新评估配置。共享遥操文件不变。回退时移除本机环境中的 `SHARPA_CAMERA_CAM0_PARAMS` 后再正常启动新服务。相机时效、对齐和源时钟检查仍然执行，不使用主机接收时间替代源时间戳。
