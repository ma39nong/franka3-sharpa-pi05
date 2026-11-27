# DAgger MCAP 保存

在 GUI 的「DAgger 数据保存设置」内选择 MCAP 或 rosbag。首次默认 MCAP，修改后通过既有 preferences 接口持久保存。仅 DAgger 使用此选项，普通部署继续使用既有 rosbag。执行期间不可修改。

MCAP 是直接录制，不生成中间 sqlite bag。三路 RGB 使用原生大小 H.264，数值保持原生样本；控制循环不调用编码器。录制回合正常封口后在 `pending/episodeXX` 下形成 `data.mcap`、`metadata.yaml`、`episode.json`、`channel_stats.json`、`manifest.json`、`checksums.sha256`。评级后移动至 A/B/C/D 下同名 episode，批注作为 `data_quality.json` 保存并重新封装哈希。共享模型在 `assets/models/<hash>`、共享配置和标定在 `_shared/<hash>`。单独交付 episode 时须一起交付其引用的共享资产。

## 一次性准备

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05
bash deploy/fr3_sharpa_control/setup_mcap.sh
cp deploy/fr3_sharpa_control/mcap-profile.example.yaml .deployment/mcap-recording.yaml
```

依赖安装到本仓库 `.deployment/mcap-python`，不修改 GELLO 的共享硬件环境。当前机器已安装依赖，真实配置尚需填写：

- `model_root`：已物化 URDF/等价机型模型、关节映射、所有 mesh 依赖的真实目录。包引用和外部依赖须在归档前物化为相对路径。
- `calibration_root`：本批次真实标定目录；不能填测试模型或占位标定。
- `session_id`：批次标识；设备、通道、相机角色、稳定设备 ID、实际原生宽高与帧率按现场配置核对。
- `clock_mapping`：实际对时映射与证据，包括有效起始时间、参考时刻、不确定度和锁定状态。默认示例故意不通过校验，禁止用当前时间假冒设备对时证据。
- `color_source`：核实 RGB/BGR 输入为 full-range sRGB 后填写依据。输出显式转换为 limited-range BT.709 matrix/primaries、sRGB transfer，与 VUI 一致。其他输入色彩需要先适配，不能直接套模板。
- `channels`：模板包含当前三路 RGB、双臂双手、动作来源/下发、CameraInfo、TF、DAgger 所有权/事件。现场启用的其他原生标准通道也须登记。未启用的深度和触觉不自动启动。snapshot 仅用于 `/tf_static`，使用 transient-local QoS。

用 `SHARPA_MCAP_PROFILE` 可指定其他配置。选择 MCAP 时，缺配置、依赖、标定、模型或实时通道均明确报错，不降级 rosbag。与 rosbag 共用 RecordingGate：点击「开始当前阶段」后，在首个正式任务动作前开始接收录制样本；Home、等待开始、初始推理和入场过渡不写入样本。MCAP 可提前预热编码器/schema和准备空暂存容器，但此时 started_unix_ns 为空且消息计数为零；未开始即停止会撤销该空暂存，不生成正式 episode。静态 TF/CameraInfo 仍按已有快照要求缓存。

## 异常与界限

队列溢出、编码错误、设备消息停滞、倒退时间戳、写盘失败会终止录制并触发现有执行清理。保持 partial/failed 证据，不生成虚假 complete。短回合接近 2 GB 时停止；不支持超长分片、断连续录、中断抢救。人工正常停止仍视为正常结束；评级不等于训练任务成功，也不等于数据通过认证。

软件下发目标来自 `/openpi/executed_joint_targets`，不代表硬件已执行。机械臂来源指令从自定义输入适配为标准 JointState。DAgger 高频所有权以标准 Int8MultiArray 保存，阶段与介入上下文仅在变化时写 events.jsonl，低频事件通过标准 String 保存；不将目标伪装为实测值。

MCAP 的 `/recording/events` 将旧 DAgger `event` 原值写入标准 `event_type`，并保留原 `event` 及其余业务字段。旧 `phase` 是控制阶段，移入 `context.phase`；这些瞬时通知的事件生命周期为 `phase=instant`。已有标准事件保留其 start/end/instant 生命周期。此适配仅在 MCAP 录制进程中进行，不改变 ROS 原始事件、rosbag 或控制流程。厂商重定向 VERSION.json 所列的两个二进制依赖随标定证据归档；色彩解释记录保留实际配置，不再使用聊天确认文字作为文件引用。新资产产生新的内容版本，不回写旧 episode 的共享资产。

## 验证

### 格式转换程序必须区分的 phase 字段

字段拼写是 **`phase`**（不是 `phrase`）。转换程序不得根据顶层 `phase` 判断策略/人工控制阶段。

| 字段及位置 | 当前含义 | 示例 / 读取规则 |
| --- | --- | --- |
| `/recording/events` 的 JSON 顶层 `event_type` | 事件名称 | `policy_applied`、`gateway_status`；作为规范事件分类字段 |
| 同一 JSON 的 `event` | 兼容旧程序的事件名称 | 从旧事件转换时与 `event_type` 值相同，不代表另一个事件，不要重复计数 |
| 同一 JSON 的顶层 `phase` | **事件生命周期** | 仅 `start`、`end`、`instant`；旧程序逐条通知转换后为 `instant` |
| 同一 JSON 的 `context.phase` | **原控制阶段** | `policy`、`approach`、`hold`、`human_follow` 等；旧顶层控制阶段保存于此 |
| episode 的 `events.jsonl` 中 `control_context` 事件的 `context` | 控制上下文变化记录 | `context.phase` 为控制阶段，`context.mode` 为控制模式，`context.stage` 为任务阶段，另含介入编号、Prompt、模型版本 |
| `/devices/rig/action/ownership` | 每个动作样本的控制归属 | 顺序为左臂、左手、右臂、右手；0=保持、1=策略、2=人工。用于区分人工监督与策略样本 |

例如旧通知 `{"event":"policy_applied","phase":"policy"}` 转换为
`{"event":"policy_applied","event_type":"policy_applied","phase":"instant","context":{"phase":"policy"}, ...}`。
顶层 `instant` 不表示机器人处于某种瞬时控制模式；顶层 `start/end` 也不能替代任务开始/结束判定。
控制模式 `mode`（例如 `POLICY/HUMAN_LEFT`）、控制阶段 `context.phase` 和番茄任务阶段 `stage`（例如 `pickup/harvest_1`）是三个不同维度。

原始 ROS `/dagger/frame`、旧 rosbag 以及历史 MCAP 保留的 `/dagger/frame` JSON 仍使用旧顶层 `phase` 表示控制阶段；这是原始控制数据，不是规范事件生命周期。读取旧归档时必须先按 topic/schema 区分，禁止对所有名为 phase 的字段统一替换。历史修复可以保留原始诊断/帧 topic 作为辅助证据；不能把它们当成第二组动作样本。

转换建议：优先读取标准 ownership 判断动作归属，用事件 `context` 和 `events.jsonl` 补充阶段/介入信息；只有读取旧帧 schema 时才用旧顶层 `phase`。保留原时间戳，不用事件抵达顺序推造动作的采集时间。历史格式修复不补造当时缺失的 CameraInfo、TF 或传感器测量，也不将 failed 样本改成 complete。

历史相机内参可从 `episode.json` 的 `supplemental_calibration.camera_info` 读取：每项含 device_id、序列号、相对 path、SHA-256、宽高及 snapshot_stamp。其 path 指向共享标定内的完整 CameraInfo YAML（D/K/R/P、ROI、binning 等）。这是当前读取的后补标定，不是当时采集的 ROS CameraInfo；snapshot_stamp 仍为实际读取时间，不可改成历史图像时间。适用依据是操作者确认相机/安装未更换或移动，并核对原图尺寸；使用者还应保证焦距、成像模式、ROI、binning 等光学配置未变。转换器可以使用适用的静态内参，但不能声称恢复了历史 CameraInfo 消息或历史对时精度。原始未记录 CameraInfo 的通道仍如实标为 not_recorded_in_original，不会补造历史 TF，未知腕部外参仍保持未知。

合成软件验证（只写指定的新目录，不连接设备）：

```bash
.venv/bin/python -m deploy.fr3_sharpa_control.verify_mcap /tmp/new-mcap-fixture
```

真实设备手动完成正常 S1 录制并收尾后，在具备 MCAP 依赖和 ffprobe 的环境运行：

```bash
PATH="$SHARPA_GELLO_REPO/portable_deps/litchi_hardware/.pixi/envs/ros2/bin:$PATH" \
.venv/bin/python -m deploy.fr3_sharpa_control.check_mcap \
  /path/to/A/episodeXX --out /path/to/reports/s1-new
```

工具默认读取 `~/.local/share/sharpa-conformance/recorder-conformance`，可用 `--tools` 覆盖。报告始终在 episode 外。合成样本通过不能替代真实设备 S1 认证。两个工具包均已解压到上述专用目录，LeRobot 校验工具本次不调用。


## 可选深度与触觉

保存设置末尾提供「录制头部深度」「录制双手触觉」，首次均为关闭，修改后持久化；执行中冻结。两个格式都支持这些选择：rosbag 保留原始深度 Image；MCAP 将原生毫米 16UC1 无损编码为 compressedDepth RVL，保存对应深度 CameraInfo。触觉包含双手十指的 raw、deformation、wrench，共 30 路；原数采原有的 20 路为 deformation+wrench，新增记录 raw 是为保存全部已启用原生输出。

深度开启时，仅在下一次正常硬件启动使用独立深度配置副本，不修改共享相机文件、不在线重启当前相机。触觉仅订阅已有 SDK/ROS 发布者，不另开设备控制连接；选中但无数据时阻止开始/报告失败。RVL 原生库由 setup_mcap.sh 编译，已使用提供的独立解码器验证随机 uint16 图像及大小端无损往返。

当前在线验证结果、模型来源限制、人工手眼验收记录见 `.deployment/mcap-assets/live/在线核验结果.md`。`sdk_mapped` 表示 SDK 全局时间映射已核对，不代表跨设备硬触发同步，也不代表时钟误差已被测定。


## 队列吞吐与录制门控修复

三路视频分别由各自单工作线程按原帧序编码，MCAP仍按接收顺序由一个写线程落盘；不改变采样、编码质量或时间戳。视频、CompressedImage、Image 的 CDR uint8 数组使用整块写入；无需转换的标准 ROS 消息复用 ROS 原生 CDR。原队列容量仍为256，没有按满队列覆盖旧帧或跳过采样。正常停止会排空已接收队列并检查每个通道接收数等于写入数；不一致不报告 COMPLETE。

保持与 rosbag 相同的阶段门控：开关设置相同、开始门为首个正式任务动作前，之后暂停/接管/保持期间持续记录，停止执行时收尾。该门控在两种格式下由同一 RecordingGate 调用，不按格式另设规则。入场过渡期间不录制，与普通 play_task_frames 一致。Gate 等待后重置既有流看门狗，并重新获取反馈，不把录制初始化误报为控制周期超时。

无硬件原速回放验证（输出目录必须尚不存在）：

```bash
SHARPA_RECORD_FORMAT=mcap bash deploy/fr3_sharpa_control/record_session.sh benchmark \
  --source /path/to/episode/data.mcap --out /tmp/mcap-benchmark-new \
  --seconds 20 --tactile --depth
```

仅读取已有文件，使用 ROS 消息类型/序列化但不创建 ROS 节点、不连接硬件。报告统计原始发送频率、生产端迟到、接收/写入计数及队列峰值；回放纹理是软件压力样本，不用于训练或真机S1认证。
