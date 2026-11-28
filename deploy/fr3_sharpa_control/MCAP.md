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

队列溢出、编码错误、设备消息停滞、倒退时间戳、写盘失败会终止录制并触发现有执行清理。保持 partial/failed 证据，不生成虚假 complete。单文件接近 10 GB 时停止（按十进制容量计算，预留 16 MB 收尾空间）；不支持超长分片、断连续录、中断抢救。人工正常停止仍视为正常结束；评级不等于训练任务成功，也不等于数据通过认证。

软件下发目标来自 `/openpi/executed_joint_targets`，不代表硬件已执行。机械臂来源指令从自定义输入适配为标准 JointState。DAgger 高频所有权以标准 Int8MultiArray 保存，新回合的阶段与介入上下文仅在变化时写入 MCAP `/recording/control_context`（历史回合可在 events.jsonl），低频事件通过标准 String 保存；不将目标伪装为实测值。

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
| 新回合 MCAP `/recording/control_context`（历史 episode 的 `events.jsonl`）事件的 `context` | 控制上下文变化记录 | `context.phase` 为控制阶段，`context.mode` 为控制模式，`context.stage` 为任务阶段，另含介入编号、Prompt、模型版本 |
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

保存设置末尾同一行提供「录制头部相机深度」「录制腕部相机深度」，首次默认均勾选，可分别控制 cam0 和 cam1/cam2；下一行提供「录制双手触觉」，首次默认关闭。三个选择独立持久化，执行中冻结。两个格式都支持这些选择：rosbag 保留原始深度 Image；MCAP 将原生毫米 16UC1 无损编码为 compressedDepth RVL，保存对应深度 CameraInfo。触觉包含双手十指的 raw、deformation、wrench，共 30 路数据；另有10路紧凑采样索引/时钟信息，接触点作为10路条件通道保存。无接触点时不要求连续消息。原数采原有的20路为 deformation+wrench。

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


深度拆为头部与腕部两个独立开关：`dagger_head_depth` / `SHARPA_RECORD_HEAD_DEPTH` 控制 cam0；`dagger_wrist_depth` / `SHARPA_RECORD_WRIST_DEPTH` 控制 cam1/cam2。只订阅和检查选中相机的深度 Image 与 CameraInfo；头部为640×400@20，腕部为640×480@30。旧启动环境未设置 WRIST_DEPTH 时沿用 HEAD_DEPTH 以兼容旧三路开关。开启选项只在下次正常硬件启动时加载相应独立配置副本，不在线重启相机，不修改原遥操工程配置。

## 四路动作存储格式（v1 历史布局）

MCAP 录制端将 `/openpi/executed_joint_targets` 的 58D 软件下发目标按关节名称验证并拆分为四条标准 `sensor_msgs/msg/JointState`：

| 输出话题 | 维数 |
|---|---:|
| `/devices/fr3_left/action/command` | 7 |
| `/devices/sharpa_left/action/command` | 22 |
| `/devices/fr3_right/action/command` | 7 |
| `/devices/sharpa_right/action/command` | 22 |

四组保留同一原始 header.stamp、MCAP publish_time/log_time；数值不重采样、不插值。消息名称须恰好包含58个预期关节，缺失、重复、额外关节或数组长度错误均拒绝写入。空 velocity/effort 保持为空。此拆分仅改变 MCAP 存储布局，不改变控制发布接口或 rosbag 录制方式，也不将软件目标改称为硬件已接收指令。gateway反馈与ownership仍独立保存，转换器仍需进行状态语义适配。

已有 MCAP 可用本工程独立工具 `python -m deploy.fr3_sharpa_control.split_mcap_archive --root <保存父目录>` 生成验证副本；加 `--commit --backup-root <备份目录>` 才进行替换。只处理 A–D 的 MCAP，不修改历史 sqlite rosbag。替换前验证原manifest、所有Chunk CRC、完整新文件回读、逐通道原始消息信封和CDR载荷哈希；原episode目录整体移到备份位置。视频不解码重编码，非动作消息CDR逐字节保留；重新生成共享设备/录制配置版本、消息计数、manifest及checksums，不覆盖旧共享资产。episode_uid、等级和批注保持不变。

2026-11-28 将 A–D 现有 MCAP 转换为此布局，原始目录备份在保存父目录 `.mcap-format-backups/command-split-v1/`；每条转换结果见同目录外的 `command-split-v1-report.json`。试验样本 episode65 的完整格式检查报告为 `.mcap-format-backups/command-split-v1-active-validation/report/report.md`。`data-auto-processing` 本次不修改。


## 部署记录精简布局（compact-v1）

新 GUI 和 record_cli 回合统一使用此布局，MCAP/rosbag、纯部署/DAgger 相同。
历史记录不改写。日期/时间目录正常收尾后仅保留 `run.log`、`video/` 和
`mcap` 或 `rosbag` 快捷方式。快捷方式指向独立 `pending/episodeXX`，评级后跟随
`A/B/C/D/episodeXX`。录制前失败仍保留 `run.log`，不会创建假的完整样本。

`video/` 只包含 `head.mp4`、`left_wrist.mp4`、`right_wrist.mp4`、
`three_cameras.mp4`，使用 H.264/yuv420p；从已关闭的 MCAP 或 sqlite rosbag 后台导出。
这些是按接收时间轴生成的 30 Hz 查看视频；原始帧和采集时间仍以原始录制载荷为准。
不保存 `policy_views/`、ROI/四联视频、逐帧 JSONL、CSV 或 NPZ。

启动、核对、执行、录制和退出诊断合并到 `run.log`。运行中的状态 JSON 仅为临时
生命周期交接，收尾/视频导出结束后合入既有 `episode.json`，随后删除。
评级与最近批注也保存在该文件中；MCAP 修改元数据后重建清单和 SHA-256，原始
`data.mcap` 不改动。rosbag 的 sqlite 文件和原 metadata.yaml 保留。

DAgger 高频反馈快照、序列及网关关联、所有权和变化的阶段上下文直接
保存在 MCAP 标准消息通道，不在部署目录再复制；rosbag 保留原 `/dagger/frame`。
MCAP 新增 `/recording/dispatch`、`/devices/rig/state/dispatch_joints`、
`/recording/control_context`；数值数组的字段顺序写在
layout.dim.label。网关序列有单独 valid 位，缺失不伪装为有效序列 0。
策略输入的精确 ROI 不再留档，查看四路 MP4 不能恢复其逐帧裁剪输入。

未连接硬件的软件回归测试：
`.venv/bin/python -m pytest -q deploy/fr3_sharpa_runtime/tests/test_compact_records.py`。


## 当前 v2：机械臂 action 使用网关后目标

录制器版本 1.2.0 起，`four-joint-state-command-streams-v2` 的左右臂 `/devices/fr3_{left,right}/action/command` 来自 `/teleop/validated_arm_commands`，与原遥操数采选择相同。显式使用 protocol.FR3_POLICY_TO_COMMAND 的逆映射，将 `left/right_fr3v2_jointN` 规范化为原四路接口的 `left/right_fr3_jointN`；值和消息时间戳不改变。网关消息仅含某一侧时不为另一侧补造目标。该含义是网关实际输出的目标，仍不等于关节已经到达目标。

左右手 action 继续来自 Sharpa SDK 提交目标。独立 `/devices/{fr3_left,fr3_right,sharpa_left,sharpa_right}/state/joints` 继续保存真机反馈；格式规范明确区分 state 与 action，不用反馈值替代动作。高频审计中的软件命令和 gateway 接受状态仍保持原含义，不改称为网关后的数值。

2026-11-28 对 A–D 的 60 条 MCAP 只读核验：29 条另存有原始 `/teleop/validated_arm_commands`，合计248055条，左右侧均有完整7关节及有效原始时间戳，可据此进行真实语义迁移；31条未找到该数值流，对应旁路录制也没有该话题。接受标志、gateway_sequence、网关前目标和实测状态不足以无损反推出每帧网关输出。历史文件本次没有再次修改，不能把60条统一标记为v2。详情见 `validation/post-gateway-semantics-20261128.json`。转换器本次不修改。

### DAgger 精简字段对照与复核边界

| 原本地记录 | 精简后 MCAP | rosbag |
| --- | --- | --- |
| `frames.jsonl` 的 sequence/frame_index/monotonic_ns/gateway_sequence | `/recording/dispatch`，gateway_valid 区分缺失序号；gateway_sequence 是最近观察的反馈序号，不代表本帧目标已执行 | `/dagger/frame` 原完整 JSON |
| 逐帧 state（58D） | `/devices/rig/state/dispatch_joints`，保留本轮下发时取到的反馈快照；时间为 dispatch 时间，不能冒充各传感器原采样时间 | `/dagger/frame.state` |
| 原生传感器 state | 各设备 `/state/joints`，保留原消息时间及非空 velocity/effort | 原始 JointState topic |
| action、action_basis | 四组标准 action/command；臂为网关处理后的目标，手为 SDK 提交目标；软件输入仍单独记录 action/source | 原数值命令及 `/dagger/frame`，不要把 frame.action 当作网关后目标 |
| action_owner | `/devices/rig/action/ownership`，左臂、左手、右臂、右手顺序不变 | `/dagger/frame.action_owner` |
| mode/phase/generation/intervention_id/stage/prompt/model_version | `/recording/control_context` 变化事件；事件消息 phase=instant，控制阶段在 context.phase | `/dagger/frame`、`/dagger/events` |
| dispatch_timing | 不再写入 MCAP（仅为逐帧性能诊断） | `/dagger/frame.dispatch_timing` |
| 控制事件、拒单及网关故障 | `/recording/events`、`/recording/gateway_feedback`；录制窗口外退出诊断保留在 run.log | `/dagger/events`、`/teleop/arm_command_status`，窗口外诊断在 run.log |
| 模型身份、采集频率、控制参数、质量及批注 | episode.json 中的 run_summary/quality；原 RTC receipt 及推理/阶段诊断在 run.log | 同一 episode.json/run.log 机制 |

原 `labels.csv`、`interventions.json` 和 `supervision.npz` 是上述事实的派生表示，
不作为独立事实再次保存。尚未执行的整块预测动作和精确策略 ROI 不属于当前归档
规范要求，本次按已确认的精简方案取消；不声称能从记录重建完全相同的推理输入。
首次门控前的 Home/过渡/等待帧不补入采集流；真实异常结束仍保留原失败证据。

版本变更的软件合成回归与历史 episode 的只读格式检查，均不能替代新版本真机
按生产配置完成的 S1 验收。rosbag 使用原 sqlite3 数据结构，不宣称其符合 MCAP
容器要求；这里验证的是采集字段和生命周期在精简后不丢失。


## 腕部静态外参

本机配置的 `static_extrinsics` 使用法兰 link8 到相机光学坐标系的 T_tool_camera：左臂对应 cam2，右臂对应 cam1。录制器只在本地 MCAP 的 `/tf_static` 中补充，不向机器人图发布 TF。相机内部光学系变换保留，按 T_flange_root = T_flange_optical × inverse(T_root_optical) 接到原相机根节点，避免多父节点或重复光学旋转；静态快照合并所有发布者已收到的边。

历史回填工具 `python -m deploy.fr3_sharpa_control.backfill_wrist_tf --root <归档根> --profile <现场yaml>` 仅用于已经确认安装未改变的关闭回合，可用 `--episode A/episodeXX` 限定。先备份原 MCAP 与元数据到 `.calibration-backups/`，只重写静态 TF，逐条校验所有其他消息的原始载荷、sequence、log/publish time 不变，CRC/计数核对后原子替换并重算清单；旧共享资产保持原样，新增版本化标定资产引用。回填不重编码视频、不改变动作或评级、不回退其他格式迁移。原文件的质量与独立验证标志保留，使用者确认的历史安装适用性另行记录。

补静态外参不等于补齐全部历史动态 TF：若原包 `/tf` 没有消息，工具如实保留空缺，并记录 dynamic_tf_recorded=false。不能把新静态标定当成历史机器人动态位姿；如需从旧关节状态计算动态 TF，应另做注明模型来源的 FK 派生，而不是冒充原始实测 TF。

### 重复标定信息精简（2026-11-28）

MCAP 对各路 CameraInfo 保存首条快照及标定内容变化；仅 header.stamp 变化不重复写入。frame_id、内参、畸变、ROI、binning 等变化仍保存。设备在线检测继续统计原始接收消息，不受存储去重影响。逐帧 dispatch_timing 不再保存；实测反馈快照、原生关节反馈、动作、接管/阶段事件及 TF 保留。

历史包可通过 `python -m deploy.fr3_sharpa_control.prune_mcap_archive --root <保存根目录> --episode A/episode84 --commit` 精简。原包先完整备份到 `.size-prune-backups/`；保留消息的 CDR、双时间戳和序号逐字节校验，更新共享配置引用、统计、清单及校验和。仅处理已结束回合。

腕部深度固定为原生 640×480、30 Hz、Y16（2026-11-28 已通过 SDK 查询两台 Gemini 305 支持列表，并实测 RGB+深度同时出帧）。RGB 保持 640×480/30 Hz；头部深度保持 640×400/20 Hz。每轮录制保存驱动当前输出的 CameraInfo，不复用旧分辨率内参。

### 触觉采集依据与验证（2026-11-28）

本工程 SDK 直连路径在现有左右手连接上读取原生 tactile payload；不调用另一个手部驱动，不重复连接，不发送额外控制指令。读取与 ROS 序列化使用独立线程及64帧FIFO，按60 Hz轮询每指，SDK当前标准模式为30 Hz；不把轮询频率当作设备出帧频率，不重采样，不用待发最新帧覆盖旧帧。FIFO溢出、输出缺失、时钟/形状/位深改变会明确报错并停止触觉发布，原有录制健康检查会报告必需流失联并终止录制。

- 官方依据：[W01 手册](https://sharpa-robotics.github.io/sharpa-docs/#tactile-sensing)及[官方 URDF 坐标系说明](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml#model-description)。F6顺序为Fx/Fy/Fz/Tx/Ty/Tz，单位N/N·m；不缩放、不换轴。header.frame_id使用`<side>_<finger>_elastomer`，拇指IP／其余手指DIP关节轴中心为力矩原点。该局部坐标系不能冒充世界坐标系。
- RAW为240×320 uint8，DEFORM为240×240 uint8，F6是当前Python SDK暴露的float64；逐像素/数值保留，禁止隐式uint8截断。DEFORM保存SDK灰度编码，不能直接解释为毫米线性值。SDK原有RAW传输开启JPEG并自行解码；MCAP无损保存解码后的SDK像素，不声称恢复JPEG编码之前的物理相机像素。
- 所有触觉数据header使用原始SDK frame.ts转换的Unix纳秒；MCAP log_time仍为录制进程接收时间。`/recording/tactile/<side>/<finger>/sample_info`以标准UInt64MultiArray保存SDK帧序号、SDK帧时间、主机读取SDK的时间，保留检测前置丢帧和时延的依据。SDK时间已与主机时钟比对，但不声称独立认证了物理曝光时刻或硬件对时。
- CONTACT_POINT若存在则以标准Image无损保存原始数值数组，不伪装为3D坐标或力。当前两台设备没有观测到DIST_FORCE、DEFORM3、DEFORM3_SE；若以后启用新的输出，程序明确拒绝静默丢弃，需先添加该模式的通道描述和网格/单位依据。
- 当前两台设备的只读在线证据、实际SDK触觉配置及官方定义来源，保存在工作机校准资产的`tactile/`子目录，随新回合共享资产归档。旧episode不改写。

运行`bash deploy/fr3_sharpa_control/validate_tactile_recording.sh`进行隔离ROS域内的合成帧→真实DDS→正式MCAP录制器验证。设备空闲、Pilot及其他SDK使用者已退出时，加`--hardware`可进行触觉专用的真机链路验证（不使能关节、不发送机械臂/手指目标）；端口占用时在连接SDK前退出。报告和测试包位于`.deployment/validation/tactile-audit/`，不混入正式ABCD目录。这不替代带正常任务执行的真实S1认证。
