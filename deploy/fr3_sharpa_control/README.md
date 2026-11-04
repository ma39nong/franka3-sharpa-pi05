# 统一部署控制台与 Chunk 录制

面向现场操作人员的逐步说明见 [部署 UI 使用手册](../../部署UI使用手册.md)。本页保留模型契约、模式实现与维护细节。

## 工作机启动

推荐在已配置工作机环境、包含所需模式的部署仓库根目录运行，再在UI中选择 main 或 Soft 的部署目录：

```bash
bash deploy/fr3_sharpa_control/ui.sh
```

沿用现场部署前提：先结束其他遥操和旧执行端，避免设备占用。打开 `http://127.0.0.1:8766`。可用 `--port 其他端口` 更改UI端口；仅监听回环，不对局域网开放。打开页面不连接设备。三终端命令集中在仓库根目录 [部署启动指令.md](../../部署启动指令.md)。

1. 选择本机部署仓库和checkpoint目录，优先自动识别动作类型；没有训练证据时按训练配置选择Absolute或双臂Delta／双手Absolute。点击识别权重，按可用范围选择串行／RTC／Paper RTC／Train RTC／Train Paper RTC／Soft模式。全量／LoRA结构仍自动核对，可选训练JSON交叉验证，Paper资产检查必须匹配。
2. 点击**启动推理**。只检查权重并加载GPU、预热、核对服务身份，不启动硬件、相机、录制或机器人执行；即使选择了Home或录制，也不需要此时提供Home配置或ROS环境。推理端预热期间可以点**停止推理**取消。
3. 等推理端显示就绪，选择Current/Home、速度、步数、轮数、录制和输出位置，点击**核对执行配置**。
4. 点击**启动执行**。必须已有与当前仓库、提交、权重和模式匹配的已预热推理端；执行入口只检查/复用该服务，不会隐式加载或切换模型。每次执行都自动运行硬件 `up`，然后按配置开始执行。
5. 点击**停止执行**，等双手失能确认与硬件清理结束。推理端保持运行。可重新选择Current/Home并再次核对、启动执行，仍自动包含 `up`。
6. 不再使用推理时，点击**停止推理**。执行或失能清理期间禁止停止推理；更换权重、模式或代码目录后需重新点击启动推理，本UI仅停止自己拥有的模型进程。外部终端启动的匹配服务可通过启动推理握手复用，停止时仍需回原终端操作。

加载推理、执行及清理期间配置冻结；推理就绪但未执行时可以修改执行配置，无需重载模型。停止不会自动回Home。关闭页面不会停止进程；UI服务器终端Ctrl-C会取消推理加载，或先停止执行并等待既有清理，再关闭自己拥有的推理服务。

### 页面布局

宽屏使用三行三列：上排依次为权重与代码、执行配置、推理与执行；中排并排显示硬件／推理／执行日志；底排为头部cam0、左腕cam2、右腕cam1三个视频预留框。预留框尚未订阅或显示视频，不改变相机启动与录制逻辑。页面铺满窗口宽度；窄屏自动改为单列。执行配置详情可折叠，避免挤占日志和后续视频区域。

顶栏使用白色背景和用户提供的透明PNG标识，只显示深蓝色图形与“厘清智能”；不显示副标题、标语或动画。源图保存在 `assets/liqing-logo.png`，使用其透明通道以CSS遮罩显示统一深蓝色，不含背景色块。状态提示保留在右侧。

### 三个终端日志

页面中排固定三个同时可见的窗口：**终端一·硬件**（up与硬件清理）、**终端二·推理**（权重检查与持续模型日志）、**终端三·执行**（参数检查、执行、录制与视频导出）。不再按每个内部文件增加日志窗口。三个窗口分别滚动，不互相切换隐藏。推理日志不会因开始新一轮执行而消失，硬件/执行显示最近一轮。

详细文件仍完整保存在磁盘。只启动推理时日志存于仓库 `.deployment/control_model_sessions/日期/小时.分钟/`；不会创建桌面部署记录。实际执行按所选父目录自动创建日期/小时.分钟目录。视频导出在后台进行，不阻塞下一轮执行。

## 当前覆盖范围

| 权重（均为30Hz、原生58D、50步） | main | soft-inpainting-test |
| --- | --- | --- |
| 全量 Absolute | 普通串行、RTC | 普通串行、RTC、串行Soft、异步Soft |
| 全量普通 Delta（双臂delta／双手absolute） | 普通串行、RTC | 普通串行、RTC、串行Soft、异步Soft |
| 全量 Paper Delta v1（包括本次7999／T3） | 普通串行、Paper RTC | 同步本次适配后：普通串行、Paper RTC、串行Soft、异步Soft |
| 未合并 Absolute LoRA（VLM16／Expert32配方） | 普通串行 | 普通串行 |

这不是通用权重加载器：LoRA Delta、其他动作顺序/维度/频率、未知模型变体、其他论文资产版本尚未接入。模式列表同时受所选仓库和模型限制；旧分支没有Paper适配时会阻止启动。现场硬件验证与离线模型验证分开记录。

Paper通过checkpoint自己的 `delta_paper.json`、`delta_paper.npz`、`norm_metadata.json` 和参数中的噪声状态自动识别；不能用手动选择Delta绕过缺失或冲突的资产。验证资产摘要、30Hz/50×58、双臂Delta掩码、统计范围、噪声矩阵与beta一致性。支持v1的global_quantile、global_zscore_arms、timestamp_zscore_arms，beta范围[0,1)。Paper T3对应的7999为timestamp_zscore_arms、beta0.5；7999只是训练步数，普通Delta的同名目录不能当作Paper权重。

Paper仍使用 `full-delta` 启动入口和端口；服务自动切换至Paper模型与逐时间步变换。普通Delta保留原变换。Soft的旧26..29参考点会按**新chunk的0..3索引**归一化；Paper RTC允许1..50步参考：先补齐50步走完整的绝对值→双臂delta→分位数归一化→Paper变换，再裁回真实参考长度；参考按**新chunk的0..K−1索引**编码。补齐部分不参与RTC引导，输出仍是50步。普通RTC仅用于非Paper权重。默认噪声按训练的Cholesky矩阵生成，显式传入的噪声视为已在目标分布，不重复相关化。推理输出仍为绝对关节位置。

2026-10-30离线验证：本次7999在RTX5090上通过真实加载、普通推理、RTC引导、串行Soft四点参考、异步Soft四点参考smoke；输出均为有限50×58。异步Soft仍为50→30→26＋4→20，Soft施加6次；未连接机器人，不能替代工作机实机验证。合并LTT录制修复后的回归：main292项；Soft431项（16条既有JAX/Flax弃用警告）。模型smoke报告保存在各工作目录忽略Git的 `.deployment/paper_7999_audit/`。

### Paper RTC 启动

更新代码后重启UI服务并刷新页面。选择Paper checkpoint、双臂Delta／双手Absolute，点击识别权重，再选择 **Paper RTC · 时间步归一化**。重规划触发节点可继续设15，速度固定Fast，轮数0表示持续重规划。它会继续产生后续chunk，直到停止、达到轮数或出现异常。

切换模式需要停止旧推理服务并重新点击启动推理。新服务使用 `--enable-paper-rtc`，启动时实际预热K=50、35、1三种参考，全部通过才发布就绪标记。执行端使用 `--broker-mode rtc --paper-rtc`，在连接运动SDK前核对适配与预热标记；旧服务不能冒充Paper RTC。模型、Home、录制等其他配置仍按页面流程设置。

如使用三终端命令，在已正确设置Paper checkpoint的 `full-delta` 环境中：

```bash
# 模型服务
bash deploy/fr3_sharpa_runtime/serve.sh --model full-delta --enable-paper-rtc
# 执行参数离线核对（不加载模型、不连接设备）
bash deploy/fr3_sharpa_runtime/runtime.sh --model full-delta --check \
  --broker-mode rtc --paper-rtc --speed fast --execute-steps 15 --num-chunks 0
```

短参考回归测试覆盖K=1/4/15/35/49/50、不同新观测状态、时间步归一化、绝对动作还原与参考之外零引导；这次改动没有启动机器人。实际GPU预热由下一次启动新模式时完成，过去仅K=50的RTC smoke不能验证本次35步故障。

## 权重身份

参数元数据用于核对全量/未合并LoRA和原生58D投影；不能从目录名推断动作语义。当前已核对的AB4 absolute和普通delta统计指纹、完整Paper资产可以自动识别。其他权重优先提供归档训练JSON，需统计SHA256、模型结构、30Hz、horizon50、token256匹配；确实没有可读记录时，可由操作员根据训练配置明确确认语义。该确认按manifest、统计以及Paper算法资产的内容指纹保存，再次选择同一权重自动恢复；路径同名但内容变更不会沿用旧绑定。

未知语义会阻止启动；不会将名字含delta的目录静默当作delta。LoRA delta当前没有新部署配置，会明确拒绝。模型全局参数形状和推理输出还会在GPU预热时核对。5999是训练步骤号，不代表某种执行模式。

## 步数含义

| 模式 | UI设置 | 行为 |
| --- | --- | --- |
| 普通串行 | 可填1..50，如15或20 | 每轮执行对应模型动作前缀后重新推理 |
| RTC／Paper RTC | 可填1..24，fast固定 | 后台重规划触发源节点；不是固定执行前缀；轮数0或至少2 |
| 串行Soft | 源执行点26，压缩目标20 | 50→30→26+4，原cubic策略允许条件不满足时回退26 |
| 异步Soft | 固定26→20 | rolling-v3、PCHIP、固定4参考，段边界切换；拒绝旧15点异步协议 |

Main提供普通串行、RTC和Paper RTC；Soft分支增加串行Soft和异步Soft。UI读取所选代码目录，只启用其中实际支持的模式。推荐两个分支各保留一个部署目录，在UI选择对应目录，不在运行中git switch。独立目录复用启动UI所用的OpenPI Python；Home文件优先选择目标仓库自己的配置，缺少时使用启动UI仓库的工作机配置，ROS overlay也可复用。选择Home时会在启动硬件前解析配置，缺失即报错。

## 启动前部署独占相机

主界面新增“启动前相机策略”，默认是“部署独占接管”，也可选“要求相机空闲”。核对配置会显示并冻结该选项；改动后需重新核对。点击启动执行即授权按此策略处理相机：独占模式会中断使用旧相机服务的遥操预览或录制，模型预热阶段不会接管相机。

接管只针对 Docker 中标签为 teleop-cameras／openpi-cameras、命令为 triple_camera.launch.py、且挂载配置的三台序列号及头部/左右语义与本机部署配置完全一致的服务。使用不可变容器 ID 停止，检查容器 Running=false、PID=0，并等待可见USB/网络/驱动占用消失；不按容器名通配停止，不执行remove-orphans、USB reset或杀未知进程。部分相机重叠、无法核对配置或未知进程占用均拒绝启动，并记录容器名、命令、可见PID或设备句柄。操作系统权限可能限制占用诊断，不能将未列出PID当作设备一定空闲。

释放后只启动一个部署相机服务。只读ROS闸门查询驱动 GetDeviceInfo 返回的实际序列号，校验头部cam0／左腕cam2／右腕cam1映射；要求三路有效RGB/BGR、唯一发布者、连续3秒稳定出图（头部至少18Hz、双腕至少25Hz、连续帧间隔不超过150ms），并复用runtime的200ms新鲜度及100ms时间对齐限制。Current模式仍在通过后启动机械臂/网关依赖，并在执行端连接SDK前再次验证。固定Home模式在支持并行启动的部署目录中改用下面的汇合流程。

固定Home并行启动顺序：完成独占接管及设备释放 → 启动唯一部署相机及机械臂/网关 → 执行会话同时启动只读视频验证与SDK连接/关节反馈校验/Home → 两边均成功后，重新获取Home之后的七路新鲜观测 → 推理及任务执行。Home目标不再通过带图像的完整观测获取，而使用原四路关节反馈及命名Home配置；原运动范围、速度、反馈新鲜度和失能确认不变。普通串行、RTC、Paper RTC、Train RTC和Train Paper RTC共用此路径；独立Soft目录只有同步支持代码后才启用，异步Soft保持原启动顺序。

固定Home的初始手部边界恢复：仅固定Home请求携带 `home_recovery=True`，在普通Home前调用独立恢复模块。初始反馈合法时直接跳过；小幅越界时，确认两手SDK支持原生插值及速度系数读回，临时把速度系数降到 `min(原值, 0.05)`，将合法目标放在最近边界内侧0.005 rad（窄关节范围按比例缩小）。恢复目标与实测值的差须小于现有SDK 0.10 rad跳变上限；不允许更大越界自动恢复。所有下发命令仍通过原上下限校验，不发送机械臂目标，不开启录制，不修改真实反馈。反馈到合法范围且距恢复目标不超过0.003 rad并连续稳定0.3秒后，读回并恢复原速度系数，再进入原Home。SDK降速设置完成后、尚未发送恢复动作前，最多等5秒，要求四路关节反馈的本地接收时刻均晚于设置完成，且通过原200ms新鲜度和同步检查；再按新实测值重算目标及跳变范围。动作阶段最多10秒，期间不重试失效反馈；无进展2秒、反馈失效、30Hz超时、停止或速度还原失败均阻止任务并走原失能退出。还原速度后，同样在不发送动作的情况下最多等5秒真实新反馈，再进入普通Home。日志新增 `home_recovery_feedback_wait/ready`、反馈源/接收年龄和限频的 `sharpa_feedback_read_error`，不延长200ms阈值。速度系数为无量纲，不代表已验证的rad/s；实际恢复速度和边界需现场验收。Current、策略prepare、串行任务及RTC任务不调用该模块；推理、裁剪、任务安全阈值和SDK配置对象保持原路径。独立Soft目录需同步对应代码后才支持此恢复。

**行为变化：选择Home并启动执行后，即使视频尚未通过稳定性/序列号检查，臂手也可能已经开始固定Home运动。** 视频失败或超时会通知执行端中止Home，并等待原失能及退出确认；Home失败则取消视频等待，GUI停止同时取消两条流程。只有视频PASS和Home完成汇合后才允许任务推理，视频失败不重启相机或继续使用旧图像。

GUI对已支持的固定Home自动使用 `hardware.sh up --exclusive --defer-camera-check`，其输出的相机就绪状态为 `PENDING`，不代表视频通过；后续必须使用 `runtime.sh --execute --enable-hardware --sharpa-backend sdk_direct --initialization home ...` 的受监督执行入口完成检查。默认不带 `--defer-camera-check` 的硬件命令仍保留原先先检查视频的行为。启动页面和模型预热不会执行Home。

相机锁跨同一用户的部署目录生效，持有至执行、失能确认及清理结束，运行中不允许另一个启动入口抢占。只读相机监控持续检查图像、时间戳和发布者身份；丢图、监控退出或心跳超时会向执行端发送一次SIGINT，等待原有失能与清理流程结束，不会在运动中自动重启相机。恢复需要结束后显式重新启动。未知占用不杀进程；失败时不自动恢复旧遥操服务。

运行中若各路图像仍满足200ms源时间/接收时间新鲜度，但暂时找不到跨度不超过100ms的对齐组合，监控会继续接收新帧并重试，默认恢复窗口为0.3秒。计时从首次失败开始，收到新帧不重置计时；只有重新取得合格对齐观测才清零。窗口到期仍未恢复（或恢复结果到达时已超时）则退出。可在启动UI/执行端前设置 `SHARPA_CAMERA_ALIGNMENT_RECOVERY_SEC` 为0至0.5秒；0恢复原来的单次失败即退出行为。该窗口仅控制监控对短暂对齐抖动的退出判定，不保证运动暂停，也不扩展已有0.75秒健康心跳上限。

等待期间只向stderr记录 `camera_alignment_wait` 及各路时间戳、年龄和历史范围，不发送PASS、不更新健康心跳、不向推理提供无效观测；恢复时记录 `camera_alignment_recovered`。实际模型观测仍须通过原来的100ms对齐与200ms新鲜度限制。时间戳过旧/未来、图像无效、发布者缺失/重复或身份变化仍按原规则立即报错，SDK及机械臂反馈watchdog保持不变。这里没有照搬数采的90秒启动等待或将150ms单路帧间隔当作跨相机对齐阈值。

已知服务正常停止并不保证设备本身没有固件/USB问题；实际序列号或出图闸门不通过时仍会阻止Home。相关信息见hardware.log与runtime.log。所有在GUI中选择的main/Soft部署目录都必须同步相机闸门代码，旧目录会在配置阶段明确拒绝。

首次更新后编译新增的相机身份查询消息（只编译消息，不启动设备）：

```bash
bash deploy/fr3_sharpa_runtime/build_messages.sh
```

只读占用核对（不停止现有服务、不打开相机）：

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh camera-check
```

三终端启动需明确选择接管：

```bash
bash deploy/fr3_sharpa_runtime/hardware.sh up --exclusive
```

不带 --exclusive 时，发现同组已知服务也会报错，不会静默中断它。camera_probe只创建图像订阅及设备信息服务客户端，不创建运动发布器、不连接SDK。

## 自动录制和归档

勾选录制后，UI在执行前检查ROS bag和视频依赖，提前建立独立录制进程和订阅，但此时不创建bag写入器。Home和初始首帧接近期间不写包。运动worker完成接近后，先取得录制就绪应答，再开始发送第一段任务chunk；RTC/异步Soft在首次stream播放前打开同一个门。

录制从第一段任务开始持续到执行结束或Ctrl-C，不在段间暂停。按停止后立即结束写包；worker正常结束/异常退出也在双手失能确认之前关闭录制，避免把最长120秒的失能等待录进视频。录制收尾错误不能绕过双手失能逻辑。UI等待执行、失能与硬件清理完成后才能再次启动。手失能后是否还能握住物体由实际设备状态决定。

默认父目录按XDG Desktop、已有的“桌面”/“Desktop”自动定位，也可选择其他目录：

```text
<选定父目录>/2026-10-30/15.43/
  run.json                 # 冻结配置、分支提交、模型与统计身份
  recording.json           # 起止时间、各topic消息数、缺失topic
  rosbag/                  # 原始SQLite3 bag和metadata.yaml
  video/
    head.mp4
    left_wrist.mp4
    right_wrist.mp4
    three_cameras.mp4
  logs/                    # 执行trace与独立shutdown报告
  runtime.log
  ui_status.json
  video_export.json
```

名称使用工作机本地的小时和分钟，如15.43、15.50，不加入秒。同一分钟重复启动时使用15.43-2、15.43-3等后缀；原子创建目录，不覆盖既有数据。失败启动也保留该时间目录和排查记录。旧编号目录保持原样。不勾录制时不启动bag/video任务，仅保存配置和运行日志。

录制三路RGB、左右FR3反馈、左右Sharpa反馈、原双臂命令，以及新增 `/openpi/executed_joint_targets`（完整58D、实际裁剪/插值后下发目标）。实测反馈与发送目标分别保存，不将命令当成实际到位。原图像header时间戳保留在bag中。视频以同一录制进程的ROS接收时间轴30fps对齐，低帧率画面保持最近收到帧，不按相机帧号硬拼；画面标明head/left_wrist/right_wrist和经过时间。

bag关闭后导出四个视频。缺失topic、写盘错误、视频编码失败不会报告成功；原始bag保留。视频在独立低优先级队列导出，界面单独显示各轮导出状态。执行、失能和硬件清理完成后即可再次启动，不必等待旧视频编码；导出使用旧轮次冻结的目录，不会写进新实验。导出文件已存在时拒绝覆盖；重新导出前将失败的视频目录移走，保留原始bag。

## 不使用UI时的三终端录制

保留硬件和模型两个终端，只在第三终端给原执行命令套一个包装：

```bash
export SHARPA_RECORD_ENABLED=1  # 0为不录制，1为录制并导出视频
bash deploy/fr3_sharpa_control/record_execute.sh -- \
  bash deploy/fr3_sharpa_runtime/runtime.sh --model full-delta \
  --execute --enable-hardware --initialization current --speed slow --num-chunks 0
```

`SHARPA_RECORDINGS_DIR` 或包装的 `--output` 可指定归档父目录；仍按本机日期和小时.分钟命名。不要再额外启动旧 `record_rosbag.sh`，也不要在执行命令后接 `| tee`。旧录制脚本仅为历史兼容，新手册使用统一包装。

## 依赖与验证边界

部署相机可单独使用 SDK 2.10.6 镜像，构建、启用与回退见 [相机 SDK 说明](../fr3_sharpa_runtime/CAMERA_SDK.md)。三路共用 SDK，保留原驱动与时序检查；此项不更新设备固件。

ROS工具沿用工作机Pixi ros2环境、teleop_interfaces overlay、相机和SDK；录制依赖rosbag2_py，导出依赖同一ROS环境中的numpy和cv2（需可用MP4编码器）。可只检查依赖，不创建ROS节点和设备连接：

```bash
bash deploy/fr3_sharpa_control/record_session.sh check
```

本次开发不自动安装或修改工作机SDK/ROS。缺依赖时，勾选录制的启动会在硬件up之前失败并显示日志。界面、模式组合、录制门、编号、停止和合成视频导出有离线测试；真实设备、DDS传感器吞吐与工作机编码环境还需现场验收。三路实时预览按约定留待下一阶段。

## Train RTC

新模式“Train RTC · 训练前缀条件”使用训练时 RTC 的硬前缀采样。选择完整 checkpoint，并在“训练记录”提供该次训练保存的运行 JSON；或把记录复制为 checkpoint 根目录的 `train_rtc_record.json`。核对记录中的 `rtc_max_delay > 0`、模型结构和统计后，界面才开放该模式。Paper/LoRA 权重不开放此入口。

选择 Train RTC 后重新启动推理服务；命令对应 `--enable-train-rtc`，执行端对应 `--broker-mode rtc --train-rtc --speed fast`。界面设置承诺前缀预算为 `min(25, rtc_max_delay)`，沿用 current/Home、录制、停止和旧块播放机制。实际推理/安装超过预算则停止。服务必须完成全部静态前缀长度的预热才能连接执行流程。工作机 GPU 与实机验证仍待完成；命令及数据坐标约束见根 README 的 Train RTC 部署模式。


## Train RTC 与 Train Paper RTC（train-rtc 分支）

| 选项 | 训练权重 | 推理机制 |
| --- | --- | --- |
| RTC | 普通全量 Absolute／Delta | 推理时前缀引导 |
| Paper RTC | 普通 Paper Delta | 逐时间步归一化、相关噪声及推理时前缀引导 |
| Train RTC | `rtc_max_delay > 0` 的普通全量权重 | 训练前缀条件，采样全过程固定已承诺前缀 |
| Train Paper RTC | Paper 与 RTC 联合训练权重 | Paper 归一化与相关噪声，加训练前缀条件 |

本分支已合并 `codex/camera-exclusive-startup`。两份已审查的7999分别为：

- `…tairan_tomato_sharpa_1016_2_multimodal_fullft_delta_s42_1031_105906_trainRTC/7999`：选择 **Train RTC**。
- `…tairan_tairan_sharpa_no_block_one_delta_paper_t3_s42_run01_rtc_d10_1031_151412/7999`：选择 **Train Paper RTC**，`timestamp_zscore_arms`、β=0.5。

在UI中选择实际本机checkpoint目录，动作类型保留自动，点击识别权重。内置的精简训练证据位于 `../fr3_sharpa_runtime/training_records/`，通过Orbax manifest、参数元数据和统计文件摘要绑定这两份权重；改目录名不影响识别，其他权重不会仅凭相同统计或“rtc”目录名套用记录。完整复制 `params/` 和 `assets/`；仓库不包含权重。

其他Train RTC权重需在“训练证据”选择对应训练JSON，或在checkpoint根目录放 `train_rtc_record.json`。命令行还可通过 `SHARPA_TRAIN_RTC_RECORD` 指定。联合训练记录中的Paper模式、β、std floor、资产摘要必须与checkpoint一致。识别后仅提供普通串行和对应的训练RTC模式；不会将联合权重误送给普通RTC／Paper RTC。

两份权重均为30Hz、58D、50步，训练最大延迟10步。RTC速度固定Fast；界面的15是**重规划触发节点**，不是训练延迟。当前训练RTC调度固定承诺10步，约333ms；推理返回超出已承诺前缀窗口会停止执行。支持的前缀0..10均在模型服务启动时实际预热；联合模式还检查50／35／1步短参考，预热成功才允许执行。

启动顺序保持“启动推理 → 等待就绪 → 核对执行配置 → 启动执行”。执行自动相机接管与硬件up，Current／Home、录制和停止清理沿用原流程。切换RTC类型需停止旧推理并重新启动。相机SDK镜像和ROS消息overlay仍需按[相机SDK说明](../fr3_sharpa_runtime/CAMERA_SDK.md)在工作机配置。


### 新训练归档与缺少训练记录时的手动入口

部署可读取原始运行 JSON，也可读取新的 `effective_config.json`。自动查找顺序为步骤目录内的 `train_rtc_record.json`、`effective_config.json`、`reproducibility/effective_config.json`，以及步骤目录上一级的 `reproducibility/effective_config.json`，最后匹配内置权重指纹。也可在 GUI 的训练证据中直接选择文件。Paper 配置必须匹配实际 Paper bundle；普通 RTC 的有效配置旁须有配套 `norm_stats.json`。

仅复制 `params/` 和 `assets/` 时，无法单凭参数结构确定是否做过 RTC 前缀条件训练。此时在 GUI 的“训练 RTC 最大延迟”填写实际训练值（例如 `rtc_max_delay=10` 就填 10），点击识别，再选择 Train RTC／Train Paper RTC。0 表示自动识别；正整数表示操作员明确确认训练方式及最大延迟。普通权重还需明确选择 Absolute／Delta。手动设置随预热和执行配置一起传递，进入模型身份与配置核对；与已有训练证据冲突时仍报错。Paper 资产、58D/50步结构、LoRA 限制及原执行检查继续生效。

2026-11-02 的 `…1102_1309_paper_t3`，6000 和 7999 已分别按对应训练 JSON 与 checkpoint 指纹绑定，可直接自动识别为 Train Paper RTC，最大延迟10步。


### Home 阶段的手指限位诊断

Home 目标和从反馈生成的中间指令分别检查。双手每次发送前先同时验证两边的22个关节，避免发现右手非法目标前已发送左手。错误给出侧别、关节、实际角度及原驱动范围；`runtime.log` 的 `prepare_initial_feedback` 保存 Home 开始时真实58D反馈和目标，供追查边界偏差。

若反馈在限位附近，Home 中间指令只能在现有 `prepare_speed_rad_s × 1/30秒` 的单步预算及严格 SDK slew 范围内收敛至合法边界；浮点边界取可表示的区间内值。真实反馈、Home 目标和原驱动限位均不修改。不套用模型预测的0.05rad裁剪额度。若距离合法边界超过该单步预算，Home 在下发前拒绝并报告数值，应检查真实手指位置和标定，不能通过扩大限位强行启动。
