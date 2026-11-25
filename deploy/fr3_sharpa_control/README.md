# 统一部署控制台与 Chunk 录制

## 自动识别与手动配置（2026-11-24）

选择 checkpoint 步骤目录后点击“自动识别并填入”。控制台读取实际参数结构、配套归一化和 Paper 资产，并尽力读取步骤目录或其父目录 `reproducibility/effective_config.json`、运行 JSON 中的字段。已知字段自动填入模型类型、图像路数、RTC 延迟、执行模式及可识别的任务流程／统一 Prompt；未知项显示提示，可直接手动选择。无需生成训练契约、登记 checkpoint 指纹或提供训练 JSON，手动配置优先于可选记录。历史模型名称只作提示。

无法只从权重区分的 Absolute／Delta、RTC 训练方式、图像路数和任务流程，需要操作员补齐。三路、四路均可手动选择；四路需填写可用的局部图定位器。五阶段、七阶段可配合普通 RTC、Paper RTC、Train RTC、Train Paper RTC，Train RTC 需填写最大延迟。四路 Paper 使用同时保留 Paper 归一化和第四路 RGB 的适配器。

参数维度、归一化数值、Paper 数值资产及噪声矩阵的匹配仍须满足加载器要求。此版本支持的动作接口仍为 30Hz、50×58；手动设置不能把缺失的数值资产或其他模型结构变成可加载模型。本文后面的历史指纹准入、训练证据冲突阻止运行及仅限普通 RTC 的描述由本节替代。

命令行可用 `SHARPA_DEPLOY_IMAGE_INPUT=three|four` 和 `SHARPA_DEPLOY_TASK_MODE=none|tomato-b|tomato-in-hand119` 明确输入与阶段流程，服务端与执行端使用相同环境。模型类型通过 `SHARPA_DEPLOY_MODEL_TYPE` 设置，Train RTC 最大延迟通过 `SHARPA_TRAIN_RTC_MAX_DELAY` 设置。

面向现场操作人员的逐步说明见 [部署 UI 使用手册](../../部署UI使用手册.md)。本页保留模型契约、模式实现与维护细节。

## 工作机启动

推荐在已配置工作机环境、包含所需模式的部署仓库根目录运行，再在UI中选择当前部署目录：

```bash
bash deploy/fr3_sharpa_control/ui.sh
```

沿用现场部署前提：先结束其他遥操和旧执行端，避免设备占用。打开 `http://127.0.0.1:8766`。可用 `--port 其他端口` 更改UI端口；仅监听回环，不对局域网开放。打开页面不连接设备。三终端命令集中在仓库根目录 [部署启动指令.md](../../部署启动指令.md)。

1. 选择本机部署仓库和checkpoint目录，优先自动识别动作类型；没有训练证据时按训练配置选择Absolute或双臂Delta／双手Absolute。点击识别权重，按可用范围选择串行／RTC／Paper RTC／Train RTC／Train Paper RTC模式。全量／LoRA结构仍自动核对，可选训练JSON交叉验证，Paper资产检查必须匹配。
2. 点击**启动推理**。只检查权重并加载GPU、预热、核对服务身份，不启动硬件、相机、录制或机器人执行；即使选择了Home或录制，也不需要此时提供Home配置或ROS环境。推理端预热期间可以点**停止推理**取消。
3. 等推理端显示就绪，选择Current/Home、速度、步数、轮数、录制和输出位置，点击**核对执行配置**。
4. 点击**启动执行**。必须已有与当前仓库、提交、权重和模式匹配的已预热推理端；执行入口只检查/复用该服务，不会隐式加载或切换模型。每次执行都自动运行硬件 `up`，然后按配置开始执行。
5. 点击**停止执行**，等双手失能确认与硬件清理结束。推理端保持运行。可重新选择Current/Home并再次核对、启动执行，仍自动包含 `up`。
6. 不再使用推理时，点击**停止推理**。执行或失能清理期间禁止停止推理；更换权重、模式或代码目录后需重新点击启动推理，本UI仅停止自己拥有的模型进程。外部终端启动的匹配服务可通过启动推理握手复用，停止时仍需回原终端操作。

加载推理、执行及清理期间配置冻结；推理就绪但未执行时可以修改执行配置，无需重载模型。停止不会自动回Home。关闭页面不会停止进程；UI服务器终端Ctrl-C会取消推理加载，或先停止执行并等待既有清理，再关闭自己拥有的推理服务。

### 番茄七阶段在线切换

本分支的 B Coarse 任务在同一次执行中点击阶段按钮切换Prompt，显示“正在切换”，待执行worker实际接入新轨迹后更新当前阶段。普通切换不重启模型或硬件；暂停、停止、任务完成走独立清理路径。交接时限、失败处理、测试和实机验收限制见 [七阶段任务说明](../fr3_sharpa_runtime/TOMATO_TASK.md)。暂停后的恢复仍受FR3物理停止确认能力限制。

活动任务阶段状态每250ms轻量更新；终端日志仍每1500ms更新。性能与时间预算详见[在线切换审计](../fr3_sharpa_runtime/ONLINE_SWITCH_AUDIT.md)。

### 页面布局

宽屏上排依次为权重与代码、执行配置、推理与执行，下方为七阶段任务控制区及并排的硬件／推理／执行日志。页面不再显示三路相机的视频预留框。页面铺满窗口宽度，窄屏自动改为单列；执行配置详情可折叠。

顶栏使用白色背景和用户提供的透明PNG标识，只显示深蓝色图形与“厘清智能”；不显示副标题、标语或动画。源图保存在 `assets/liqing-logo.png`，使用其透明通道以CSS遮罩显示统一深蓝色，不含背景色块。状态提示保留在右侧。

### 三个终端日志

页面中排固定三个同时可见的窗口：**终端一·硬件**（up与硬件清理）、**终端二·推理**（权重检查与持续模型日志）、**终端三·执行**（参数检查、执行、录制与视频导出）。不再按每个内部文件增加日志窗口。三个窗口分别滚动，不互相切换隐藏。推理日志不会因开始新一轮执行而消失，硬件/执行显示最近一轮。

详细文件仍完整保存在磁盘。只启动推理时日志存于仓库 `.deployment/control_model_sessions/日期/小时.分钟/`；不会创建桌面部署记录。实际执行按所选父目录自动创建日期/小时.分钟目录。视频导出在后台进行，不阻塞下一轮执行。

## 当前覆盖范围

### 任务流程、图像输入与训练版本

日常入口直接选择完整 checkpoint 步骤目录，再核对部署方式、模型类型和图像输入。

| 部署方式 | 指令与控制 | 图像输入 |
| --- | --- | --- |
| 统一 Prompt · 不分阶段 | 摘一颗或统一指令摘一串；填写训练时的英文指令，不启用阶段状态机，也不要求Agent | 按实际权重核对三路／四路 |
| 分阶段 · 放回盘中调整 | 原七阶段；手动切换或显式选择Agent自动切换 | 三路／四路分别核对 |
| 分阶段 · 手中调整 | 五阶段；手动切换或显式选择Agent自动切换 | 四路基线及独立核验后的三路对照 |

普通部署的统一Prompt会保存进执行配置并通过`--prompt`原样传给执行端；只改指令或图像输入选择也需要重新核对执行配置。选择不分阶段时隐藏阶段、恢复和Agent控制区。三路输入隐藏ROI配置与预览。默认使用普通不分阶段入口，已有保存的任务选择仍保留。

数据条数不是任务类型；例如手中调整从119条扩充到204条，仍使用同一五阶段顺序及英文指令。

模型还须分别核对三路全图或四路RGB（包含局部图）。同一任务的三路对照不需要第四路ROI输入，不能套用四路模型的加载配置。归一化统计也不单独证明图像路数：204条三路和四路使用相同的状态／动作统计。界面识别权重后分别显示任务、输入、数据规模／版本和checkpoint；只对已有完整指纹的模型显示已核验信息。

历史基线是已有权重的路径预设；选择“手中调整 · 四路 · 119条基线”并点击“填入所选基线”，仍填入原119条20000步权重。也可跳过基线直接选择checkpoint。新训练结果需独立核验后再接入，不能仅改按钮名称或按目录里的数据条数推断兼容。图像输入下拉框是核对条件，不会强行把未适配模型改为四路加载。Agent权重与动作模型的训练版本、输入及控制验收单独核对。

| 权重（均为30Hz、原生58D、50步） | 支持模式 |
| --- | --- |
| 全量 Absolute | 普通串行、RTC |
| 全量普通 Delta（双臂delta／双手absolute） | 普通串行、RTC |
| 全量 Paper Delta v1 | 普通串行、Paper RTC |
| 未合并 Absolute LoRA（VLM16／Expert32配方） | 普通串行 |

这不是通用权重加载器：LoRA Delta、其他动作顺序/维度/频率、未知模型变体、其他论文资产版本尚未接入。模式列表同时受所选仓库和模型限制；旧分支没有Paper适配时会阻止启动。现场硬件验证与离线模型验证分开记录。

Paper通过checkpoint自己的 `delta_paper.json`、`delta_paper.npz`、`norm_metadata.json` 和参数中的噪声状态自动识别；不能用手动选择Delta绕过缺失或冲突的资产。验证资产摘要、30Hz/50×58、双臂Delta掩码、统计范围、噪声矩阵与beta一致性。支持v1的global_quantile、global_zscore_arms、timestamp_zscore_arms，beta范围[0,1)。Paper T3对应的7999为timestamp_zscore_arms、beta0.5；7999只是训练步数，普通Delta的同名目录不能当作Paper权重。

Paper仍使用 `full-delta` 启动入口和端口；服务自动切换至Paper模型与逐时间步变换。普通Delta保留原变换。Paper RTC允许1..50步参考：先补齐50步走完整的绝对值→双臂delta→分位数归一化→Paper变换，再裁回真实参考长度；参考按**新chunk的0..K−1索引**编码。补齐部分不参与RTC引导，输出仍是50步。普通RTC仅用于非Paper权重。默认噪声按训练的Cholesky矩阵生成，显式传入的噪声视为已在目标分布，不重复相关化。推理输出仍为绝对关节位置。

2026-10-30离线验证：7999在RTX5090上通过真实加载、普通推理和RTC引导；输出为有限50×58。未连接机器人，不能替代工作机实机验证。

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

当前部署目录提供普通串行、RTC、Paper RTC和训练RTC模式。部署目录复用启动UI所用的OpenPI Python；Home文件优先选择目标仓库自己的配置，缺少时使用启动UI仓库的工作机配置，ROS overlay也可复用。选择Home时会在启动硬件前解析配置，缺失即报错。

## 启动前部署独占相机

控制台默认采用部署独占接管，局部图定位器使用 GPU。启动执行时，独占接管会中断使用旧相机服务的遥操预览或录制。模型预热阶段不会接管相机。

接管只针对 Docker 中标签为 teleop-cameras／openpi-cameras、命令为 triple_camera.launch.py、且挂载配置的三台序列号及头部/左右语义与本机部署配置完全一致的服务。使用不可变容器 ID 停止，检查容器 Running=false、PID=0，并等待可见USB/网络/驱动占用消失；不按容器名通配停止，不执行remove-orphans、USB reset或杀未知进程。部分相机重叠、无法核对配置或未知进程占用均拒绝启动，并记录容器名、命令、可见PID或设备句柄。操作系统权限可能限制占用诊断，不能将未列出PID当作设备一定空闲。

释放后只启动一个部署相机服务。只读ROS闸门查询驱动 GetDeviceInfo 返回的实际序列号，校验头部cam0／左腕cam2／右腕cam1映射；要求三路有效RGB/BGR、唯一发布者、连续3秒稳定出图（头部至少18Hz、双腕至少25Hz、连续帧间隔不超过150ms），并复用runtime的200ms新鲜度及100ms时间对齐限制。Current模式仍在通过后启动机械臂/网关依赖，并在执行端连接SDK前再次验证。固定Home模式在支持并行启动的部署目录中改用下面的汇合流程。

固定Home并行启动顺序：完成独占接管及设备释放 → 启动唯一部署相机及机械臂/网关 → 执行会话同时启动只读视频验证与SDK连接/关节反馈校验/Home → 两边均成功后，重新获取Home之后的七路新鲜观测 → 推理及任务执行。Home目标不再通过带图像的完整观测获取，而使用原四路关节反馈及命名Home配置；原运动范围、速度、反馈新鲜度和失能确认不变。普通串行、RTC、Paper RTC、Train RTC和Train Paper RTC共用此路径。

固定Home的初始手部边界恢复：仅固定Home请求携带 `home_recovery=True`，在普通Home前调用独立恢复模块。初始反馈合法时直接跳过；小幅越界时，确认两手SDK支持原生插值及速度系数读回，临时把速度系数降到 `min(原值, 0.05)`，将合法目标放在最近边界内侧0.005 rad（窄关节范围按比例缩小）。恢复目标与实测值的差须小于现有SDK 0.10 rad跳变上限；不允许更大越界自动恢复。所有下发命令仍通过原上下限校验，不发送机械臂目标，不开启录制，不修改真实反馈。反馈到合法范围且距恢复目标不超过0.003 rad并连续稳定0.3秒后，读回并恢复原速度系数，再进入原Home。SDK降速设置完成后、尚未发送恢复动作前，最多等5秒，要求四路关节反馈的本地接收时刻均晚于设置完成，且通过原200ms新鲜度和同步检查；再按新实测值重算目标及跳变范围。动作阶段最多10秒，期间不重试失效反馈；无进展2秒、反馈失效、30Hz超时、停止或速度还原失败均阻止任务并走原失能退出。还原速度后，同样在不发送动作的情况下最多等5秒真实新反馈，再进入普通Home。日志新增 `home_recovery_feedback_wait/ready`、反馈源/接收年龄和限频的 `sharpa_feedback_read_error`，不延长200ms阈值。速度系数为无量纲，不代表已验证的rad/s；实际恢复速度和边界需现场验收。Current、策略prepare、串行任务及RTC任务不调用该模块；推理、裁剪、任务安全阈值和SDK配置对象保持原路径。

**行为变化：选择Home并启动执行后，即使视频尚未通过稳定性/序列号检查，臂手也可能已经开始固定Home运动。** 视频失败或超时会通知执行端中止Home，并等待原失能及退出确认；Home失败则取消视频等待，GUI停止同时取消两条流程。只有视频PASS和Home完成汇合后才允许任务推理，视频失败不重启相机或继续使用旧图像。

GUI对已支持的固定Home自动使用 `hardware.sh up --exclusive --defer-camera-check`，其输出的相机就绪状态为 `PENDING`，不代表视频通过；后续必须使用 `runtime.sh --execute --enable-hardware --sharpa-backend sdk_direct --initialization home ...` 的受监督执行入口完成检查。默认不带 `--defer-camera-check` 的硬件命令仍保留原先先检查视频的行为。启动页面和模型预热不会执行Home。

相机锁跨同一用户的部署目录生效，持有至执行、失能确认及清理结束，运行中不允许另一个启动入口抢占。只读相机监控持续检查图像、时间戳和发布者身份；丢图、监控退出或心跳超时会向执行端发送一次SIGINT，等待原有失能与清理流程结束，不会在运动中自动重启相机。恢复需要结束后显式重新启动。未知占用不杀进程；失败时不自动恢复旧遥操服务。

运行中若各路图像仍满足200ms源时间/接收时间新鲜度，但暂时找不到跨度不超过100ms的对齐组合，监控会继续接收新帧并重试，默认恢复窗口为0.3秒。计时从首次失败开始，收到新帧不重置计时；只有重新取得合格对齐观测才清零。窗口到期仍未恢复（或恢复结果到达时已超时）则退出。可在启动UI/执行端前设置 `SHARPA_CAMERA_ALIGNMENT_RECOVERY_SEC` 为0至0.5秒；0恢复原来的单次失败即退出行为。该窗口仅控制监控对短暂对齐抖动的退出判定，不保证运动暂停，也不扩展已有0.75秒健康心跳上限。

等待期间只向stderr记录 `camera_alignment_wait` 及各路时间戳、年龄和历史范围，不发送PASS、不更新健康心跳、不向推理提供无效观测；恢复时记录 `camera_alignment_recovered`。实际模型观测仍须通过原来的100ms对齐与200ms新鲜度限制。时间戳过旧/未来、图像无效、发布者缺失/重复或身份变化仍按原规则立即报错，SDK及机械臂反馈watchdog保持不变。这里没有照搬数采的90秒启动等待或将150ms单路帧间隔当作跨相机对齐阈值。

已知服务正常停止并不保证设备本身没有固件/USB问题；实际序列号或出图闸门不通过时仍会阻止Home。相关信息见hardware.log与runtime.log。所有在GUI中选择的部署目录都必须同步相机闸门代码，旧目录会在配置阶段明确拒绝。

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

勾选录制后，UI在执行前检查ROS bag和视频依赖，提前建立独立录制进程和订阅，但此时不创建bag写入器。Home和初始首帧接近期间不写包。运动worker完成接近后，先取得录制就绪应答，再开始发送第一段任务chunk；RTC在首次stream播放前打开同一个门。

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


## Train RTC 与 Train Paper RTC

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

## 手动配置模型

选择checkpoint步骤目录后，直接选择模型类型：Full Absolute、Full Delta、LoRA Absolute、Paper Delta、Train RTC Absolute/Delta或Train Paper RTC。无需另行生成或复制部署契约；训练JSON为可选核对项，手动模式不自动采用旁边的旧训练记录。原有自动识别保留为兼容选项。

Train RTC需填写实际训练的最大延迟（1..49），可设置初始延迟（1..min(25,训练上限)，0自动）；重规划触发节点仍由执行配置填写。普通RTC无需训练最大延迟。界面仅开放所选模型支持的模式，Paper继续读取原有归一化和噪声资产。权重结构、配套统计、服务预热、关节范围、反馈新鲜度及停止收尾检查保留。

### Agent 实时接入与手动／自动选择（2026-11-12）

启动推理前，可以点击番茄任务面板的阶段按钮选择初始阶段。选中的按钮会显示“初始阶段”。从后续阶段启动时，需要勾选“人工恢复”并核对执行配置。

阶段切换默认手动，也可以提前选择“Agent 自动切换”。预选在当前页面等待，直到指定 Agent 的实时预测和任务执行端都就绪，才向控制台申请启用自动切换。执行端就绪后，仍由操作员点击“开始当前阶段”。已结束的执行会话不会把自动授权带到下一次执行。已有相机和关节反馈时，也可以单独启动 Agent 观察概率。

部署界面使用工作机实时模式。展开“首次配置／修改工作机上的 Agent 路径”可修改仓库、Python、checkpoint、编码器、Gello和ROS overlay。mengxiang默认仓库为`~/vjepa2_training`，权重为`runs/tomato/optimization_20261108/bundles/independent_h15_s42.pt`。相对模型路径以Agent仓库为基准；overlay以部署仓库为基准。Python保留虚拟环境调用路径，不解引用解释器软链接。模型运行于该仓库 `.venv-tomato` 独立进程，编码器只加载一次。ROS 采集运行于现有 Pixi ros2 环境，并复用现有 domain、CycloneDDS 配置和消息 overlay。Agent 不启动相机、硬件、Home 或 SDK，不创建动作发布者。

只读采集保留约3秒历史；每0.5秒按源时间选三路相机各16帧、10Hz，以及58维实测关节位置。相机固定 cam0/cam1/cam2，RGB、256×256、INTER_AREA 与训练一致；关节带完整 `.position` 名称，由模型按名称重排。历史各采样点仅取目标时间之前的原始样本，最大偏差40ms；缺帧、时钟域不一致和过期数据不会伪造补齐。必须预热至少1.5秒且各流持续有效。

Current 的下一合法类概率≥0.8、两次新鲜2Hz观测确认后，协调器通过现有版本化任务 socket 请求切换。Future1.5s 概率只显示，不授权动作。模型SHA、观测ID、会话ID、概率/logits、输入/结果时间都要验证；请求后清空确认，等 active_stage 与 requested_stage 一致且 RUNNING 才再次积累证据。没有另写 RTC 桥接或硬件控制算法。

“切回手动接管”撤销后续自动请求并清空旧证据；已经提交执行端的切换仍按原交接流程完成，接管本身不停止当前运动。需要停止请使用暂停或停止执行。暂停、恢复、人工跳转和新执行会话均回到手动；恢复后须重新选择自动。自动模式下禁止手动推进阶段；停止 Agent、模型/观测异常或任务通信异常会调用原执行停止路径。本机控制服务负责执行通信心跳。页面刷新、关闭或休眠不会停止任务。请使用“停止执行”结束任务。本机控制服务失联时仍会停止。最终 return_stem 绝不自动宣告成功，保留人工完成。

面板显示 Current/Future 概率、确认次数、源观测年龄、模型推理耗时、请求和实际阶段。完整 Agent 日志写在 `.deployment/sharpa-agent-*/`（events.jsonl、model.log、observation.log）；短期NPZ历史只保留最近约10秒。推理耗时不等于完整 ROS→Agent→VLA→执行闭环延迟。B0 已知的 adjust→pickup 提前判断仍存在，本接入不改变分类权重和判定阈值。

现有 CycloneDDS 配置要求系统 `net.core.rmem_max` 至少8388608。仅工作机实时模式检查这个条件；开发机无需修改。若工作机控制台报告不足，请在连接机器人的工作机管理员终端运行（按该机器Gello路径替换）：

```bash
sudo bash /home/user/gello_upper_body_teleop/ops/setup/install_dds_buffers.sh
```

该现有脚本只提高接收缓冲上限并写入重启后保留的配置，不启动设备。修复后重新启动 Agent。软件测试与模拟接入不等于实机采摘验收；真实三路流、模型并发负载、切换时机及采摘结果需现场核验。


### 工作机迁移与内部测试

开发机验证由内部测试与后端工具完成，不进入部署界面。内部自检核对本机资产、Python依赖、CUDA/BF16和离线NPZ；不访问DDS缓冲配置、不启动ROS。自检加载真实B0编码器和分类头，对同一个已有录制NPZ发送3次测试请求，返回内部自检结果。请求使用测试时钟，仅验证模型和进程通信，不能当作在线时序或泛化评估。开发机模式的后端拒绝自动控制，即使伪造就绪状态也不能授权切换。

成功启动会将该机器的配置保存到 `.deployment/agent_settings.json`；刷新页面会回填。配置只保存环境与路径，不保存自动控制授权。不要把开发机的绝对路径配置直接当作工作机配置。可从本目录 `agent_settings.example.json` 开始填相对路径；该模板供内部工具使用，默认开发机；工作机UI提交的profile固定为workstation，无需再选择环境。已有模型推理、RTC和硬件入口沿用原方式。

迁移清单：

1. 将本部署仓库的Agent接入代码和Agent仓库代码部署到工作机；在UI填写两者实际目录。Agent、控制台、机器人执行端需要在同一工作机。当前没有跨机器传图或远程自动切换协议。
2. 在工作机Agent仓库建立独立Python环境。依赖以该仓库的 `requirements-tomato-local.lock` 为已验证版本记录（包含Torch/CUDA组件）；根据工作机驱动准备兼容的CUDA与BF16环境，再在UI选择真实解释器。不要直接复制开发机 `.venv` 的软链接。
3. 复制所选B0 bundle、完整 `pretrained/vjepa2-vitl-fpc64-256/`（包括权重、config和来源元信息）；如要离线自检，再复制一份已授权的真实观测NPZ。模型和数据不在本代码交付包内。记录并对比文件SHA，模型启动会重新核验编码器来源和checkpoint身份。
4. 在工作机选择“工作机 · 实时 ROS 观测”，填写Gello与ROS overlay，点击环境检查；仅在这台工作机修复其报告的DDS配置。启动原部署环境及真实三路相机、关节反馈后，先观察概率，再显式选择自动。

也可在目标机器部署仓库执行只读环境检查，不启动模型或设备（退出码0表示静态资产与依赖检查通过，1表示需修复）：

```bash
.venv/bin/python -m deploy.fr3_sharpa_control.agent_settings --config deploy/fr3_sharpa_control/agent_settings.example.json
```

该命令会在所选Agent Python的独立短进程中导入模型依赖并核对CUDA/BF16；不会安装依赖或下载权重。检查通过不等于真实ROS流或实机动作已验收。

### mengxiang / tairan 双 Agent（2026-11-15，历史配置）

当前 UI 的 `tairan` 使用 119 条四路模型。下文的三路权重、握手配置与旧入口仅供查阅。当前的启动方式见 119 条任务一节。

工作机同步部署代码（先处理工作机自己的未提交修改，避免覆盖现场配置）：

```bash
git fetch origin
git switch main
git pull --ff-only origin main
bash deploy/fr3_sharpa_control/ui.sh --port 8787
```

随后在工作机浏览器打开 `http://127.0.0.1:8787/`。代码同步不包含 `.deployment/` 下的模型、Agent源码快照或本机路径配置；按上面的迁移清单另外准备。tairan 需要交付提交的完整 `source/`、`artifacts/joint/` 六文件和 `encoder/` 三文件。复制编码器时必须复制实际文件或在目标机重建有效链接，不能保留指向开发机的失效绝对链接。填写工作机自己的解释器与路径，检查工作机配置；真实数据就绪后先观察，再授权自动。

材料已导入本机 `.deployment/agents/tairan/`：`source/` 是交付提交 `90ac57ea9eb1de93821ebddeff5a88bb7b0edfc7` 的独立源码快照；`artifacts/joint/` 保存用户提供的6个正式文件；`encoder/` 单独保存同事的来源元信息，两个哈希一致的大文件复用本机编码器。原 mengxiang 仓库、权重、编码器元信息均保留。该目录含模型和运行材料，不纳入 Git；迁移机器时按交接清单拷贝并重新验证，复用的软链接需指向目标机器真实编码器文件。

UI 的“Agent 实现”可选 `mengxiang` / `tairan`，两套各自回填仓库、解释器、checkpoint、编码器和握手配置；切换实现需先停止 Agent。界面不再显示运行环境选项，始终按工作机实时模式提交；手动／自动授权仍独立于Agent选择。内部开发机工具始终禁止自动执行。

| 实现 | 模型 | 切换规则 |
|---|---|---|
| mengxiang | 独立双头 B0，Future+1.5s | Current≥.8/K2 确认下一合法阶段；Future只展示 |
| tairan | 正式 local93 Joint，Future+1s，模型SHA `9324cdb3287c407c259eafbc725ac3bc3d3df6d38646545e8435e0faa9ad03d9` | 保留同事 Future 提议、规划tick、Current ACK、WAIT_LATE_ACK 和总硬截止；下述执行桥接是新增部署约束 |

Tairan 启动先用原 `tools.verify_tairan_deployment` 校验9文件、训练归一化、源码和完整15字段配置，再加载该源码中的 Joint 和共享编码器。使用用户给的元信息原文件，不覆盖 mengxiang 的元信息。在线不重拟合归一化，不混用两套权重、未来时距或判定阈值。`vendor/tairan_handshake_90ac57e.py` 为原策略逐字快照，用于可移植单测；生产加载已校验交付目录中的同一策略。

实时桥接语义必须与离线回放区分：策略 `commit` 只成为下一合法七阶段的请求；执行端的 active_stage/active_stage_version/active_applied_unix_s 确认接入后，才以实际生效时间开始 Current 感知ACK的1.5秒软窗口和总5秒硬截止。在途执行最多等待5秒且不连续推进；安装前的Current观测不能充当安装后的ACK。正常ACK用一次新Current，迟到ACK仍需两次新Current；无新预测时也调用计时器。硬截止或执行版本冲突沿原停止路径退出，需要人工检查再授权。

规划周期/相位沿用导出值，但使用真实结果接收时间与实际软件tick，不伪造 `observed_at + .09`。握手日志保留同事策略事件以及独立 `execution_applied` 事件；该适配会改变相对离线模拟的时钟，不宣称与原离线commit时间逐值一致。最终 return_stem 不再授权 adjust，不自动宣告物理成功。手动接管清除候选和等待证据，已登记的执行请求仍按既有交接完成；下一次自动从执行端实际阶段重新绑定。

开发机自检对两套都使用已有录制NPZ重复3次，验证完整视觉编码器与模型进程；tairan 面板显示 Future+1s 及策略等待状态。它不测试真实机器人成功率，也不代表消除了模型提前判断。正式现场仍需核验相机、关节、时钟、并发负载及动作反馈；跨机器Agent通信尚未实现。

## complete100 四路 r11：手动阶段＋实时局部 RGB

本次目标是让旧100条训练的四路策略在推理时持续收到第四路图，阶段仍由操作员手动切换，无需启动Agent。历史开发分支为 `tomato-four-view-live-crop`，从已接入双Agent的 `197ffca` 开发；现已包含在 `main`，旧分支已删除。

训练权重固定为 `pi05_sharpa_tomato_complete100_b_crop_fullft_delta_r1_offline_r11 / tomato_complete100_b_crop_delta_r1_offline_r11_s42_b256_s16000_c4000 / 16000`。服务按checkpoint自带统计、Orbax manifest和参数元数据的SHA256识别该权重，启用四路及1024视觉token。同次4000/8000/12000也登记四路身份，GPU推理本次只验收16000；未知r11组合会报错，避免误按三路加载。模型输入顺序与训练一致：`base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb, interaction_0_rgb`；文本上限256，58D实测state，50步、30Hz；双臂Delta在输出端还原绝对目标，手部保持Absolute。

**ROI给谁：**第四路当前RGB送给π₀.₅动作模型；本次不改变mengxiang/tairan Agent输入和切换机制。

**ROI谁生成：**独立的Faster R-CNN / MobileNetV3-large-320-FPN，沿用旧数据的原检测模型与难例修复模型。先用原模型；缺少高置信手部时再调用修复模型。在线选区以右手番茄串为锚点，邻近左手加入交互区域；跟随左手转移还需实测手指弯曲及当前RGB共同运动证据。检测与几何启发式不代表确认摘取成功。

**训练和推理的区别：**r11训练坐标使用了后帧辅助修框。实时推理每次从同步快照的原分辨率cam0生成局部RGB，再缩放224；仅使用当前与过去信息，不查离线框、训练阶段真值或未来帧。局部框保留1.5倍上下文及至少0.55倍原图高度；短时光流跟踪只可延续相同关注对象，最多0.75秒。无法验证时回退同一时刻全图，单独记录 `current_full_fallback`，第四路仍available。不会重复旧图片，不能将全图回退算作精准局部图。

这属于为现有四路动作模型补充在线ROI，并非FoveaVLA预测未来框的方法。现有权重没有该预测头，也未训练深度或π₀.₅ Memory。

### 工作机同步与材料

```bash
git fetch origin
git switch main
git pull --ff-only origin main
```

权重和定位器不进入Git。工作机权重应放本地SSD：网络挂载盘逐次文件身份检查实测增加约50–80ms，可能耗尽200ms观测新鲜度余量；不能仅挂载服务器目录代替本地复制。权重从28复制 `16000/params`、`16000/assets`，也可保留原完整checkpoint。28源目录：

```text
/mnt/public_data/local-realrobot-train-data/checkpoints/mengxiang/tomato-openpi-sharpa/pi05_sharpa_tomato_complete100_b_crop_fullft_delta_r1_offline_r11/tomato_complete100_b_crop_delta_r1_offline_r11_s42_b256_s16000_c4000/16000
```

若工作机已配置SSH别名`28`，在部署仓库目录可直接复制：

```bash
roi_ckpt_source=28:/mnt/public_data/local-realrobot-train-data/checkpoints/mengxiang/tomato-openpi-sharpa/pi05_sharpa_tomato_complete100_b_crop_fullft_delta_r1_offline_r11/tomato_complete100_b_crop_delta_r1_offline_r11_s42_b256_s16000_c4000/16000
mkdir -p .deployment/checkpoints/tomato-complete100-r11/16000
rsync -a "$roi_ckpt_source/params" "$roi_ckpt_source/assets" .deployment/checkpoints/tomato-complete100-r11/16000/
# 定位器材料包另行复制到工作机后：
tar -xzf /材料位置/ROI定位器材料.tar.gz -C .deployment
```

UI“四路r11 · 16000配置”的默认本地路径为 `.deployment/checkpoints/tomato-complete100-r11/16000`，可改成工作机实际位置。

定位器材料解压后结构如下，可将整目录复制到工作机 `.deployment/crop_locator`：

```text
crop_locator/
  model.pt             # 修复模型 SHA256 8b06ab9a…
  receipt.json
  primary/
    model.pt           # 原模型 SHA256 19069cbd…
    receipt.json
```

修复模型receipt中的开发机绝对路径不用于加载，依赖固定从 `primary/` 读取。无需同步tomato-openpi训练仓库。Python环境需要现有OpenPI JAX环境，以及torch/torchvision和OpenCV；开发验证使用torch2.7.1、torchvision0.22.1。定位器默认GPU，可在UI选择CPU后重新实测耗时。

### 手动测试

1. 在工作机启动 `bash deploy/fr3_sharpa_control/ui.sh --port 8787`，打开控制台，点击“四路r11 · 16000配置”，填写checkpoint及定位器实际路径。
2. 点“识别权重”“启动推理”，等待四种Prompt与普通RTC预热完成。该按钮不启动相机或机器人。未设置定位器、材料摘要不符时会明确报错。
3. 保持“阶段切换方式＝手动”。核对Current/Home、相机、速度和录制配置后再点“启动执行”；执行端就绪后点“开始当前阶段”。需要切换时点“下一阶段”，界面保留请求阶段与实际生效阶段。无需启动Agent。最终完成仍由人确认。
4. UI显示原图框、模型实际收到的干净第四路图、局部/跟踪/全图回退计数和裁剪耗时。预热零图明确显示“模型预热样本”；超过2秒无查询标记过期。

如果已有相机与关节流，先做只读推理检查：

```bash
# 在工作机终端，按实际路径设置；与UI选择的checkpoint保持一致。
export SHARPA_MODEL_PROFILE=full-delta
export SHARPA_FULL_DELTA_CHECKPOINT=/实际路径/16000
export SHARPA_CROP_LOCATOR=/实际路径/crop_locator
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PWD"
.venv/bin/python -m deploy.fr3_sharpa_runtime.crop_live_check \
  --ros-overlay .deployment/sharpa_ros_messages/install \
  --stage pickup_1 --samples 20 --hz 2 --out .deployment/crop_live_check.json
```

此命令订阅已运行的相机/实测关节，调用本机模型并更新预览，不启动硬件、不发送动作。它报告包含采集和传输后的 `snapshot_to_action_ms` 及200ms内的样本数。若使用其他Python、Gello/ROS路径，沿用工作机原来的环境设置。

离线回放（只用于旧录制接线与定位检查）：

```bash
.venv/bin/python -m deploy.fr3_sharpa_runtime.crop_replay \
  --dataset /旧LeRobot数据集 --episode 4 --stride 15 \
  --out .deployment/crop_validation/replay
```

生成当前框预览视频、逐次记录和热态P50/P95/P99；stride15对应30Hz录制上的2Hz查询。它不预测或执行动作，也不读取离线ROI或阶段标注。

### 已验证与现场边界

验证结果保存在 `deploy/fr3_sharpa_runtime/validation/tomato-complete100-r11-online.json`。GPU加载、原分辨率录制推理、固定噪声仅改变第四路的因果输入检查、普通/RTC及本机WebSocket检查均与实机验收分别报告。旧录制回放中仍有定位失败和全图回退；离线r11的完整局部覆盖不能直接套到在线。开发机与28的延迟不等于工作机延迟，明天需用真实相机/ROS链路确认200ms观测新鲜度和实际裁剪内容。

2026-11-16 实时ROI恢复修订 `causal_hand_recovery_v2`：

- 左右手分别保存可靠检测、各自跟踪。交互从单手变成双手时，不再因组合变化丢掉另一个手的可靠历史。
- 当前全图检测缺少目标时，先在最近1.5秒内的历史位置周围扩大搜索，重新检测当前像素；需要时把当前全图短边从默认320提高到480再次检测。沿用原定位器、修复定位器及原置信度门槛，不引入预测未来框的头。
- 只有当前RGB重新检测成功才能刷新检测期限。光流只允许跨过距真实检测最多0.75秒的丢失，不能自己不断延长期限。会话、时间倒退/重复、超过0.75秒的观测断档、分辨率变化仍重置历史。
- 恢复期间将其他当前可靠可见手保留为窗口上下文；扩窗立即包含证据，缩窗按源时间缓慢收回。裁剪始终来自当前cam0，恢复失败仍明确记录 `current_full_fallback`。
- 控制台分别显示全图检测、恢复、裁剪合计耗时。`status.json`包含 `hand_sources`、`recovery_attempts`、`window_evidence_box` 和 `roi_revision`，便于区分检测、跟踪、当前帧重定位。
- 同一分支更新后，需要重新启动模型服务才会加载修订。阶段继续由操作员手动切换。

本次修复的配对回放结果与真实权重推理检查见 `deploy/fr3_sharpa_runtime/validation/tomato-complete100-r11-recovery-v2.json`。对照使用同一录像、同一实测state、相同查询帧序；离线r11框仅在评分时读取，不提供给在线ROI。标注覆盖是与已有训练标注的一致性指标，不能作为独立定位准确率或实机成功率。

配对测试：三段旧录像原始帧序13,837次查询，全图回退715→0；2Hz稀疏924次查询，全图回退29→2。715个新恢复的逐帧局部框均覆盖至少99%的已有离线标注区域；这些标注来自训练数据，不能视为独立语义准确率。真实16000权重120次录制输入推理及60次WebSocket调用通过；RTX5090回环普通/RTC P95分别102.9/131.2ms，全部通过200/500ms预算，包含ROI、预览写入、动作模型及通信，不含实际ROS/相机采集。689项运行时回归测试通过，生产入口四提示词及RTC预热通过。


### 共享ROI与后续Memory Agent接口（2026-11-16）

当前开发分支为`tomato-four-view-live-crop`。旧B0仍使用三路历史，自动切换授权、Current0.8/K2、Prompt/Chunk与RTC规则不变。新Memory模型必须另行训练和验收，尚未替换默认模型。

四路服务通过`SHARPA_SHARED_ROI_DIR`发布实际送入策略的ROI224；默认计划指向`.deployment/shared_roi`。后台单槽队列只保留最新待写观测，64份记录有界保留；丢弃和写入错误单独记录。每份绑定policy session、原cam0像素SHA、完整ROS topic时间戳、ROI像素SHA和裁剪版本。ROI生成仍在该次策略输入前完成，发布是异步观察通道，不改变动作执行逻辑。

Agent服务握手声明`requires_local_roi`与`control_eligible`。旧B0不要求ROI，新增Memory适配器只观察，不授予自动切换。需要ROI时，Agent sidecar按16个真实10Hz历史时刻精确匹配共享记录；来源、会话或像素不一致以及缺帧均拒绝。**仅按2Hz策略请求产生的裁剪不足以构建10Hz历史；本轮未实现常驻10Hz ROI生产调度，也不复用旧帧补齐。**

Agent仓库`tools.serve_tomato_agent`可自动识别B0或新Memory checkpoint；后者需要四路RGB历史、ROI116元数据、命名58D关节和版本匹配的编码器。r11未来辅助坐标训练的Memory checkpoint仅允许显式离线回放，不能直接当作已验收的实时模型。

离线验证入口为`python -m deploy.fr3_sharpa_runtime.shared_roi_replay_check --dataset DATASET --episode 0 --out NEW_OUTPUT --norm-stats CHECKPOINT_OWNED_NORM`，并设置`SHARPA_CROP_LOCATOR`。读取旧录像和实测状态，调用真实定位器，核对两个并发消费者的像素与时间戳；不包含动作/Memory模型计算，不启动ROS或机器人，不证明端到端并发延迟。

## 119条手中调整：独立五阶段四路部署（2026-11-18）

当前部署分支为 `main`。119条部署最初在 `codex/tomato-in-hand119-deploy` 开发，基于 `tomato-four-view-live-crop/9d5d4d2`；2026-11-20将这条部署线设为 `main`，原分支名已删除。
旧100条七阶段、原英文Prompt及ROI保持原契约。119条使用独立任务模式 `tomato-in-hand119`：

| 阶段ID | 操作员看到的阶段 | 训练英文Prompt |
| --- | --- | --- |
| pickup | 拿起番茄串 | Right hand grasps and lifts the truss to the center |
| harvest_1 | 首次采摘 | Left hand picks tomatoes into left basket while right hand holds truss |
| adjust_in_hand | 手中调整 | Right hand adjusts tomato truss position and orientation at the center |
| harvest_2 | 再次采摘 | Left hand picks tomatoes into left basket while right hand holds truss |
| return_stem | 最终放回 | Right hand returns the held object to the right plate |

“无放回”是中途不放回整串、在手中调整，最终仍有放回动作。内部ID `return_stem` 为兼容保留，
不要求已经摘光或持有物一定是裸杆。两段采摘共用英文Prompt，但拥有不同进度位置。
只更换任务阶段图及Prompt目录，复用现有执行状态机、单一硬件所有者、普通RTC、
在线有时限轨迹交接、版本失效、暂停／停止／清理机制。手动点击不会立即伪报新阶段已生效；
新观测和新Prompt首块经执行端接入才更新实际阶段。最后阶段不自动完成，仍需人工确认。

### 工控机复制与操作

1. 更新并切到上述分支，保持工控机原有 Python／ROS／Gello 环境。
2. 复制一个完整步骤目录的 `params/` 和 `assets/`。本轮已登记新权重10000／15000／20000，
   真实GPU和通信验收使用20000；25000／30000尚未取得，不按名称自动登记。
   默认位置为部署仓库 `.deployment/checkpoints/tomato-in-hand119/20000/`。
   也可以在UI选择工控机的实际位置，不能选只包含多个步骤的父目录。
3. 复制整个 `.deployment/crop_locator/`，包括 `model.pt`、`receipt.json` 和 `primary/`；
   Git不会传输这些模型文件。与100条共用定位器权重，但选框及窗口规则按模型分别选择。
4. 启动 `bash deploy/fr3_sharpa_control/ui.sh --port 8788`，打开 `http://127.0.0.1:8788/`。
5. 直接选择完整 checkpoint，核对仓库及定位器路径。将模型类型设为 Full Delta、图像输入设为四路、
   部署方式设为手中调整五阶段。选择普通 RTC、Fast、15 步触发、连续轮数 0、
   初始阶段“拿起番茄串”，并按本轮需要选择 Current/Home 和录制。
6. 点击“启动推理”，预热就绪后“核对执行配置”。实际机器人启动由操作员现场点击“启动执行”；
   READY后点击“开始当前阶段”。按实物动作完成情况点击“切换到下一阶段”，最终放回后人工完成任务。
   非首次阶段启动需要明确恢复确认；运行中不能直接跳段或回退。

手动切换不会让上一阶段自动停在完成姿态。右手拿稳果串并到达中央、左手刚准备靠近时，
即可切到“首次采摘”，不必等左手到达中央。运行时可按 **N** 请求下一阶段，避免寻找按钮；
输入框、弹窗、自动模式、长按以及尚未完成的阶段切换不会响应快捷键。按键只发送一次人工
请求，实际生效仍以执行端回执为准。 119 条任务收到切换请求后会发完当前已接纳的动作块，随后短暂保持最后一条已发送的命令，从新的观测推理下一阶段，并沿用逐帧限速。停止指令在等待期间仍有效。需要从固定 Home 开始时，另将启动姿态设为 Home，重新核对配置。

在线任务将模型目标限制在 FR3 网关和 Sharpa 驱动的原有关节范围内，并在每一帧从上一条已发送命令
限速接近目标。限速使用网关实际配置和部署已有的手部逐帧变化约束。原始预测仍保存在推理日志中，
执行日志记录目标投影及逐帧调整。生成的命令仍须通过下发前检查，执行故障和停止沿用原失能流程。

本分支的固定 Home 到位容差为 `0.025 rad`，Sharpa 部署层逐帧变化要求小于 `0.25 rad`。
两项设置分别用于判断实测姿态是否到位和约束相邻手部命令，设备关节角度范围由原驱动和网关提供。

仅加载模型、不连接机器人可执行：

```bash
export SHARPA_FULL_DELTA_CHECKPOINT=/工控机实际路径/20000
export SHARPA_CROP_LOCATOR=/工控机部署仓库/.deployment/crop_locator
bash deploy/fr3_sharpa_runtime/serve.sh --model full-delta --smoke-only --enable-rtc \
  --report .deployment/inhand119-20000-smoke.json
```

### 局部图与Agent边界

119条局部图采用训练 `full-block-right-truss-180-v1` 的当前帧目标规则：
右手及附近番茄串为锚，左手移动到篮子不会把局部图带走；640×400参考尺寸下窗口180～300像素、
上下文比例1.12。当前RGB检测与有限过去跟踪可复用已有恢复机制；不调用离线未来插值、
不输入人工阶段真值。缺少可验证右手或目标超过局部窗口时，回退当前时刻全图并明确记录原因。
这是离线未来辅助坐标的因果部署对应方案，不能声称像素与离线局部视频完全一致或定位已获独立验收。

本任务在 UI 选择 `tairan · 四路`，默认使用 `.deployment/agents/inhand119_four`
下的 `head/best_head.pt` 与 `encoder/`。默认手动，可在「阶段切换方式」提前选择「Agent 自动切换」。
启动 Agent 并等实时预测与任务就绪后，预选的自动切换才会生效。tairan 采用 K1，单次观测的当前和 +1 秒预测对下一阶段的概率均达到 80% 时请求切换。当前生效的英文 Prompt 显示在阶段按钮下方。

本轮证据见 `deploy/fr3_sharpa_runtime/validation/in-hand119-20000-online.json`：
真实20000权重恢复、四路输入、四种训练原文Prompt、50×58有限动作、普通RTC预热均通过；
新录制三路RGB＋具名实测关节经真实定位器与GPU模型完成30次本机普通／RTC通信，
这批样本均为当前检测局部图，往返中位数约116ms、P95约131ms。
五阶段真实Unix IPC与生产TaskRTC配合模拟硬件所有者验证，全套运行时回归705项通过，
浏览器核对权重识别、五阶段按钮及执行配置预检通过；未运行ROS、SDK或机器人。
以上耗时不包含工控机真实相机／ROS传输，不是新数据泛化、动作可执行性或采摘成功率。

### 四路模型的录制与四联回放（2026-11-18）

勾选“录制ROSbag及视频”后，三路模型继续原录制流程；已识别的100条／119条四路模型会额外保存
**每次推理实际使用的三路原图、第四路ROI原像素及时间戳**，不是导出时重新定位。
模型服务通过本机响应回传ROI，执行客户端用有界后台队列写无损NPZ，不阻塞等待磁盘写入。
源时间戳、ROI像素SHA、裁剪模式／回退原因、Prompt和任务请求版本保留在各NPZ metadata中。
队列丢帧、像素校验失败或写盘失败会记录为INCOMPLETE／FAILED；不会把缺失ROI当成成功录制。

日志目录新增：

```text
policy_views/                 # 实际四路输入NPZ、写入状态
video/four_view/head.mp4
video/four_view/left_wrist.mp4
video/four_view/right_wrist.mp4
video/four_view/interaction.mp4
video/four_view/four_cameras.mp4   # 头部／左腕；右腕／局部，2×2布局
```

原始ROSbag、`video/head.mp4`、左右腕MP4和`video/three_cameras.mp4`仍保留完整相机流。
第四路是派生的策略输入，按真实推理请求产生，不是独立30Hz相机，也没有新增一个常驻ROS图像话题。
四联与对应四份单路视频采用录制gate的单调时钟，从同一推理样本同步显示四个面板；
30fps只用于播放，在下一份实际输入到来前保持上一帧，画面标注`policy input HELD`及保持时长。
开始gate之前最后一份输入可作为首块的初始画面并单独记数；结束后样本排除。
推理输入记录不等于动作已执行，过时／取消请求可能仍有记录，应按metadata中的任务版本与执行日志核对。
正常停止后等待后台视频导出完成；ROI样本缺失或不完整时导出明确失败，原bag及NPZ保留。

本次用119条20000真实GPU模型和新录制RGB／实测关节完成30次普通／RTC请求，
录下30份四路输入、丢帧0；真实MP4导出得到五个视频，四联1280×720。
采集gate为离线模拟，未运行真实ROS录制或机器人；不宣称全硬件链路已验收。
开启四路记录的本机往返P95约132ms，工控机磁盘、ROS与GPU竞争仍需现场实测。
