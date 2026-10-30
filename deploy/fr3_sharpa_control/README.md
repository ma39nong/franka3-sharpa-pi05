# 统一部署控制台与 Chunk 录制

## 工作机启动

推荐在已配置工作机环境的 main 部署仓库根目录运行，再在UI中选择 main 或 Soft 的部署目录：

```bash
bash deploy/fr3_sharpa_control/ui.sh
```

沿用现场部署前提：先结束其他遥操和旧执行端，避免设备占用。打开 `http://127.0.0.1:8766`。可用 `--port 其他端口` 更改UI端口；仅监听回环，不对局域网开放。打开页面不连接设备。三终端命令集中在仓库根目录 [部署启动指令.md](../../部署启动指令.md)。

1. 选择本机部署仓库目录和直接含 params/、assets/ 的 checkpoint 步骤目录。
2. 点击识别权重，选择执行模式、Current/Home、速度、步数、轮数和录制开关。
3. 点击核对最终配置。修改配置后，或核对后权重/代码身份变化，需要重新核对。
4. 点击启动部署。UI自动检查参数和模型契约、执行硬件 up、启动或复用匹配的已预热服务，然后启动执行端。
5. 点击停止执行，等失能确认与硬件清理结束。再次选择Current/Home并启动时，自动重新包含up。关闭模型服务是独立按钮；更换模式/权重时会重启UI自己拥有的模型服务，不会终止身份不匹配的外部模型进程。

运行期间配置冻结。停止不会自动回Home，Current始终使用重启时新采集的状态和图像。页面关闭不会等于停止执行；请用停止按钮。UI服务器终端Ctrl-C也会先停止执行、等待既有清理，再关闭模型。

## 当前覆盖范围

| 权重（均为30Hz、原生58D、50步） | main | soft-inpainting-test |
| --- | --- | --- |
| 全量 Absolute | 普通串行、RTC | 普通串行、RTC、串行Soft、异步Soft |
| 全量普通 Delta（双臂delta／双手absolute） | 普通串行、RTC | 普通串行、RTC、串行Soft、异步Soft |
| 全量 Paper Delta v1（包括本次7999／T3） | 普通串行、RTC | 普通串行、RTC、串行Soft、异步Soft |
| 未合并 Absolute LoRA（VLM16／Expert32配方） | 普通串行 | 普通串行 |

这不是通用权重加载器：LoRA Delta、其他动作顺序/维度/频率、未知模型变体、其他论文资产版本尚未接入。模式列表同时受所选仓库和模型限制；旧分支没有Paper适配时会阻止启动。现场硬件验证与离线模型验证分开记录。

Paper通过checkpoint自己的 `delta_paper.json`、`delta_paper.npz`、`norm_metadata.json` 和参数中的噪声状态自动识别；不能用手动选择Delta绕过缺失或冲突的资产。验证资产摘要、30Hz/50×58、双臂Delta掩码、统计范围、噪声矩阵与beta一致性。支持v1的global_quantile、global_zscore_arms、timestamp_zscore_arms，beta范围[0,1)。本次7999为timestamp_zscore_arms、beta0.5。

Paper仍使用 `full-delta` 启动入口和端口；服务自动切换至Paper模型与逐时间步变换。普通Delta保留原变换。Soft的旧26..29参考点会按**新chunk的0..3索引**归一化；RTC使用完整50步参考。默认噪声按训练的Cholesky矩阵生成，显式传入的噪声视为已在目标分布，不重复相关化。推理输出仍为绝对关节位置。

2026-10-30离线验证：本次7999在RTX5090上通过真实加载、普通推理、RTC引导、串行Soft四点参考、异步Soft四点参考smoke；输出均为有限50×58。异步Soft仍为50→30→26＋4→20，Soft施加6次；未连接机器人，不能替代工作机实机验证。回归：main283项；Soft422项（16条既有JAX/Flax弃用警告）。模型smoke报告保存在各工作目录忽略Git的 `.deployment/paper_7999_audit/`。

## 权重身份

参数元数据用于核对全量/未合并LoRA和原生58D投影；不能从目录名推断动作语义。当前已核对的AB4 absolute和普通delta统计指纹、完整Paper资产可以自动识别。其他权重优先提供归档训练JSON，需统计SHA256、模型结构、30Hz、horizon50、token256匹配；确实没有可读记录时，可由操作员根据训练配置明确确认语义。该确认按manifest、统计以及Paper算法资产的内容指纹保存，再次选择同一权重自动恢复；路径同名但内容变更不会沿用旧绑定。

未知语义会阻止启动；不会将名字含delta的目录静默当作delta。LoRA delta当前没有新部署配置，会明确拒绝。模型全局参数形状和推理输出还会在GPU预热时核对。5999是训练步骤号，不代表某种执行模式。

## 步数含义

| 模式 | UI设置 | 行为 |
| --- | --- | --- |
| 普通串行 | 可填1..50，如15或20 | 每轮执行对应模型动作前缀后重新推理 |
| RTC | 可填1..24，fast固定 | 后台重规划触发源节点；不是固定执行前缀；轮数0或至少2 |
| 串行Soft | 源执行点26，压缩目标20 | 50→30→26+4，原cubic策略允许条件不满足时回退26 |
| 异步Soft | 固定26→20 | rolling-v3、PCHIP、固定4参考，段边界切换；拒绝旧15点异步协议 |

Main提供普通串行和RTC；Soft分支增加串行Soft和异步Soft。UI读取所选代码目录，只启用其中实际支持的模式。推荐两个分支各保留一个部署目录，在UI选择对应目录，不在运行中git switch。独立目录复用启动UI所用的OpenPI Python；Home文件优先选择目标仓库自己的配置，缺少时使用启动UI仓库的工作机配置，ROS overlay也可复用。选择Home时会在启动硬件前解析配置，缺失即报错。

## 自动录制和归档

勾选录制后，UI在执行前检查ROS bag和视频依赖，提前建立独立录制进程和订阅，但此时不创建bag写入器。Home和初始首帧接近期间不写包。运动worker完成接近后，先取得录制就绪应答，再开始发送第一段任务chunk；RTC/异步Soft在首次stream播放前打开同一个门。

录制从第一段任务开始持续到执行结束或Ctrl-C，不在段间暂停。按停止后立即结束写包；worker正常结束/异常退出也在双手失能确认之前关闭录制，避免把最长120秒的失能等待录进视频。录制收尾错误不能绕过双手失能逻辑。UI等待执行、失能与硬件清理完成后才能再次启动。手失能后是否还能握住物体由实际设备状态决定。

默认父目录按XDG Desktop、已有的“桌面”/“Desktop”自动定位，也可选择其他目录：

```text
<选定父目录>/2026-10-30/01/
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
  model.log                # 本轮新启动模型时
  ui_status.json
  video_export.json
```

每日01、02…采用原子建目录分配，不覆盖既有数据；失败启动也保留该编号和排查记录。不勾录制时不启动bag/video任务，仅保存配置和运行日志。

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

`SHARPA_RECORDINGS_DIR` 或包装的 `--output` 可指定归档父目录；仍自动日期和编号。不要再额外启动旧 `record_rosbag.sh`，也不要在执行命令后接 `| tee`。旧录制脚本仅为历史兼容，新手册使用统一包装。

## 依赖与验证边界

ROS工具沿用工作机Pixi ros2环境、teleop_interfaces overlay、相机和SDK；录制依赖rosbag2_py，导出依赖同一ROS环境中的numpy和cv2（需可用MP4编码器）。可只检查依赖，不创建ROS节点和设备连接：

```bash
bash deploy/fr3_sharpa_control/record_session.sh check
```

本次开发不自动安装或修改工作机SDK/ROS。缺依赖时，勾选录制的启动会在硬件up之前失败并显示日志。界面、模式组合、录制门、编号、停止和合成视频导出有离线测试；真实设备、DDS传感器吞吐与工作机编码环境还需现场验收。三路实时预览按约定留待下一阶段。
