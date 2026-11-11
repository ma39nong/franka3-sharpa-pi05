# 番茄 B Coarse：七阶段人工任务与普通 RTC

本分支是 `tomato-subtasks-manual-deploy`，从 `train-rtc/e060393` 开发。训练仓库只读。

本次性能审计、精确调用时间线、相位预算、前后对比和实机待测指标见 [在线切换审计](ONLINE_SWITCH_AUDIT.md)。跨阶段现在保留分数phase进行精确时间补偿；普通RTC整数跳步保持原实现。

## 交付范围与安全限制

任务状态机、当前部署 UI、控制器、Unix Socket、推理和 RTC 执行端已经接线；离线模拟覆盖七阶段完整任务。真实 checkpoint 的四种 Prompt 和普通 RTC 已用 GPU 假观测运行，未启动机器人、SDK 或 ROS 硬件。

普通阶段切换现在使用**同一会话内的在线轨迹交接**，不调用 `stop_boundary()`、不关闭 SDK、不重新加载模型。`SWITCHING` 与 `PAUSING`、`STOPPING`、`FAULT` 是不同路径。

在线交接有硬性截止时间。当前已接纳轨迹仅在最多0.75秒、且不超过旧50步剩余时间（预留一个30Hz周期）的窗口内继续。执行线程清除旧 RTC pending、递增任务 epoch 和 generation，使用独立 token 绑定目标；连续点击不能延长尚未完成交接的原截止时间。新阶段用最新三路图像和58D实测关节重新推理，**不带旧 RTC 前缀**。

收到新首块后按观测时间补偿节点索引，将旧轨迹与时间对齐的新轨迹通过6个30Hz间隔（0.2秒）的五次平滑权重交接。实际30Hz tick保持同一发布者、同一SDK连接和同一watchdog。新轨迹不从0重新计时；跨阶段Delta始终先由本次观测还原为绝对目标，再进入绝对轨迹交接。

执行端在SDK连接前读取真实 FR3 gateway 的 `max_joint_speed`，读不到合法数值就拒绝启动在线任务。桥接及候选后续轨迹必须通过既有关节限位和手部严格小于0.20rad的slew。双臂速度每tick上限用于检查实际桥接入口和桥接段；候选后续轨迹的速度仅记录为 `continuation_arm_speed_rad_s` 和 `continuation_exceeds_gateway_speed`，不因可能被普通RTC替换的后缀提前拒绝整个切换。实际桥接tick再次核对实时反馈新鲜度和实际指令变化，普通执行继续由原FR3网关限制实际输出速度。在线切换不额外套用启动首目标与实测位置的偏差阈值。缺乏预算、反馈异常、推理超时、越界或实际拼接检查失败都进入故障停止。未关闭watchdog，也没有伪造 `arms_safe`。

**命令连续性不等于物理停止／保持证明，也不等于任务阶段成功验收。** 暂停和完整停止继续采用已有失能读回及硬件清理。当前没有双臂安全静止ACK，故真实暂停后仍禁止静默重新使能，完整停止或完成任务也不能虚假报告整机安全已确认。需要静止保持／暂停后恢复时，必须先补齐并验收这项硬件能力。

同一会话完成七阶段在线切换已在模拟中验证；未运行真实机器人。上线前须由具备资质的人员在隔离环境验收本文末尾列出的项目。

## 模型与资产

最终 checkpoint：

```text
sftp://mengxiang@47.116.73.163:27/home/mengxiang/workspace/code/tomato-openpi-sharpa/checkpoints/pi05_sharpa_tomato_truss_normal93_20261109_annotation_r2_all_sources_v2_b_coarse_fullft_v1_delta/mengxiang_tomato_truss_normal93_20261109_annotation_r2_all_sources_v2_b_coarse_v1_s42_delta_1109_134531/7999
```

本机推荐目录：`/home/user/sharpa-pi05/.deployment/checkpoints/tomato-b-coarse/7999`。复制完整 `params/` 和 `assets/` 后才能加载，不需要 `train_state/`。权重与运行日志由 `.deployment/` 忽略，不能提交 Git。

通过以下三份文件的 SHA-256 联合识别本次模型，不使用目录名猜测身份：

| 文件 | SHA-256 |
| --- | --- |
| `params/manifest.ocdbt` | `55d037b0002ca3a76112ae848944663c3b602eac8c9a813847f67445050bdf60` |
| `params/_METADATA` | `78df4cd15c8e0c09c250299c846da7504ec5da44b1463df3757db647beea4df0` |
| checkpoint 自带 `norm_stats.json` | `3f82a27ba53cb6a0553886d453c8e05ccb466354d131ee9c956cd0a12637350a` |

这是原生 π0.5 FullFT、58D、50步、30Hz、256 token，普通全局分位数归一化。动作顺序仍为左臂7／左手22／右臂7／右手22。仅 `[0:7]` 和 `[29:36]` 在本块观测关节参考状态上还原 Delta，双手始终 Absolute。RTC 前缀保留 canonical absolute 表示，经原有适配器在新观测上重锚定。未使用 Train RTC、Train Paper RTC 或 Paper T3。

归档 `reproducibility/norm_stats.json` 的 `openpi.shared.normalize.NormStats` type/fields 包装可被准确解开后与 checkpoint 数值逐项严格比较。没有更改数值，也没有放宽容差。

## 阶段契约

| 阶段 ID | 中文 | 类别 |
| --- | --- | --- |
| pickup_1 | 首次拿起 | pickup |
| harvest_1 | 首次采摘 | harvest |
| return_truss | 放回整串 | return |
| adjust | 调整 | adjust |
| pickup_2 | 再次拿起 | pickup |
| harvest_2 | 再次采摘 | harvest |
| return_stem | 放回裸杆 | return |

```text
pickup: Right hand grasps and lifts the truss to the center
harvest: Right hand holds truss while left hand picks tomatoes into left basket
return: Right hand returns the held object to the right plate
adjust: Right fingers reposition the truss on the right plate
```

阶段名与序号不输入模型。两轮同类阶段共用原文，但拥有不同进度位置；切换始终递增版本。没有时间、chunk 数或模型输出触发的自动阶段推进。进入第七段不自动完成任务。“已离开”标记只表示人工推进，不表示模型成功验收。

## UI 启动与操作

仅打开 UI 不连接设备：

```bash
cd /home/user/sharpa-pi05
bash deploy/fr3_sharpa_control/ui.sh --port 8766
```

浏览器打开 `http://127.0.0.1:8766`。如已有服务占用端口，可用 `--port 8786`。

固定 Home 与相机启动可以并行；Home 完成后仍须等待相机监督进程明确返回 PASS，
再获取首次实时观测，全部成功后才进入 READY。首次观测使用观测 worker 的启动等待预算
（15秒，进程响应另留10秒），后续任务观测继续使用原有时限；相机启动失败或通道关闭会终止任务。

临时本机选项 `SHARPA_TASK_SKIP_FIRST_TARGET_CHECK=1` 可跳过七阶段任务启动时的首帧偏差拒绝，
改用普通 RTC 现有的低速首帧衔接，速度由 `prepare_speed_rad_s` 控制，接入应答最多等待180秒。
关节范围、逐帧变化、实时反馈和watchdog检查仍生效。未设置该选项或设为0时恢复严格首帧检查。
低速首次衔接等待不占用任务状态锁，UI状态与5秒租约可以正常续租；停止、暂停或阶段变更会
中断本次首帧接入并进入退出清理，不会把已失效的首块标记为生效。租约中断记录为
`lease_interrupt`，包含最后联系间隔与触发状态，便于区分主动退出和录制器先行失败。

1. 点击“使用本次 B 模型推荐配置”，或选择完整 checkpoint。推荐参数为 Full Delta、普通 RTC、15 节点触发、fast、Current、轮数0、番茄 B 任务。旧保存配置会被保留供兼容使用，推荐按钮显式切换本次配置。
2. 点击“识别权重”，确认 B Coarse 身份、Delta、global_quantile。
3. 点击“启动推理”，等待真实 GPU 预热和身份握手。
4. **只有在操作员确认实机条件并授权后**，核对执行配置，再点击“启动执行”。这会走原有硬件、相机、SDK 和录制启动流程。Home 选项仍使用原有配置，启动执行时可能进行 Home 动作。
5. 任务到 `READY` 后，点击“开始当前阶段”。首块通过观测新鲜度、真实反馈首目标差值、关节范围和 slew 检查后，才显示实际阶段生效。
6. 阶段内部正常 RTC：消费15个模型节点触发后台推理，按实际耗时跳过已消费节点，保留 token／generation／digest／回执检查。
7. 点击下一阶段或阶段按钮：只有相邻下一阶段允许普通推进。请求阶段立即登记，界面显示“正在切换”；完成有时限的在线轨迹交接后才更新实际阶段，不关闭推理或硬件。
8. 普通在线切换无需重新授权或重新部署。跳转／回退必须先独立暂停；若进入故障或已经失能，则等待清理，人工检查后在“任务初始阶段”选定阶段并明确重新授权启动。恢复确认不会因页面刷新自动勾选。
9. “停止执行”保留原有 Ctrl-C、双手失能读回、硬件清理、录制收尾与导出。它不等待普通任务切换请求，不是硬件急停。

页面刷新只读取执行进程实际状态，不重新开始任务。任务的 `/api/status` 轮询通过 IPC 续租，独立租约监控也覆盖启动与阻塞 Home，失联超过5秒会中断所有者并停止；普通无状态机部署保持旧行为。浏览器被系统长时间挂起也可能触发此边界，不应依赖后台页面无限维持运动。

## 独立推理端

以下命令仅加载模型，不启动机器人：

```bash
cd /home/user/sharpa-pi05
export SHARPA_FULL_DELTA_CHECKPOINT=/home/user/sharpa-pi05/.deployment/checkpoints/tomato-b-coarse/7999
export SHARPA_DEPLOY_MODEL_TYPE=full-delta
bash deploy/fr3_sharpa_runtime/serve.sh --model full-delta --enable-rtc
```

普通 RTC 端口为 `18767`。UI 可以通过身份握手使用外部推理端，但不会关闭它不拥有的进程。

仅检查结构或 GPU 假观测：

```bash
bash deploy/fr3_sharpa_runtime/serve.sh --model full-delta --check-only
bash deploy/fr3_sharpa_runtime/serve.sh --model full-delta --enable-rtc --smoke-only --report /tmp/tomato-b-gpu.json
```

任务执行请使用 UI 创建每次运行独立的私有 socket 与身份；直接传 `--task-mode tomato-b` 而未创建控制通道会被阻止。不要手工伪造 socket 身份以绕过部署方案核对。

可离线检查执行参数，不连接硬件：

```bash
bash deploy/fr3_sharpa_runtime/runtime.sh --model full-delta --check \
  --broker-mode rtc --task-mode tomato-b --initial-stage pickup_1 \
  --initialization current --speed fast --execute-steps 15 --num-chunks 0
```

## 运行真值、并发与边界

`TaskControl` 由执行进程持有。UI HTTP 接口保留 Host／Origin／控制令牌检查；控制器转发到每次运行 `/tmp/sharpa-task-*/task.sock`，父目录0700、socket0600，并检查 Linux peer UID 和 run_id。录制 socket 不复用。每个请求有 request_id 与预期版本，重复同请求幂等，冲突拒绝。

IPC 线程只登记意图，不调用设备 SDK。设备仍由执行主线程和既有 SDK worker 唯一持有。推理线程只做观测／模型请求。结果接纳与意图登记在同一状态锁上线性排序；推理期间不持锁，停止可及时登记。已运行的 GPU 请求不会假装被取消，旧结果即使晚返回也会被丢弃。

执行状态：IDLE、READY、RUNNING、SWITCH_REQUESTED、SWITCHING、PAUSING、PAUSED、RESUMING、STOPPING、STOPPED、COMPLETED、FAULT；合法转换表在 `task_control.py`。版本、请求 ID、运行身份、阶段、可安装状态均必须匹配，worker 还检查任务 epoch 与原有 RTC token、source generation、digest 和延迟。安全重置清空 pending context 并使时间轴失效。

在线切换：登记新版本 → worker 限定过渡窗口并撤销旧回执 → 丢弃旧 Future → 新观测与新 Prompt 普通首块 → 预验证桥接 → 实际30Hz执行 → worker 的 applied_token 确认 → active_stage 更新 → 阶段内部 RTC。`stream_handoff` 的 accepted 仅代表桥接已排定，不代表已经生效；只有实际tick通过检查并完成桥接才给出 applied_token。

通信、反馈、推理、RTC 超时、动作非法、旧 chunk 耗尽、日志写入失败都会撤销执行权并进入停止／故障处理。不能将日志写失败后的半个阶段请求继续用于旧时间轴。

## 日志与排查

保留 UI 中原有硬件／推理／执行三类日志及录制导出。每次执行在所选归档父目录的 `日期/小时.分钟[-序号]/` 中保存：

- `run.json`：审批过的仓库、提交、模型指纹、参数。
- `runtime.log`、`hardware.log`、录制和导出日志：原部署生命周期。
- `logs/task-events.jsonl`：人工请求、替代、版本、状态转换、Prompt、推理起止、观测状态、完整动作、接纳与丢弃、安全停止证明。
- `logs/control-*.jsonl`：30Hz 实际命令、测量反馈、generation、task_epoch、安装／跳过节点、最后失效边界。
- `logs/task-run.json`、`logs/task-final.json`：最终执行进程状态；归档文件仅用于审计，不用作运行控制。
- `logs/shutdown-*.json`：原有双手失能读回结果，不等于双臂安全停止证明。

按 run_id、stage_version、inference_request_id、chunk_id 及 monotonic 时间关联阶段请求、旧动作最后节点、新推理和实际安装。推理服务日志位于仓库 `.deployment/control_model_sessions/`；没有在日志中记录 HTTP 控制令牌。

## 自动化验收

不连接真实机器人：

```bash
.venv/bin/python -m pytest -q deploy/fr3_sharpa_runtime/tests/test_tomato_task.py
.venv/bin/python -m pytest -q deploy/fr3_sharpa_runtime/tests packages/openpi-client/src/openpi_client/rtc_test.py
.venv/bin/ruff check --select E9,F deploy/fr3_sharpa_runtime/task*.py deploy/fr3_sharpa_runtime/tomato_model.py deploy/fr3_sharpa_runtime/tests/test_tomato_task.py deploy/fr3_sharpa_control/{ui,controller,plan}.py
```

新增测试包含独立执行进程的真实 Unix Socket、现有 HTTP UI 的鉴权与命令转发、幂等、刷新恢复、快速点击、七阶段完整模拟、同类 Prompt 不混淆、旧 Future／回执／epoch 拒绝、暂停恢复新观测、超时、推理错误、观测过期、耗尽、失能失败、租约丢失、审计写入失败和严格归档统计解包。其他既有测试继续覆盖 Delta 还原、限位、watchdog、失能确认、serial、普通 RTC、Train RTC 回归、录制与导出。

GPU 实测报告在 `validation/tomato-b-7999-gpu.json`：27服务器 GPU0（RTX PRO 5000 72GB），使用本分支的部署源码临时副本、原始7999和自身统计。四类普通首块约51–52ms、普通RTC引导约72ms；均为有限 `(50,58)`。源码副本在 `/tmp/sharpa-tomato-gpu-47Trn40h`，未修改远程训练仓库。这不是包含真实摄像头、IPC、网络和硬件的闭环延迟结论。

仍待实机验收：双臂受控停止／保持与明确ACK、手部失能后的机械行为、重新授权、三路相机与关节反馈时序、30Hz调度和watchdog、实际阶段任务效果，以及完整录制停止与导出。在线交接的软件检查不代替实机验收；暂停后的重使能和物理安全停止确认仍保留安全阻止。

## 本次变更文件清单

所有路径相对 `/home/user/sharpa-pi05`：

| 文件 | 用途 |
| --- | --- |
| `deploy/fr3_sharpa_runtime/task_stages.py` | 七阶段、中文标签、四条训练原文 |
| `deploy/fr3_sharpa_runtime/task_control.py` | 状态真值、合法转换、版本、私有 IPC、租约 |
| `deploy/fr3_sharpa_runtime/task_rtc.py` | 任务 RTC、推理隔离、结果接纳、安全边界 |
| `deploy/fr3_sharpa_runtime/task_execution.py` | 实际执行入口、SDK 所有者、Home、清理 |
| `deploy/fr3_sharpa_runtime/tomato_model.py` | 本次 checkpoint 联合指纹与身份 |
| `deploy/fr3_sharpa_control/ui.py` | 当前 UI、七阶段控制区、状态和推荐配置 |
| `deploy/fr3_sharpa_control/controller.py` | HTTP 到执行进程 IPC、状态和生命周期 |
| `deploy/fr3_sharpa_control/plan.py` | 审批方案中的任务模式、初始阶段、人工恢复 |
| `deploy/fr3_sharpa_runtime/run.py` | 命令行与版本化任务入口 |
| `deploy/fr3_sharpa_runtime/rtc_runtime.py` | 拒绝任务误入旧协调器 |
| `deploy/fr3_sharpa_runtime/rtc_stream.py` | task epoch、pending 失效与时间轴撤销（普通RTC保持原实现） |
| `deploy/fr3_sharpa_runtime/stage_handoff.py` | 有时限在线交接、纯时间轨迹、五次平滑桥接 |
| `deploy/fr3_sharpa_runtime/tests/test_stage_handoff.py` | 轨迹边界、时钟、越界、过渡超时与gateway读取测试 |
| `deploy/fr3_sharpa_runtime/sdk_direct_motion_ros.py` | 严格首块接入、epoch 检查、最终失效事件 |
| `deploy/fr3_sharpa_runtime/motion_worker.py` | 有界回复等待、保留真实失能读回报告 |
| `deploy/fr3_sharpa_runtime/model_loader.py` | B 身份与 Full Delta 类型一致性 |
| `deploy/fr3_sharpa_runtime/serve.py` | 四 Prompt GPU smoke 与普通 RTC 预热 |
| `deploy/fr3_sharpa_runtime/train_rtc_contract.py` | 训练归档 NormStats 包装兼容，严格数值校验 |
| `deploy/fr3_sharpa_runtime/tests/test_tomato_task.py` | 本次无硬件自动化验收 |
| `deploy/fr3_sharpa_runtime/validation/tomato-b-7999-gpu.json` | 真实远程 GPU 假观测结果 |
| `deploy/fr3_sharpa_runtime/validation/tomato-b-7999-gpu-local.json` | 本机 RTX 5090 假观测结果 |
| `deploy/fr3_sharpa_runtime/TOMATO_TASK.md` | 本说明及验收边界 |
| `README.md` | 任务文档入口与当前限制 |
| `部署UI使用手册.md` | 任务入口与安全限制 |
| `部署启动指令.md` | 任务入口与安全限制 |

## 初版停止边界的历史验证（2026-11-10）

- 新增任务测试：`pytest -q deploy/fr3_sharpa_runtime/tests/test_tomato_task.py`，最终 **41 passed**。
- 最终相关全量回归：`pytest -q deploy/fr3_sharpa_runtime/tests packages/openpi-client/src/openpi_client/rtc_test.py`，**565 passed**，1328条既有 JAX/Flax 弃用警告，没有失败。
- 开发期间还运行了任务＋部署中心＋失能确认联合测试（101 passed）、任务＋归档识别＋手动模型选择联合测试（66 passed）、任务＋失能确认联合测试（65 passed）；这些是当时测试数量，最终回归已包含后来新增的全部测试。
- `ruff check --select E9,F` 对所有新增和修改 Python 文件通过；`compileall` 通过；嵌入 UI JavaScript 经 Node `--check` 通过并纳入新增测试。
- `serve.sh --model full-delta --check-only`、`runtime.sh --model full-delta --check --broker-mode rtc --task-mode tomato-b --initial-stage pickup_1 --initialization current --speed fast --execute-steps 15 --num-chunks 0` 均退出0。
- 27服务器原始权重 GPU 假观测通过，报告为 `validation/tomato-b-7999-gpu.json`，普通 RTC 约72ms。
- 本机权重已经完整同步到推荐目录。`rsync -anic --no-group` 对整个 `params/` 内容逐文件校验匹配，仅原 SCP 文件的时间戳不同，无文件内容或大小差异；本机 Orbax 完整恢复并 GPU 推理通过。
- 本机 RTX 5090 四种 Prompt 的有限 `(50,58)` 输出约46–48ms，普通 RTC 引导约61.5ms，报告为新增文件 `deploy/fr3_sharpa_runtime/validation/tomato-b-7999-gpu-local.json`。不包含相机、IPC和硬件的闭环耗时。
- 本机详细原始日志：`.deployment/tomato-validation/pytest-final.log`、`task-final.log`、`checkpoint-check.json`、`runtime-check.json`、`gpu-smoke-local.log`、`gpu-smoke-local.json`、`params-checksum-diff.txt`。
- 真实 UI 已在本机打开，验证推荐配置、模型识别、七阶段按钮、固定指令展开与未启动时的禁用状态。打开和检查 UI 未启动设备。
- 实机运动、物理暂停／保持、重新使能与任务效果验收：**未执行**。此处为初版停止边界的历史验收；当前在线交接仍需单独实机验收，不能把模拟成功或 GPU smoke 当成实机通过。


## 在线连续切换验证（2026-11-10，基于 d9b8726）

- 完整离线回归：`.venv/bin/python -m pytest -q deploy/fr3_sharpa_runtime/tests packages/openpi-client/src/openpi_client/rtc_test.py`，**600 passed**，1328条既有 JAX/Flax 弃用警告，43.09秒。原始结果位于忽略目录 `.deployment/tomato-validation/online-regression.log`。
- 模拟设备使用生产 `StageTimeline`，按30Hz消费实际命令；覆盖一次启动完成七阶段、正确四Prompt、同Prompt不同阶段隔离、旧Future晚返回、块中途切换、桥接端点和时间补偿、连续点击（包括桥接中再次请求且不延长截止时间）、错误接纳回执、超时、反馈错误、越界、失败清理及暂停独立授权。既有普通RTC、serial、IPC、UI、录制、日志和模型测试同时通过。
- 本次不改变模型、归一化和Delta还原；保留此前两份GPU验证报告，没有将模拟推理称为新增GPU或实机验证。
- `active_stage` 只在worker实际30Hz tick成功发送命令、发布目标且报告匹配的token/epoch/generation后更新。这个ACK说明新轨迹已接入命令流，**不证明执行器已到位或物理任务已成功**。

### 隔离环境实机验收（须由具备资质的人员执行）

1. 先以受限工况确认FR3 gateway真实参数可读、单一控制所有者、硬件限位、接触保护、SDK slew和watchdog均有效；验证反馈丢失、超时及调度超期的真实停止行为。
2. 测量三路新图像、真实58D反馈、推理与IPC端到端时序；验证200ms观测新鲜度及0.75秒切换总窗口在现场负载下成立，超时不能继续播放旧轨迹。
3. 从无负载、低风险工况逐步验收桥接，再验收携带果串、接触及抓握状态；记录实际关节速度、加速度、jerk、跟踪误差及gateway裁剪。五次权重和平滑命令不构成机械臂动态安全认证。
4. 在块中段、末端和快速连续点击时检查旧Future丢弃、阶段显示、普通RTC恢复、同一SDK/ROS会话与连续录制；实测七阶段作业效果。
5. 补齐并验收FR3双臂受控停止／保持及明确物理ACK、手部失能后的机械行为，再验收暂停后恢复。当前接口只有命令发布与反馈，缺少可验证的物理静止／保持确认，因此保留 `arms_safe=false` 和故障/暂停恢复阻止；停止、任务完成均不能被描述为已确认整机安全。

开发和上述600项测试均未启动真实机器人。
