# 番茄 B Coarse：七阶段人工任务与普通 RTC

本分支是 `tomato-subtasks-manual-deploy`，从 `train-rtc/e060393` 开发。训练仓库只读。

## 交付范围与安全限制

任务状态机、当前部署 UI、控制器、Unix Socket、推理和 RTC 执行端已经接线；离线模拟覆盖七阶段完整任务。真实 checkpoint 的四种 Prompt 和普通 RTC 已用 GPU 假观测运行，未启动机器人、SDK 或 ROS 硬件。

**当前实际硬件不能自动连续完成跨阶段切换。** 原 `stream_stop` 只删除时间轴，不能证明机器人保持；Sharpa 指令连续性 watchdog 为 0.25 秒，不能靠停止发送实现暂停。现有能力只有双手失能读回，没有 FR3 安全保持／停止确认接口。因此生产后端跨阶段、暂停和完成任务都先终止执行 worker，等待原有双手失能读回与完整生命周期清理，再报告 `FAULT / safe_to_resume=false`。它不会显示虚假的 `PAUSED` 或 `COMPLETED`，也不会自动重新使能。

使用者必须等待清理、检查实机，再明确授权从指定阶段重新启动新运行。软件重启成功也不证明上一次机器人已安全停止。要启用同一运行中的自动跨阶段接入，必须另行实现并实机验收双臂安全停止 ACK、失能后的合法重新授权，以及整机停止证明；禁止将 `arms_safe` 常量改为真。

离线后端在提供完整整机停止证明时可以走 `PAUSED → RESUMING → RUNNING`，用新三路图像、真实状态和无 RTC 前缀的首块重新开始。该模拟证明的是调度逻辑，不是实机能力。

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

1. 点击“使用本次 B 模型推荐配置”，或选择完整 checkpoint。推荐参数为 Full Delta、普通 RTC、15 节点触发、fast、Current、轮数0、番茄 B 任务。旧保存配置会被保留供兼容使用，推荐按钮显式切换本次配置。
2. 点击“识别权重”，确认 B Coarse 身份、Delta、global_quantile。
3. 点击“启动推理”，等待真实 GPU 预热和身份握手。
4. **只有在操作员确认实机条件并授权后**，核对执行配置，再点击“启动执行”。这会走原有硬件、相机、SDK 和录制启动流程。Home 选项仍使用原有配置，启动执行时可能进行 Home 动作。
5. 任务到 `READY` 后，点击“开始当前阶段”。首块通过观测新鲜度、真实反馈首目标差值、关节范围和 slew 检查后，才显示实际阶段生效。
6. 阶段内部正常 RTC：消费15个模型节点触发后台推理，按实际耗时跳过已消费节点，保留 token／generation／digest／回执检查。
7. 点击下一阶段或阶段按钮：只有相邻下一阶段允许普通推进。请求阶段立即登记，实际阶段不会立即改变。当前硬件会走上述安全阻止和完整清理。
8. 清理结束后人工检查设备，在“任务初始阶段”选定请求阶段，勾选“人工恢复”，重新核对并明确启动新运行。必须重新观测／推理；不使用旧 chunk、prefix 或 Future。恢复确认不会因页面刷新而从保存配置自动勾选。
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

跨阶段不传旧轨迹承诺：安全边界 → 明确重新授权 → 新观测 → 新 Prompt 普通首块 → worker 确认 → active_stage 更新 → 阶段内部 RTC。新边界未确认时禁止自动重使能。首块不使用隐式慢桥接，而是原首目标限值与实际拼接 slew 检查；超出即阻止。

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

仍待实机验收：双臂受控停止／保持与明确ACK、手部失能后的机械行为、重新授权、三路相机与关节反馈时序、30Hz调度和watchdog、实际阶段任务效果，以及完整录制停止与导出。当前软件会阻止依赖尚未确认安全能力的自动续跑。

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
| `deploy/fr3_sharpa_runtime/rtc_stream.py` | task epoch、pending 失效与时间轴撤销 |
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

## 本次实际验证结果（2026-11-10）

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
- 实机运动、物理暂停／保持、重新使能与任务效果验收：**未执行**。真实后端仍明确阻止未获得整机安全确认的续跑，不能把模拟成功或 GPU smoke 当成实机通过。
