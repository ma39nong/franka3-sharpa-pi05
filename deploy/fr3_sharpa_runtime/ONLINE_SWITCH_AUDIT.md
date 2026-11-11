# 七阶段在线切换：性能与可靠性审计

基线：`edeb45f`，分支 `tomato-subtasks-manual-deploy`。本次只修改部署代码、离线工具、测试和文档；没有修改训练仓库、权重，未启动真实机器人、ROS 硬件或模型服务。

## 结论和变更范围

沿用单执行会话、单控制所有者、普通 Soft-Prefix Guidance。阶段内部仍使用原来的 RTC snapshot / receipt / install、DelayHistory 和 30Hz 时间轴。B Coarse 没有改成 Train RTC。跨阶段首块依然不使用旧 Prompt 的前缀。

确认并修复的具体问题：

| 位置 | 基线问题 | 修改 |
| --- | --- | --- |
| `task_rtc.py:tick / predict`、`observation_worker.py:snapshot` | 版本变化从 `on_poll` 抛异常，`snapshot` 的清理会终止持久观测子进程；后续读取已退出进程失败 | 普通切换只撤销推理/安装授权，继续有界读取当前观测回复，再丢弃过期版本；暂停、停止和真正传感器错误仍保留中断清理 |
| `task_rtc.py:tick` | 新观测排在唯一 policy worker 中，必须等旧 GPU 请求返回才开始；丢弃旧结果又额外返回一轮 | 独立单线程只读观测池与旧推理重叠；完成旧结果同一轮继续调度；仍只有一个 policy worker |
| `task_rtc.py:run`、`task_control.py:command` | 固定10ms睡眠，命令和Future完成不能唤醒 | 操作员意图、观测及推理完成唤醒协调器；保留最多10ms状态检查周期 |
| `task_rtc.py:predict` | 观测和推理各自使用整个预算，未扣除观测耗时 | 推理只用剩余预算，首块同时受200ms snapshot新鲜度限制 |
| `runtime.py:infer_task_chunk` | 原 `response_timeout` 只约束动作 `recv`；连接和metadata各自可等待10秒，关闭可等待5秒 | 任务路径共享总预算，连接、metadata及动作接收使用剩余时间，关闭最多50ms，结束和解码后再次检查；非任务普通RTC默认超时保持原样 |
| `stage_handoff.py:prepare_handoff`、worker preflight | `ceil(delay*30)` 提前最多一个节点，产生永久时间偏移；preflight也比较取整后的邻近目标 | 跨阶段使用浮点phase，使采样等于 `30*(实际采样时间-snapshot时间)`；preflight和执行采用同一个对齐函数；普通RTC整数跳步保持原实现 |
| `stage_handoff.py:source_curve` | 桥接采样和预检循环反复复制50×58数组、计算digest | 切换时冻结一次源轨迹，桥接和快速重定向复用；不复制/延长未接纳动作 |
| `ui.py`、`controller.py:task_status` | 阶段显示依赖1500ms全量状态轮询，包含日志文件读取；命令后又等待全量刷新 | 活动任务250ms轻量状态轮询，命令回复立即显示；日志仍1500ms更新；拒绝较旧stage_version的迟到UI回复 |
| 任务日志和 `switch_audit.py:summarize` | 缺少采集起止、客户端细分耗时，难以区分安排与生效 | 增加完整时间点、对齐信息、实际发送ACK时间、轨迹导数诊断及离线汇总工具 |

观测预取不是放宽新鲜度：等待旧推理过久时会重新观测；使用上一次推理耗时作调度估计，给估计设置上限以免慢请求导致“允许观测年龄=0”的无限刷新。实际执行前、推理后、worker安装时的硬检查继续生效。推理不能在其观测尚未完成时开始。

## 实际调用时间线

| 时间点 | 调用及可观察证据 |
| --- | --- |
| 操作员点击 | `ui.py:taskCommand`，request_id及`client_clicked_unix_ms`；浏览器wall clock只用于诊断，不决定安全预算 |
| HTTP控制器入口 | `controller.py:task_command`写入`http_received_monotonic_ns`，通过独立Unix Socket转发；不等待下一次UI轮询 |
| 执行端收到并登记 | `TaskControl.command`先记录IPC入口时间，再拿安装互斥锁；登记requested_stage、递增stage_version，`stage_requested`保存锁等待时间；此时active_stage不变 |
| 开始重规划 | 协调器醒来，`SWITCH_REQUESTED → SWITCHING`；worker `stream_switch`固定已接纳源轨迹、撤销旧pending，生成epoch/source/token和不可延长的deadline；`online_switch_boundary` |
| 旧请求处理 | 仅未开始的Future可以真正cancel；已运行请求继续占用唯一policy线程，结果/异常因旧ticket失效而丢弃；`switch_policy_wait`、`inference_discarded` |
| 新观测开始/完成 | `observation_start / observation_complete`，可与旧GPU调用重叠；持久ROS进程仍串行处理snapshot，返回三路图像和真实58D反馈；保存captured时间、各topic stamp、source_age/span和ROS编码/IPC解码耗时 |
| 新Prompt推理开始/结束 | 有唯一policy空位、匹配ticket且观测有效后提交；`inference_start / inference_complete`，保存`SharpaRuntime.last_timing`的连接、metadata、msgpack、收发、服务端和适配器耗时；不能把这一段等同GPU采样时间 |
| 交接安排 | `stream_handoff`核对token/epoch/source、实时反馈、限位、slew及速度，验证桥接和候选后续轨迹；`handoff_armed / online_switch_armed`只表示安排成功；回复含bridge_start/end |
| 交接开始 | 唯一worker在计划开始时间附近的30Hz tick采样旧/新绝对轨迹；按真实tick时间进行五次权重过渡；中途每tick继续检查反馈与实际指令差值 |
| 实际接入 | 桥接末tick成功发送SDK命令并发布目标后，worker记录`handoff_applied`及`applied_monotonic_s`；父进程之后读到匹配token/epoch/generation，才`activate`并记录`online_switch_applied` |
| 普通RTC恢复 | 新Chunk消费到配置的重规划触发节点时，重新观测并取得普通RTC snapshot，提交新阶段内部的引导推理；`ordinary_rtc_resumed`标记第一次提交，而非阶段刚生效时就开始引导 |

“实际接入”ACK证明命令流已切入，不证明执行器已到位或果串操作成功。父进程轮询确认时间与worker实际发送时间分开保存。现有`execute_steps=15`按**接纳后消费节点**计时；约6个桥接周期已经消费，通常还需约9个周期再触发普通RTC，具体以时间轴为准。

## 唯一推理线程、取消与最坏情况

模型服务的 `src/openpi/serving/websocket_policy_server.py:_handler`在async handler内同步调用`policy.infer`；没有经验证的GPU任务抢占协议，客户端也共享`last_receipt/last_timing`。本次没有增加GPU并发。

等待旧推理不可被`Future.cancel()`消除：运行中的Future返回False；断开客户端连接也不能证明GPU计算被取消。新观测可提前进行，收益约为原等待与观测耗时中可重叠的部分，必要时刷新过期预取。旧请求超时之后，服务器还可能继续处理旧计算；新连接等待metadata同样受任务总预算约束，不能假定GPU立即空闲。

基线中，旧引导请求的动作回复等待通常受prefix horizon约束（最多25/30秒），但连接和关闭还有独立超时，因此不能给出“总等待只有约60ms”的结论。优化后任务客户端共享总预算，运动worker仍有独立deadline和30Hz watchdog；网络发送、操作系统调度或同步GPU卡住并不具有Python硬实时保证。预算不足时停止，不尝试第二个GPU请求绕过拥塞。

## Chunk相位预算

下表为**worker开始切换时**的绝对model phase，30Hz原速，不包含点击到worker之间已经消耗的时间。50步Chunk剩余时间为`(50-phase)/30`，不是从第0步重新计算。

- 旧轨迹窗口 `W = min(0.75, (50-phase)/30 - 1/30)`。
- 协调器保守观测/推理预算 `B = W - (6+2)/30`，还需扣除已经等待旧Future、IPC和调度的耗时。
- worker最后校验要求 `bridge_start + 6/30 + 1/30 < deadline`；两周期的协调器预留中另一个周期用于安装和调度裕量。

| phase | Chunk剩余ms | W ms | 初始B ms |
| --- | ---: | ---: | ---: |
| 10 | 1333.3 | 750.0 | 483.3 |
| 15 | 1166.7 | 750.0 | 483.3 |
| 25 | 833.3 | 750.0 | 483.3 |
| 35 | 500.0 | 466.7 | 200.0 |
| 40 | 333.3 | 300.0 | 33.3 |
| 45 | 166.7 | 133.3 | -133.3，立即拒绝 |

第40步只有极快的观测及推理才可能接入；第45步不能完成现有6周期桥接。这是当前已接纳轨迹长度的约束，未通过延长0.75秒、保持最后目标、减速或播放失效Future来掩盖。

已研究原RTC时间轴：`RTCTimeline.install`成功即替换为新50步时间轴，切换总是使用**最新已接纳**的那一块，因此成熟RTC在第15步及时完成重规划会降低进入末端的概率。没有第二个“已经接纳但待播放”的队列可以额外取用。点击后旧Future即使已完成也不能重新接纳：它仍属于旧版本；只有在请求登记前完成原子安装的有效Chunk能作为切换源。新增“跨版本安装旧推理”会破坏隔离，本次未启用。

## 观测时效及轨迹质量

200ms有不同检查点，不能混称整个闭环只花200ms：ROS在snapshot选择时检查真实图像/反馈的源年龄与对齐；`captured_monotonic_ns`是snapshot选择完成时刻，后续检查的是该snapshot的年龄。图像源在snapshot之前可能已有年龄；两者应结合`alignment.source_age_ms`分析。观测等待、图像编码、JSON/base64 IPC、客户端校验、连接/metadata、GPU和结果解包、动作检查及安装均可能消耗预算。观测完成后的新首块到worker接纳必须满足原有200ms检查，桥接本身继续推进时间轴并每tick读取真实反馈。

服务端原有`AbsoluteActions`使用本次观测state，把双臂Delta还原为canonical absolute；双手本来就是absolute。部署交接仅在这些绝对目标之间进行，没有再加一次当前state，也没有修改权重或统计。分数phase修复只作用于跨阶段，按真实snapshot时间对齐，桥接末端保留原时间参考。

五次权重端点的一、二阶权重导数为0，但输入轨迹是分段线性的，**这不构成轨迹加速度/jerk已受限的证明**。新增`trajectory_metrics`按30Hz差分记录左右臂/手的速度、加速度、jerk；目前只有已有arm speed、手slew和原硬件限制可作执行门，不凭空设定加速度或jerk阈值。

双手44维中的右手22维仍按模型提供的新旧手部目标交接，没有新增固定抓握姿态、冻结手指或自动张合。逐关节凸插值不保证抓握力、接触或物体稳定性；本次没有足够证据启用手部独立交接策略。模型目标差异大时已有速度/slew/preflight会拒绝，但小关节差值也不能证明不会松果串，必须实机验收。

worker在交接前和每个过渡tick对真实反馈执行原preflight：双臂目标差值每组最多0.05rad、双手每组最多0.15rad，反馈本身必须新鲜；保留每tick实际命令变化检查、手slew严格小于0.10rad、既有关节限位和watchdog。gateway速度在任务启动前从真实参数读取，不写参数；若运行中更改gateway速度，实际gateway保护仍执行，但缓存的交接验证值及跟踪行为也需现场核对。

## 可重复离线对比

```bash
.venv/bin/python -m deploy.fr3_sharpa_runtime.switch_audit \
  --baseline edeb45f --repeats 3 \
  --output deploy/fr3_sharpa_runtime/validation/tomato-switch-audit.json

# 分析后续真实或模拟任务日志；工具不自动认证日志来源
.venv/bin/python -m deploy.fr3_sharpa_runtime.switch_audit \
  --events /绝对路径/logs/task-events.jsonl --output /绝对路径/switch-summary.json
```

工具从Git直接读取基线`TaskRTC`与`StageTimeline`，分别运行于同一模拟backend/control；使用生产协调器和时间轴，模型和观测用可配置sleep模拟，真实OS线程调度。结果是**本机离线模拟中位数，不能当成相机/GPU/机器人闭环实测**。每组3次是回归样本，不是统计安全认证；接入时间受30Hz量化和本机负载影响。完整结果含每组参数、成功数、原因、模型并发数以及源采样开销微基准，见 `validation/tomato-switch-audit.json`。

对比涵盖10/15/25/35/40/45相位、旧Future、观测/推理变慢、过期预取刷新、连续点击、轨迹差异、预算不足。自动化测试额外覆盖七阶段、普通RTC与跨阶段交替、同Prompt不同版本、迟到错误、真实持久观测子进程在切换时保持存活、任务客户端总预算及日志缺失值。

## 已有真实记录查找

已只读搜索本机Desktop/桌面、Documents、Downloads、当前/soft-ui/trex部署仓库的`.deployment`、`desploy_sharpa`及company-remote/mounts目录中常见runtime/RTC trace/task-events记录。找到当前仓库 `.deployment/rtc_audit/rtc_smoke.json`、`paper_7999_audit/rtc_smoke.*`、`tomato-validation/`里的GPU假观测报告和配置检查，以及`control_model_sessions/2026-10-30/01/run.json`模型会话配置。它们不能提供真实相机到机械臂的闭环耗时；Desktop上视频也不能还原缺失的推理/安装时间点。

本次没有找到可确认包含真实机器人RTC完整请求/观测/安装/反馈时间线的日志，**真实闭环P50/P95/P99、实际跟踪误差、抓握力、加速度/jerk均标为缺失**。这不否定原RTC已有真机使用经验，只限定本次可以量化的证据。没有用约60ms的GPU smoke填补这些数据。

## 与train-rtc的差异及实机待验收

与本机`train-rtc`代码逐项比较：原`rtc_runtime.py:run_rtc`仍保留；此前仅增加任务模式拒绝走未版本化入口。原`RTCTimeline`此前仅增加epoch/invalidation检查，本次没有修改`rtc_stream.py`。普通RTC前缀算法、归一化、Delta适配、延迟历史、原硬件owner、录制与失能逻辑均沿用。新增只在七阶段任务协调器及`StageTimeline`工作：版本、观测重叠、无旧前缀首块、有界桥接和实际生效ACK；非任务RTC不创建观测线程池。

隔离环境、具备资质人员仍需验收：

1. 实际HTTP/IPC锁等待、旧policy等待、三路图像源年龄、ROS/编码/网络/GPU/安装各段及闭环P50/P95/P99，尤其高负载时第35/40步；不能承诺末端任意时刻都能无停顿切换。
2. 实际30Hz周期、抖动、丢tick、硬件反馈延迟、跟踪误差、gateway裁剪，以及真实速度/加速度/jerk；模拟不包含机器人动力学。
3. 携带果串时右手的接触/抓握稳定性及跨阶段手部目标差异。缺少经过验证的抓握状态/力约束，不能由位置插值推导物体安全。
4. 超时、断图像、断反馈、非法动作、连续请求和故障清理的物理行为；保持现有限位、watchdog、独占控制。
5. FR3受控静止/保持ACK仍缺失；暂停后的自动恢复、完整停止的整机安全证明仍被阻止，不伪造`arms_safe`。本次没有将原RTC的真机经验替代这项接口能力。


## 本次离线结果与回归

下表为成功次数及成功样本的实际模拟tick接入时间中位数（从执行端请求登记开始），3次/组；失败样本不纳入成功延迟。没有测量真实机械臂。

| 模拟条件 | edeb45f | 优化后 |
| --- | --- | --- |
| phase40：观测20ms、推理60ms | 0/3，拒绝 | 0/3，拒绝 |
| phase45：预算不足 | 0/3，拒绝 | 0/3，拒绝 |
| phase25：旧请求120ms、观测80ms、推理60ms | 3/3，533.0ms | 3/3，432.9ms |
| phase35：旧请求100ms、观测80ms、推理60ms | 0/3，拒绝 | 3/3，399.6ms |
| phase25：旧请求220ms，预取过期后刷新 | 3/3，566.3ms | 3/3，566.3ms |
| phase15：观测150ms、推理60ms | 3/3，466.5ms | 3/3，466.5ms |
| phase15：新推理250ms | 0/3，拒绝 | 0/3，拒绝 |
| phase15：旧推理900ms | 0/3，拒绝 | 0/3，拒绝 |
| phase15：旧请求100ms，35ms时再次切换 | 3/3，466.3ms | 3/3，399.6ms |
| phase15：新旧目标相差0.5rad | 0/3，拒绝 | 0/3，拒绝 |
| phase40：观测1ms、推理5ms | 3/3，266.6ms | 3/3，266.6ms |

其余phase10/15/25/35的常规20ms观测、60ms推理均为3/3，接入约333.2ms，优化没有声称让这类无旧请求场景更快。过期预取会主动刷新，因此长等待场景也不保证提速。所有组最大模型调用并发数都是1。冻结源轨迹采样微基准的中位数为9.30µs → 1.78µs/次；只说明消除了复制和hash，不能外推成系统闭环收益。

自动化验证命令：

```bash
.venv/bin/python -m pytest -q deploy/fr3_sharpa_runtime/tests packages/openpi-client/src/openpi_client/rtc_test.py
.venv/bin/ruff check --select E9,F <本次修改的Python文件>
.venv/bin/python -m compileall -q deploy/fr3_sharpa_runtime deploy/fr3_sharpa_control
```

全量回归已通过 **624 passed**（1328条既有JAX/Flax弃用警告）；最终检查记录保存在忽略目录 `.deployment/tomato-validation/audit-regression.log`。嵌入UI JavaScript语法检查纳入测试；具体日志耗时见该文件末尾。前后对比原始JSON纳入版本控制，重复运行会有OS调度造成的毫秒级差异。

## 实验验收与自动分析工具（2026-11-11）

基于`9f3f3c2`新增的工具只做离线分析，没有修改以上RTC、状态机、桥接时间或硬件控制实现。它复用task/control/终态日志，加入时钟域检查、按版本和token关联、失败/缺失分类、七阶段统计、实际日志间隔的动态量、CSV/JSON及中文报告。

- [分析使用方法及字段定义](EXPERIMENT_ANALYSIS.md)
- [独立实机验收规程](HARDWARE_ACCEPTANCE_ZH.md)
- [明确标记synthetic的完整样例](examples/experiment_acceptance/README.md)

```bash
.venv/bin/python -m deploy.fr3_sharpa_runtime.experiment_analysis \
  /绝对路径/实验归档 --out /绝对路径/报告目录
```

旧`switch_audit --events`接口保留，但现在拒绝无时钟证据的跨生产者时间相减；精确的Click-to-Applied需在实验采集时保留浏览器wall偏移与主机/boot域证据。缺失值不能用模拟数值或GPU smoke替代。
