# 七阶段实验日志自动分析工具

开发基线：`9f3f3c2794a44f201375ad77a6d11fdabfcb23b1`，分支 `tomato-subtasks-manual-deploy`。

本次只增加离线分析、测试、文档和合成样例，未修改RTC、状态机、桥接周期、模型、限位、watchdog、SDK或硬件控制算法。分析入口没有ROS、SDK或模型导入，不连接机器人。实机验收的独立文件见 [实机验收文档](HARDWARE_ACCEPTANCE_ZH.md)。

## 使用方法

在仓库根目录：

```bash
# 单次归档：可以传归档根目录、logs目录或task-events.jsonl
.venv/bin/python -m deploy.fr3_sharpa_runtime.experiment_analysis \
  /绝对路径/实验归档 --out /绝对路径/实验分析结果

# 批量：递归发现task-events.jsonl，也可以提供多个目录/文件
.venv/bin/python -m deploy.fr3_sharpa_runtime.experiment_analysis \
  /绝对路径/实验A /绝对路径/实验B --out /绝对路径/批量报告

# 现有switch_audit的新增报告入口；不会运行benchmark
.venv/bin/python -m deploy.fr3_sharpa_runtime.switch_audit \
  --experiment /绝对路径/实验归档 --output /绝对路径/实验分析结果

# 保留旧的单JSON摘要接口；同样使用新的时钟检查
.venv/bin/python -m deploy.fr3_sharpa_runtime.switch_audit \
  --events /绝对路径/logs/task-events.jsonl --output /绝对路径/摘要.json
```

自动读取每个`task-events.jsonl`相邻的`control-*.jsonl`及`task-run.json`、`task-final.json`（快照也可位于上一级）。自动查找同目录或上一级的`analysis-manifest.json`，或用`--manifest /路径/analysis-manifest.json`显式指定。批量分析不同机器/不同启动实例时，应分别保留清单；不要用一个全局清单错误覆盖不同归档的时钟域。

工具只读取输入，输出为新建或已有报告目录中的固定报告文件；重跑会覆盖该目录内的同名报告，不修改原日志。逐行解析JSONL并记录校验摘要，截断行会标记为质量问题。为计算精确经验分位数和导数，单次归档的有效时序保留在内存；长时间高频归档建议分批分析。

## 数据来源与时钟清单

旧日志没有足够的主机/启动时钟身份。数字接近、文件相邻、文件名相似，都不能证明不同进程或机器的monotonic可比。默认仅计算同一文件连续时钟段内的时间差；跨文件、ROS复制时间、worker复制时间和浏览器点击相关指标无法证明时保持缺失。

建议从这个**保守模板**开始，不要直接套用样例里的虚构时钟：

```json
{
  "schema_version": 1,
  "source_kind": "unknown",
  "clock_domains": {},
  "embedded_clock_domains": {},
  "browser_wall_clock": {
    "verified": false,
    "offset_to_task_ms": null,
    "uncertainty_ms": null
  },
  "max_frame_gap_s": 0.1,
  "notes": "填写本次实验编号、操作员、提交、机器人/相机身份及采集证据"
}
```

只有实验人员确认后才填写：

- `source_kind`：`synthetic`、`hardware_recording`或`unknown`。`hardware_recording`只是清单声明，不是工具认证；发现`simulation_only`或synthetic checkpoint证据会强制标记synthetic并记录冲突。混合批次标记mixed，并保留来源分组。
- `clock_domains`：相对于logs目录的文件名/通配符到**主机身份+OS启动身份**的映射。比如`task-events.jsonl`与`control-*.jsonl`只有确证同一主机、同一启动时才能填相同值。匹配到冲突声明会拒绝跨域比较。
- `embedded_clock_domains.worker`：父进程回执中`applied_monotonic_s`、`bridge_start`等复制字段的实际worker时钟。
- `embedded_clock_domains.observation`：`captured_monotonic_ns`的ROS观测进程时钟。不能因它出现在task-events里就当成父进程时间。
- `browser_wall_clock.offset_to_task_ms`：定义为`执行端Unix毫秒 = 浏览器Unix毫秒 + offset`。必须同时给出已验证的`uncertainty_ms`并置`verified=true`。只在**wall与wall之间**算点击到登记，再加已验证的服务端monotonic时长；绝不直接用Unix减monotonic。

主机名和`/proc/sys/kernel/random/boot_id`应在**实验采集时**记录。不能把分析机今天的boot_id补到以前实验上，也不能假定两台机器NTP同步就共享monotonic。日志中的前后wall/monotonic偏移突变、倒序、重复时间、跨文件声明冲突会记录质量问题；不排序或平移时间来隐藏异常。

若worker控制日志存在且token/epoch匹配，复制到父进程的实际发送时间可追溯回worker域；若其值不在对应worker发送日志的合理包络内，则标记异常并使用明确注明的worker日志时间代理。没有worker日志时，只能依赖清单显式声明的复制时钟域。浏览器offset的不确定度随每次切换导出，不当成精确零误差。

## 指标定义与缺失规则

| 字段 | 定义与证据 |
| --- | --- |
| `click_to_registered_ms` | 已验证wall时钟关系下，浏览器点击到`stage_requested`登记日志 |
| `registered_to_observation_ms` | 登记到最终用于新Prompt的观测完成；没有已接纳推理时可记录最后完成的观测 |
| `old_future_wait_ms` | `switch_policy_wait`到旧Future被回收，或请求先被新请求替代；队列任务明确取消时为0；无证据时缺失 |
| `new_prompt_inference_ms` | 绑定新推理ID的`inference_start`到完成或明确失败事件，`inference_timing_end`区分complete/failed；不是GPU采样时间 |
| `inference_to_armed_ms` | 新推理完成到父进程记录交接已安排 |
| `bridge_start_to_applied_ms` | 计划桥接开始到实际post-send字段或标明的worker post-send日志代理；不是执行器到位时间 |
| `click_to_applied_ms` | 已验证的点击→登记时长，加登记→worker实际命令接入时长 |
| `registered_to_applied_ms` | 执行端登记到worker接入；不能用更晚的父进程轮询时间冒充 |
| `old_phase` | worker处理切换之前最近的已记录frame.phase，不外推成精确点击相位；另附样本年龄，过旧样本记缺失 |
| `observation_age_at_inference_ms` | ROS snapshot选择完成到推理开始，要求观测时钟域声明 |
| `observation_age_at_result_ms` | 推理完成日志中直接报告的snapshot年龄，或在有时钟证据时计算 |
| `observation_age_at_armed_ms` | worker安排时直接报告的snapshot年龄；`observation_alignment`另外保留图像源年龄/跨度 |

这些分段会相互重叠，尤其观测与旧Future等待，**不能简单相加**。snapshot年龄也不是图像曝光到执行器的全链路年龄。旧Future最终被丢弃可发生在新请求之后；原始回收间隔和旧推理ID单独保留，替代请求的等待不算在原请求活动窗口内。

每次切换使用`run_id + request_id + stage_version`，新推理再绑定`inference_request_id`，worker使用token/epoch/source generation。普通RTC引导推理不混入跨Prompt首块统计；同Prompt的pickup_1/pickup_2、harvest_1/harvest_2保留独立阶段身份。快速连续点击产生superseded记录，不合并成一个成功请求。复制归档按task-events内容SHA-256去重，避免重复计算同次实验；去重事件可见于质量记录。

统计总体是**已登记的stage_requested**，不包括没有落入这些日志的HTTP/UI拒绝点击。因此“全部请求接入比例”不是“全部鼠标点击成功率”。READY/PAUSED的阶段选择不计入在线切换分母；无法判定范围的请求保留unknown并注明。只有匹配的父进程应用回执才计`applied`；worker单边回执为`worker_applied_unconfirmed`；不完整日志为`incomplete`，不自动猜成超时。回执冲突、先故障后应用、source generation不合法会标为证据冲突。timeout/rejected/failed依据现有原因文本分类，保留原始原因；日志没有结构化错误码时不把文本分类当成硬件诊断。

同时给出全部请求分母和有明确结果分母。P50/P95/P99默认仅对确认接入的切换、该指标非缺失样本计算；JSON还保留全部尝试的指标。使用NumPy线性经验分位数，每项附n；少量样本的P99不是尾延迟保证。

## 跟踪误差与动态量

58D分组：左臂`0:7`、左手`7:29`、右臂`29:36`、右手`36:58`。输出每个关节及每组的绝对误差均值、RMSE、峰值、P50/P95/P99。

误差是**同一日志帧的command减measured**；当前格式缺少每个反馈测量的同步硬件采样时间，未做延迟补偿，不能称为精确同步跟踪误差。

命令和反馈分别按实际日志间隔，用尾随2/3/4点Newton差商乘阶乘计算速度/加速度/jerk，并归属最后一个贡献帧；不固定除以1/30，不平滑、不插值。时钟段变化、非正dt、超过`max_frame_gap_s`的缺口中断历史；缺command或measured只中断相应通道，不丢弃另一通道的有效数据。这里100ms是分析缺口阈值，**不是修改硬件watchdog或允许延迟**。

桥接统计按对应token的计划开始和worker应用事件选帧，包含紧随应用事件记录的末端frame。导数可以引用桥接入口之前的有效历史，便于观察入口变化。计划开始、日志post-send时间和真正硬件采样时间仍有区别，测量噪声会显著放大高阶导数；工具不自动判断动态安全通过。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `report.md` | 中文实验报告、每次切换详细时间线、七阶段记录、来源/相位/Prompt分组、关节指标、失败与质量问题 |
| `summary.json` | 完整统计、逐请求/逐推理关联、终态快照、原日志SHA-256和时钟清单 |
| `switches.csv` | 每次切换的指标、结果、来源和不确定度 |
| `timeline.csv` | 各时间点、原文件与行号、时钟域和距登记时长 |
| `inference_attempts.csv` | 同一次请求的多个观测/推理/丢弃事件，不只保留最终接纳任务 |
| `stages.csv` | 七阶段软件执行证据；没有阶段生效日志但有终态快照时标记snapshot_only |
| `groups.csv` | 来源、具体阶段、四类Prompt转换和5步phase区间分组统计及样本数 |
| `joint_errors.csv` | 58个关节的全程误差统计 |
| `joint_dynamics.csv` | 全程及每个桥接的四组误差、命令/反馈导数统计 |
| `frames.csv` | 按帧、四关节组的误差/导数时间序列，供论文绘图；原始58D保留在输入日志 |

CSV为UTF-8 BOM，缺失单元格为空；JSON为null；中文报告显示“缺失”。CSV保留source_kind、session_id、单位明确的列名；不要把mixed批次当成纯实机数据。分析不会自动推断抓握稳定性、接触力或采摘成功率，即使终态是COMPLETED。

## 示例与验证

已提交的完整合成样例及其报告见 [examples/experiment_acceptance](examples/experiment_acceptance/README.md)。生成新的可重复样例：

```bash
# 必须是尚不存在的新目录，避免覆盖实验记录；只写合成数据，不运行设备
.venv/bin/python -m deploy.fr3_sharpa_runtime.experiment_example \
  --out .deployment/acceptance-demo-new

.venv/bin/python -m pytest -q deploy/fr3_sharpa_runtime/tests/test_experiment_analysis.py
.venv/bin/python -m pytest -q deploy/fr3_sharpa_runtime/tests packages/openpi-client/src/openpi_client/rtc_test.py
```

测试既使用确定性的合成归档，也使用现有`switch_audit.simulate`产生的真实协调器事件。覆盖多次切换、同Prompt不同阶段、多run版本重置、旧Future回收与取消、快速替代、超时/拒绝/故障、单边回执、日志截断、快照失败、字段缺失、乱序/倒退时间、wall跳变、不同机器时钟、归档去重、缺帧/缺反馈、非均匀时间求导、CSV/JSON/Markdown与两个CLI入口。

当前样例只有软件命令流结果。真实实验还需要采集时钟身份与同步误差、原始图像/反馈时间戳、完整控制记录，以及独立的接触、抓握和任务结果标注；不能用样例或此前GPU smoke补充这些实测数据。

## 本次交付验证（2026-11-11）

- 新分析用例32项；与已有switch_audit专项一起运行：**56 passed**。
- 部署运行时及openpi-client RTC全量回归：**656 passed**，1328条既有JAX/Flax弃用警告，无失败；运行耗时见原始测试日志。
- 分析相关Python的`ruff check --select E9,F`、`compileall`及`git diff --check`通过。
- 已通过两个分析CLI入口重算示例；六次切换的合成Click-to-Applied均为305ms，物理结果均缺失。
- 原始本机检查记录位于忽略目录`.deployment/tomato-validation/experiment-analysis-tests.log`和`experiment-analysis-regression.log`；原始合成输入及完整示例输出纳入版本控制。
- 本次未启动真实机器人、SDK/ROS硬件或模型，没有新增实机验证结论。
