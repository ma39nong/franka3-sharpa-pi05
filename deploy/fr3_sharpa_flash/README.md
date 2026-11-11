# FLASH Absolute 部署

直接选择原始checkpoint步骤目录和模型类型即可，无需运行导出器、生成 `flash_deployment.json` 或复制训练契约/tokenizer；无需提交执行验证报告。训练参考源码为 `LTT-BIT/tomato-openpi:paper-flash-test` 的 `e64c9a41ecb24494ea6c81e2df3dd4bbbfdcc45c`，模型服务使用独立解释器导入该训练工程，不替换旧部署模型环境。

已核对本次8000步checkpoint的51个全局参数路径、形状和dtype，以及配套58D统计、动作分位数和原始模型配置。真实参数恢复、GPU推理与服务数值对照、在线历史与训练转换结果配对尚未完成，本次按要求暂缓；离线回归通过不代表实机验收完成。本次没有启动机器人或执行Home。

## 在GUI中使用

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05
bash deploy/fr3_sharpa_control/ui.sh
```

打开终端提示的网址（默认 `http://127.0.0.1:8766/`）。若该端口已有控制台，使用原页面；修改代码后需停止旧UI进程并重新启动，刷新网页。

1. 选择直接包含 `params/` 和 `assets/` 的完整checkpoint步骤目录。
2. 模型类型选择 **FLASH Absolute · K6 / stride1 / NFE1 / history8**。
3. 执行模式选择 **flash**（串行）或 **flashRTC**（后台推理）。flash每轮预测50点，执行前N点后重新观测，默认15；flashRTC的15表示开始后台规划的触发进度，实际切换通常更晚。轮数0持续运行。旧配置的serial仍兼容。
4. 选择Current或Home与录制设置。flash保留原速度选项；flashRTC仅fast，并需在FLASH参数中填写现场已验证的接缝位置、速度差及反馈误差上限，0表示未配置，不能启动异步执行。训练RTC参数保持0。
5. 点击“启动推理”；预热成功后核对执行配置，再点击“启动执行”。

训练JSON不是必填项。FLASH会就地读取步骤目录或其父目录已有的 `flash_training_contract.json` 中的模型配置；缺少时采用本次已核对的K6/stride1/history8配置，action quantiles始终读取该checkpoint自己的 `norm_stats.json`。不生成训练文件副本，不改写checkpoint。若已有训练配置与支持的模型结构/统计冲突，会明确报错；可用 `SHARPA_FLASH_TRAINING_CONFIG` 指向原始配置文件，无需复制。

## 模型服务环境

在 `.deployment/flash.env`（忽略Git）中配置机器相关路径，例如：

```bash
export SHARPA_FLASH_REPO=/home/descfly/LTT/tomato-openpi-flash
export SHARPA_FLASH_PYTHON=/实际独立FLASH环境/bin/python
# GUI选取权重时无需在这里填写checkpoint；命令行启动则需要：
# export SHARPA_FLASH_CHECKPOINT=/实际训练实验目录/8000
export SHARPA_FLASH_PORT=18768
# 可选：直接指向已有tokenizer，不复制到checkpoint中。
# export SHARPA_FLASH_TOKENIZER=/home/descfly/.cache/openpi/big_vision/paligemma_tokenizer.model
```

示例解释器路径需替换成已安装的环境。已核对训练端核心版本为JAX 0.5.3、Flax 0.10.2、Orbax 0.11.13、Torch 2.7.1、Transformers 4.53.2；还需训练模型依赖和本目录 `requirements.txt` 中的msgpack/websockets。此修改没有安装/替换GPU环境。

tokenizer优先使用显式路径；否则读取checkpoint已有的 `tokenizer.model`、用户的OpenPI缓存或训练仓库缓存。服务启动不联网下载。GUI/CLI传入的权重、统计和参数优先于 `flash.env` 中旧值。

## 文件要求与命令行检查

原生Orbax步骤需要已提交的 `_CHECKPOINT_METADATA`、`params/_METADATA`、`params/manifest.ocdbt` 和实际OCDBT数据文件，`assets/` 中需要配套的 `norm_stats.json`。只下载元数据的审查目录不能当权重使用。有原始训练数据manifest时继续核对58D关节顺序、Absolute语义、K6/history8等设置；不要求人为创建这些文件。原有封存包仍兼容，旧导出工具仅用于兼容归档，不是启动步骤。

```bash
# 命令行启动时先设置真实权重路径（或写入flash.env）
export SHARPA_FLASH_CHECKPOINT=/实际训练实验目录/8000

# 元数据检查，不加载JAX/模型，不连接硬件
bash deploy/fr3_sharpa_runtime/serve.sh --model flash-absolute --check-only

# 真权重恢复、合成输入预热后退出；不连接硬件
bash deploy/fr3_sharpa_runtime/serve.sh --model flash-absolute --smoke-only

# 预热成功后开放本机WebSocket服务
bash deploy/fr3_sharpa_runtime/serve.sh --model flash-absolute

# 另一个终端：执行参数检查，不执行动作
bash deploy/fr3_sharpa_runtime/runtime.sh --model flash-absolute --check
```

上述真实模型smoke尚未执行通过；这是后续验证命令。合成预热输入只用于编译，不能代替真实观测。

## 在线输入及保留的检查

- 模型使用8行实测Absolute rad历史（包含当前帧），NFE固定1；输出完整 `50×58` Absolute rad。全局参数恢复采用strict load，不从base模型拼装。
- 关节顺序为左臂7、左手22、右臂7、右手22；相机为头cam0、左腕cam2、右腕cam1。
- state量化、tokenizer、RGB缩放/resize/padding沿用训练逻辑；history原样交给模型内部使用action quantiles，输出不再次反归一化、不加Delta anchor。
- FLASH独立积累历史，普通模型继续使用原有观测路径。候选因果重采样规则为 `causal-zoh-30hz-v1`，每行选择不晚于网格时间的真实反馈，最多复用75ms；原始更新间隔上限100ms。乱序、断流、时钟变化或publisher切换会重建历史会话，禁止用零值或命令值拼历史。
- 实际选中观测仍要求200ms新鲜度及七路100ms跨度；保留模型预热/握手、关节范围、反馈、命令deadline、停止和失能收尾检查。
- FLASH附加计划年龄检查改为可选：GUI“FLASH附加参数”中填写毫秒数，**0默认关闭**；CLI对应 `SHARPA_FLASH_MAX_PLAN_AGE_MS`。正数限制的是观测anchor到首个模型目标的年龄，包含桥接时间；不读取验证报告。它不修改上述实时反馈检查。
- 支持flash串行和flashRTC异步；普通Full Absolute/Delta、LoRA、Paper及Train RTC原有合法模式保留。flashRTC是执行调度，不使用模型RTC前缀引导，`rtc_supported`仍为false。

trace继续记录真实history/context、请求回执、配置/统计身份及推理耗时；即使关闭附加计划年龄检查，仍保存执行时刻。桥接和慢速播放会延后动作，不承诺每个相机tick都执行一个FLASH动作。


## flash / flashRTC 的执行入口

两种模式继续使用相同的 `serve.sh --model flash-absolute` 服务、`runtime.sh`、Current/Home、相机准入、SDK反馈、录制和失能读回流程。无需生成新的契约或验证报告。只切换这两种模式时，GUI可复用已经预热的同一FLASH模型服务；切换其他模型沿用原有身份检查。

```bash
# 串行：先检查参数；将 --check 替换为 --execute --enable-hardware 才会执行。
bash deploy/fr3_sharpa_runtime/runtime.sh --model flash-absolute \
  --broker-mode flash --initialization current --speed fast \
  --execute-steps 15 --num-chunks 0 --check

# 异步：三项限值必须由现场已验证配置提供；这里没有推荐数值。
# 先在当前shell设置 FLASH_SEAM_POSITION_RAD / FLASH_SEAM_VELOCITY_RAD_S / FLASH_FEEDBACK_ERROR_RAD。
bash deploy/fr3_sharpa_runtime/runtime.sh --model flash-absolute \
  --broker-mode flashRTC --initialization current --speed fast \
  --execute-steps 15 --num-chunks 0 \
  --flash-seam-position-rad "${FLASH_SEAM_POSITION_RAD:?请填写已验证的位置及单帧指令变化限值rad}" \
  --flash-seam-velocity-rad-s "${FLASH_SEAM_VELOCITY_RAD_S:?请填写已验证的接缝速度差限值rad/s}" \
  --flash-feedback-error-rad "${FLASH_FEEDBACK_ERROR_RAD:?请填写已验证的反馈跟踪误差限值rad}" --check
```

`flashRTC`的触发节点允许1..24。轮数正数表示实际完成安装的计划总数，首段计1；达到总数后，最后一代执行到触发进度就受控结束，不播放完全部50点。轮数1也可用于首段检查。取消或拒绝的候选不计数；连续两次接缝拒绝就结束本轮，耗尽前预留100ms停止预算。不会自动切回串行，也不重复末端模型点。

时间线使用采集时保存的ROS/monotonic时钟配对，把锚点年龄计入相位；worker在真正未来切换边界使用 `ceil(30*(t_switch-t_anchor)-eps)` 选择尚未过期的节点。至少保留20个节点，并提高到足以覆盖下一轮触发进度、本次已测延迟、100ms停止余量及两个提交间隔的节点数。旧计划的已下发指令不可撤回，接管前还保留两个未来控制间隔。若实际延迟使剩余窗口不足，停止或拒绝候选；默认15触发和20节点下限不构成设备性能保证，需要按实测延迟选择更早触发。

初次预测只用于现有受限接近；接近及录制准备完成后，重新采集图像、state和真实8帧历史，重新预测连续执行的首段。异步不套用串行的“推理返回时200ms整段丢弃”规则，采样时新鲜度仍为200ms，执行期间独立检查相机、历史会话及四肢反馈。当前实现为P0直接接管：检查接缝位置、离散速度差、反馈误差以及实际指令变化和既有关节限位；不合格原子拒绝。未启用可选的混合/平滑桥接。

异步记录保存在execution trace目录下的 `flashRTC-<session>/`：`requests.jsonl`包含请求、数值历史、原始预测、回执、拒绝及安装记录；`control.jsonl`包含修正后的轨迹、实际命令、反馈及相位；`run.json`记录实际安装数、拒绝数和采集到计划切换延迟的p50/p95/p99。单个模型请求在后台等待期间，硬件worker独立维护30Hz，推理线程不能调用worker。轨迹耗尽、反馈/时钟/传感器故障、截止时间及用户停止均回到原有SDK退出和失能确认链路。

## 离线回归

```bash
SHARPA_SCHEMA_PATH=/home/descfly/ZH/gello_upper_body_teleop/config/litchibot/sharpa_schema.json \
.venv/bin/python -m pytest -q deploy/fr3_sharpa_runtime/tests
```

覆盖原始checkpoint直读、无训练副本/无报告、手动模型选择与模式限制、原始训练配置/分位数匹配、参数数据完整性、轻量身份缓存失效、预处理、历史与回执、可选年龄限制、旧封存包兼容，以及既有模型/RTC/录制/停止流程。模型真实加载与现场性能、历史配对、实机执行仍待后续验证。

wire codec来源为本仓库 `packages/openpi-client/src/openpi_client/msgpack_numpy.py`；许可证见 `UPSTREAM_LICENSE`（Apache-2.0）。


## 2026-11-06 实施与验证记录

基线为现有 `flash-test` 分支，HEAD `df5d4b8024d076bca1cca4369e13225fafb83efc`，修改前工作区干净。没有新建分支、worktree或clone，没有提交/推送，也未触碰训练工程与真实硬件。

修改前运行runtime全部测试：529通过、4失败。4项均为本机默认 `/home/user/.../sharpa_schema.json` 不存在；使用上方显式 `SHARPA_SCHEMA_PATH` 指向本机已有原限位文件后复验，无需更改限位或测试期望。

实现后全量574项通过（38.30s）；最后增加仅异步启用的WebSocket发送截止时间后，FLASH及异步相关93项再次通过（4.14s）。同时通过Python编译、GUI脚本 `node --check` 和 `git diff --check`。原Flash能力测试仅更新了新增模式列表断言，旧普通/Delta/LoRA/Paper/TrainRTC测试及其预期保持。

| 模型/模式 | 本次实际执行的代表性回归 | 结果 |
| --- | --- | --- |
| Full Absolute串行、普通RTC | test_full_finetune 5项；test_rtc_pipeline 8项；test_rtc_stream 20项 | 通过 |
| Full Delta串行、普通RTC | test_full_delta 8项及公共RTC回归 | 通过 |
| LoRA Absolute串行 | test_lora_deployment 19项 | 通过 |
| Paper串行、Paper RTC | test_paper_deployment 23项及公共RTC回归 | 通过 |
| Train RTC Absolute/Delta串行与训练前缀 | test_train_rtc 10项 | 通过 |
| Train + Paper串行与训练前缀 | test_train_paper_rtc 12项 | 通过 |
| flash串行、checkpoint/历史/服务/GUI | test_flash_deployment 58项 | 通过 |
| flashRTC相位、提交保护、拒绝、超时、传感器故障与多代重规划 | test_flash_async 35项 | 通过 |
| 控制台生命周期及既有模式 | test_deployment_center 46项 | 通过 |

异步测试使用可控时钟和假推理/假worker，覆盖300ms延迟下15节点触发及500ms延迟下6节点触发，旧命令持续以30Hz推进并安装4代计划。真实loopback WebSocket验证模型sink的请求/回执和有界传输，不能等同于真实模型数值对照。

共享改动仅在run入口分流、SDK worker新增FLASH操作、client增加默认关闭的总截止时间，以及控制台模式/参数/FLASH服务身份复用。原RTCTimeline、模型Policy和普通/Paper/TrainRTC输出变换未改；FLASH封存manifest继续按原serial能力契约校验，新的执行能力只在运行时metadata中声明。

尚未完成：12.44GB真实参数完整恢复、GPU smoke、同checkpoint/同噪声的旧模式模型数值对照、FLASH训练转换与在线历史专项配对、真实p95/p99延迟、硬件接缝阈值标定、台架与实机连续执行。当前审查目录是元数据副本，不能充当可运行权重；FLASH独立环境尚需配置。P1平滑混合为可选研究功能，本次未实现。上述未验证项没有以PASS报告替代，也未作为生成契约副本的新门槛。
