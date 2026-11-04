# FLASH Absolute 部署适配（首版）

在 `train-rtc:e04dade` 上增量增加 `flash-absolute`，不合并训练源码，不替换既有 Python/JAX 环境。训练参考为 `LTT-BIT/tomato-openpi:paper-flash-test` 的 `e64c9a41ecb24494ea6c81e2df3dd4bbbfdcc45c`。源码可独立检出到本机；部署服务通过经过指纹验证的训练源码导入 `Pi0Flash`。本目录只包装推理、协议和导出，不修改训练数学或数据集。

当前没有正式 FLASH checkpoint。本版完成代码和离线契约适配，**尚未验证真实参数恢复、GPU 编译/推理、真实历史重采样配对、模型数值等价或实机运行**。默认不允许 FLASH 执行动作，包括 Home；模型检查、无硬件 smoke 与无动作观测评估可以单独进行。

## 模型及隔离边界

- K6、7 个系数 token、stride1、NFE1、history8；输入历史为实测 Absolute rad，包含当前帧；输出仍为完整 `50×58` Absolute rad。
- 顺序：左 FR3 7、左 Sharpa 22、右 FR3 7、右 Sharpa 22。头部 cam0、左腕 cam2、右腕 cam1。
- `state` 使用 checkpoint 的 state quantiles，维持训练 numpy 运算和 tokenizer 分桶顺序；history 原样交给模型内部 action quantiles；返回动作不再次 Unnormalize，不加 Delta anchor。
- RGB/resize/padding 和原生训练预处理一致；tokenizer 从部署包本地加载，启动不下载 tokenizer；模型 strict load 全部参数，不从 pi05_base 拼装，不保留随机投影。
- 只开放 serial。FLASH 的 RTC/noise/未来标签/Delta/Paper/TrainRTC/LoRA 混用均拒绝。旧配置保持原支持矩阵。
- 服务默认 `127.0.0.1:18768`。`SHARPA_FLASH_PYTHON` 只用于模型服务；GUI、协议、ROS、SDK 继续使用原来的解释器。启动器只将训练 `src/` 和部署仓库根目录放入服务 `PYTHONPATH`，不包含部署仓库 `src/`。
- `codec.py` 来自本部署仓库 `packages/openpi-client/src/openpi_client/msgpack_numpy.py`，保留其来源说明，许可证见 `UPSTREAM_LICENSE`（Apache-2.0）。测试中的两段原生预处理参考固定到上述训练 commit。

## 准备环境

首先准备一个与训练端兼容的**独立**解释器。优先使用训练端相同锁定环境的独立副本/导出依赖，不修改正在训练的环境或旧部署环境。训练树的 uv 配置引用其 workspace `openpi-client`，不能假设当前训练 checkout 已自带完整推理客户端；本适配自带 wire codec 和服务，不依赖该缺失入口。

已核对训练端核心版本：JAX 0.5.3、Flax 0.10.2、Orbax 0.11.13、Torch 2.7.1、Transformers 4.53.2。还需训练模型已有依赖，以及本目录 `requirements.txt` 中的 msgpack/websockets。真实 checkpoint 到位后，在独立解释器里执行 smoke 才能确认整套依赖可用；本次未全量安装训练/GPU环境，也未修改现有锁文件。

在部署仓库 `.deployment/flash.env`（忽略 Git）配置，例如：

```bash
export SHARPA_FLASH_REPO=/home/descfly/LTT/tomato-openpi-flash
export SHARPA_FLASH_PYTHON=/实际独立FLASH环境/bin/python
export SHARPA_FLASH_CHECKPOINT=/实际已完成训练目录/2000
export SHARPA_FLASH_PORT=18768
# 一般无需设置；若设置，只能指向同一个包内、已封存的统计。
# export SHARPA_FLASH_NORM_STATS=.../2000/assets/asset_id/norm_stats.json
```

以上解释器和 checkpoint 是待提供路径，不能把示例占位值当作已安装。普通 full/Delta/LoRA/Paper/TrainRTC 不要求这个文件，也不会启动 FLASH 进程或历史缓存。

## 导出已完成保存的 step

必须有完整的 `params/` 数据文件、Orbax 的 commit marker 和以下同包资产：`norm_stats.json`、`flash_manifest.json`、`flash_seal.json`、`flash_validation.json`。`flash_training_contract.json` 位于实验目录。仅 `_METADATA` 和 `manifest.ocdbt` 不算完整参数。

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05
.venv/bin/python -m deploy.fr3_sharpa_flash.export \
  --checkpoint /实际已完成训练目录/2000 \
  --training-repo /home/descfly/LTT/tomato-openpi-flash \
  --training-contract /实际实验目录/flash_training_contract.json \
  --tokenizer /home/descfly/.cache/openpi/big_vision/paligemma_tokenizer.model
```

若 step 目录被改名，显式传 `--step 实际保存步数`。导出器不会推测参数来自哪个普通模型；它核对训练契约、模型配置、源代码、统计与数据 seal，复制小型 tokenizer/契约资产，并生成 `flash_deployment.json`。参数文件保持原位置，逐个记录大小和 SHA256；不复制或改写训练权重。已有封存 manifest 不覆盖，检查失败不留下有效 manifest。不要对正在保存的目录执行导出。

训练契约中的 `Pi0FlashConfig(...)` 仅用 AST literal 解析，不执行 eval。action quantiles 必须与同包统计精确一致；来源训练数据指纹与训练源码指纹也必须匹配。检查会读取参数内容做校验，首次耗时/磁盘带宽需要在真实包到位后测量。

## 检查、smoke、无动作评估

```bash
# 元数据/文件完整性检查：不创建模型，不初始化 GPU，不连接硬件
bash deploy/fr3_sharpa_runtime/serve.sh --model flash-absolute --check-only

# 加载实际模型，固定形状合成预热后退出：仍不连接硬件
bash deploy/fr3_sharpa_runtime/serve.sh --model flash-absolute --smoke-only

# 预热成功后才开放 WebSocket
bash deploy/fr3_sharpa_runtime/serve.sh --model flash-absolute

# 旧部署解释器执行参数检查
bash deploy/fr3_sharpa_runtime/runtime.sh --model flash-absolute --check

# 读取同 episode 至少七帧真实过去数据；默认旧 frame=0 行为不变
bash deploy/fr3_sharpa_runtime/runtime.sh --model flash-absolute \
  --episode-parquet /实际数据/data/chunk-000/episode_000000.parquet --sample-frame 20

# 有真实传感器和服务后，用既有只读预检；不调用执行入口准备设备
bash deploy/fr3_sharpa_runtime/runtime.sh --model flash-absolute --preflight-only
```

FLASH 的 check/smoke 使用其独立入口，不调用普通 `serve.py`。`runtime.sh --check` 是执行参数检查；完整包检查用上面的 service `--check-only`。合成预热没有真实历史来源，不能通过执行入口使用它。当前没有 checkpoint，因此这些真实模型命令尚未运行通过。

## 在线历史与执行准入

仅选择 FLASH 才创建 `FlashObservationProvider`/`FlashHistoryBuffer`。普通模式继续原先的当前观测缓存、200ms/100ms 闸门，不要求历史。FLASH 独立缓存默认保留 1 秒、每组最多 8192 个轻量数值条目；源时间、接收单调时间和源 publisher 身份可追溯。八行时间为 `t0-round((7-i)*1e9/30)`，允许相邻纳秒差相差 1ns。历史末行直接成为当前 state，图像从同次冻结视图中选不晚于 t0 的帧。

候选重采样规则 `causal-zoh-30hz-v1`：每个网格选不晚于该时刻的最近真实样本，允许正常约20Hz手反馈被有界复用。候选上限为每格来源年龄75ms、原始更新间隔100ms；当前实际选中状态/图像仍必须满足200ms新鲜度和七路100ms跨度。它们只是第一版保守配置，**尚未证明与训练转换器等价**。不提高 SDK 读取频率，不复制当前姿态凑历史。乱序/重复源时间、断流、无效值、时钟跳变、publisher 重连均使旧历史失效，新 session 重新积累；连续同会话的 Home/执行期间不清空有效历史。

恢复执行准入前，需要真实离线/现场证据，生成并审查 `execution_validation.json`，至少包含：

```json
{
  "status": "PASS",
  "resampler_version": "causal-zoh-30hz-v1",
  "history_pairing_verified": true,
  "model_equivalence_verified": true,
  "max_plan_age_ms": 0,
  "checkpoint_manifest_sha256": "实际校验值",
  "normalization_sha256": "实际校验值",
  "flash_backend_identity": "实际校验值"
}
```

这里的 `max_plan_age_ms: 0` **故意无效**，不能照抄以放行动作；应由实际桥接/端到端延时和现场允许预算确定正数。也不能只把两个验证字段改成 true 冒充已做配对。验收报告应保留原始记录、direct-model/WebSocket 输出差值和时延测量证据。三个 identity 值可从包检查输出取得。导出时用 `--execution-validation 实际报告路径` 将它封存到包；未提供、身份不匹配或数值无效均拒绝执行。

GUI 启动动作前、执行 CLI、SDK worker 启动前均检查预算；worker 算出桥接长度后、发送任何帧之前检查“anchor 到首个模型目标的预计年龄”。桥接帧以30Hz输出，不按任务节点速度缩放再次计算。发送首帧时重新检查；如果桥接后的录制准备又造成延迟，在首个模型目标处再次检查，超时停机。正常播放后面的15个节点不反复套用首节点年龄阈值。无无限重试、无重置 t0 掩盖过期、无 RTC receipt 伪装。

## trace 与性能边界

仅 FLASH 增加真实 `history_rad` 数值、原 source/grid/context、session/request、模型/统计/代码身份、回执和请求/响应时刻；模型输出来源标为 `flash_internal_denormalize`。执行报告包含桥接长度、首个命令/首个模型目标/最后命令时刻；原动作与修正/执行记录仍分开保存。旧 trace 字段与 post-Unnormalize 描述保留。

默认仍预测50、串行执行15，再观测。桥接和速度缩放会延后节点实际播放，不能宣称本版为 observation-time 连续实时 FLASH。NFE1 不表示每 tick 只执行一帧。

CPU预处理、host/device copy、WebSocket往返和同步后的模型总耗时可记录；正式 VLM/Expert/多项式路径保持 fused/JIT，以 `fused_vlm_expert_polynomial_ms` 标注，尚未单独拆阶段 profiling。没有真实 checkpoint/GPU测量，尚无热态p50/p95/p99或部署收益结论。

## 回归及后续验收

修改前基线：`e04dade`、工作树干净，运行时目录 **472项通过**。新增 FLASH 测试覆盖包/代码篡改、双方 profile 错选、无 GPU 检查、参数数据文件存在性、8帧/20Hz有界因果历史、状态原子性、时钟/重连/断流、wire codec互通、回执错配、NFE1/history透传、无二次输出变换、预算过期拒绝和trace。预处理对照使用固定训练源码片段；本机实际tokenizer还验证了分桶后token/mask一致。mock结果不代表真实权重恢复通过。

共享改动均按显式 profile 分流：profiles/launchers/contract/config、GUI能力与路由、ROS callback工厂及FLASH provider选择、客户端history/receipt、串行执行计划年龄门和FLASH trace。普通模型函数和采样数学不改。原 registry 测试只增加新 profile 的预期项，保留旧项；不删除/skip/xfail既有测试。

| 既有模式 | 保留的合法组合 | 对应离线回归 | 尚未完成 |
| --- | --- | --- | --- |
| Full Absolute | Serial、普通RTC | full_finetune 5项、rtc_pipeline 8项、rtc_stream 20项通过 | 同真实checkpoint/GPU数值对照 |
| Full Delta | Serial、普通RTC | full_delta 8项、rtc_pipeline 8项通过 | 同真实checkpoint/GPU数值对照 |
| LoRA Absolute | Serial | lora_deployment 19项通过 | 同真实checkpoint/GPU数值对照 |
| Paper/T3 | Serial、Paper RTC | paper_deployment 23项通过 | 同真实checkpoint/GPU数值对照 |
| Train RTC Absolute/Delta | Serial、Train RTC | train_rtc 10项、training_archive_detection 27项通过 | 同真实checkpoint/GPU数值对照 |
| Train + Paper RTC | Serial、Train Paper RTC | train_paper_rtc 12项通过 | 同真实checkpoint/GPU数值对照 |
| FLASH Absolute | Serial；未验收默认禁动作 | test_flash_deployment 42项通过 | 参数恢复、GPU、真实源配对、完整模型等价、实机 |

最终回归：**515项通过、0失败、0跳过**；其中FLASH专项42项。只有现有JAX/Flax弃用警告。新增模块Ruff、共享模块未定义名称/语法检查、Shell语法、GUI JavaScript语法与git diff空白检查通过。

完整测试命令：

```bash
SHARPA_SCHEMA_PATH=/home/descfly/ZH/gello_upper_body_teleop/config/litchibot/sharpa_schema.json \
.venv/bin/python -m pytest -q deploy/fr3_sharpa_runtime/tests
```

先完成真实包check/smoke、同一episode/time/history/RNG的直接模型与WebSocket对照，再做只读在线观测、原始记录与转换数组配对和延迟注入。通过后才按既有受控流程验证Home/current、15节点桥接、范围/反馈/deadline、断流停止、录制与失能收尾。本版没有新增异步scheduler、Delta-FLASH、T3-FLASH、TrainRTC-FLASH、速度控制接口或新的相机/录制器行为。
