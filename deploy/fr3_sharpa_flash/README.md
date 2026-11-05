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
3. 执行模式选择**普通串行**，每轮执行动作数通常15，轮数0表示持续运行；每轮都预测完整50点，执行前N点后重新观测推理，不限于chunk1和chunk2。
4. 选择Current或Home、速度与录制设置。FLASH不支持RTC，RTC参数保持0。
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
- 仅支持串行；普通Full Absolute/Delta、LoRA、Paper及Train RTC原有合法模式保留。FLASH不因填写RTC参数而获得RTC能力。

trace继续记录真实history/context、请求回执、配置/统计身份及推理耗时；即使关闭附加计划年龄检查，仍保存执行时刻。桥接和慢速播放会延后动作，不承诺每个相机tick都执行一个FLASH动作。

## 离线回归

```bash
SHARPA_SCHEMA_PATH=/home/descfly/ZH/gello_upper_body_teleop/config/litchibot/sharpa_schema.json \
.venv/bin/python -m pytest -q deploy/fr3_sharpa_runtime/tests
```

覆盖原始checkpoint直读、无训练副本/无报告、手动模型选择与模式限制、原始训练配置/分位数匹配、参数数据完整性、轻量身份缓存失效、预处理、历史与回执、可选年龄限制、旧封存包兼容，以及既有模型/RTC/录制/停止流程。模型真实加载与现场性能、历史配对、实机执行仍待后续验证。

wire codec来源为本仓库 `packages/openpi-client/src/openpi_client/msgpack_numpy.py`；许可证见 `UPSTREAM_LICENSE`（Apache-2.0）。
