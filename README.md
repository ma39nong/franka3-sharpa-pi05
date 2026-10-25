# FR3 + Sharpa π0.5 番茄任务部署

双 FR3 机械臂、双 Sharpa 灵巧手的视觉语言动作模型工程。当前部署使用 **π0.5 全量微调、30 Hz、原生 58 维绝对关节目标**。

## LoRA 部署版本

`codex/sharpa-lora-deploy` 分支新增 [30 Hz LoRA 独立入口](deploy/fr3_sharpa_lora/README.md)。模型配置、端口和日志与全量版本分开，复用相同 Home／SDK／执行与退出清理逻辑；以下全量命令默认行为保留。新的 LoRA checkpoint 尚未提供，目前只完成代码和配置测试，未完成真实 LoRA 加载或工作机实机验证。

## 当前配置

| 项目 | 配置 |
| --- | --- |
| 模型 | `gemma_2b` + `gemma_300m`，无 LoRA |
| 输入 | 三路 RGB、58 维实测关节位置、英文任务文本 |
| 输出 | 每次预测 50 帧 × 58 维，默认执行前 15 帧后重新观测推理 |
| 初始姿态 | poses.yaml 的 powder_weighing 双臂 Home，双手采用由左手实测角映射的固定 Home |
| 关节顺序 | 左臂 7、左手 22、右臂 7、右手 22 |
| 相机 | 头部 cam0、左腕 cam2、右腕 cam1 |
| 模型动作与设备下发 | 30 Hz，名义间隔约 33.33 ms |
| 执行后端 | `sdk_direct`：FR3 Safety Gateway + Sharpa Wave SDK |

本机配置绑定以下权重，代码仓库不包含权重：

```text
/home/descfly/桌面/sharpa_tomato_ckpt_4000
```

归一化统计自动读取该 checkpoint 的 `assets/tomato_sharpa_ab4_normal_rgb_absolute_v1_30hz/norm_stats.json`。更换路径时，在两个终端设置相同的 `SHARPA_CHECKPOINT`；不要替换为其他实验的统计。

## 完整启动、持续执行与停止

**第一次拉取仓库请先阅读 [新机器部署与研究交接](docs/SHARPA_HANDOFF.md)。**下列命令面向当前已配置的机器，不是从零安装步骤。换机器请替换每段 `cd` 的仓库路径，并先生成自己的 `.deployment/local.env`。权重、SDK、现场配置、双臂 Home 和固定双手 Home 不会随 git clone 下载。

本机启动脚本会自动读取 `.deployment/local.env` 并使用选定的 Python，不需要手动激活虚拟环境。请使用新的终端，或确认未遗留覆盖本机配置的 `SHARPA_*` 环境变量。

### 启动前：需要时同步双臂 Home

GUI 修改过双臂 Home 后，先同步到部署快照；没有更新时可跳过。同步不会覆盖 `.deployment/hand_home.json` 的固定双手配置。

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
source deploy/fr3_sharpa_runtime/env.sh
"$SHARPA_PYTHON" -m deploy.fr3_sharpa_runtime.home_pose \
  --source /home/descfly/franka_teleop_data/operator_gui/poses.yaml \
  --task powder_weighing \
  --via-docker
```

这里的 Docker 仅用于读取 root 专属权限的 poses.yaml，采用无网络、只读挂载，不启动设备。双手配置需要单独交接或标定；当前固定 Home 模式缺少手部配置会报错，不会自动改为保持启动时姿态。

### 终端一：检查配置并启动硬件依赖

先结束已有遥操任务和旧 `sharpa_driver`，避免与 sdk_direct 同时占用设备。前三条命令只检查配置，不连接设备；`hardware.sh check` 成功时可能没有输出。

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test

bash deploy/fr3_sharpa_runtime/hardware.sh check
bash deploy/fr3_sharpa_runtime/serve.sh --check-only
bash deploy/fr3_sharpa_runtime/runtime.sh \
  --check --initialization home --execute-steps 15 --speed slow --num-chunks 0

# 以下命令启动现场硬件服务
bash deploy/fr3_sharpa_runtime/hardware.sh up

docker ps --format 'table {{.Names}}\t{{.Status}}'
```

`up` 启动机械臂控制、使用 OpenPI 配置的 Safety Gateway 和三路相机，不启动旧 sharpa_driver 或 bag recorder。容器显示 Running 只说明进程状态，不能代替设备及实时观测检查。

### 终端二：启动模型服务

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
bash deploy/fr3_sharpa_runtime/serve.sh
```

等待出现以下预热成功信息，以及服务监听 `127.0.0.1:18765` 的信息，再运行执行端。保持终端二运行。

```text
Warmup OK: actions=(50, 58), full, absolute, 30 Hz
```

### 终端三：启动机器人持续执行

**以下命令连接设备并发送运动指令。**首次现场验证应先将最后的 `--num-chunks 0` 改成 `--num-chunks 1`，完成单轮检查后再持续运行。

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test

bash deploy/fr3_sharpa_runtime/runtime.sh \
  --execute \
  --enable-hardware \
  --sharpa-backend sdk_direct \
  --initialization home \
  --home-task powder_weighing \
  --prepare-speed-rad-s 0.03 \
  --execute-steps 15 \
  --speed slow \
  --num-chunks 0
```

执行过程与参数：

- 双臂读取 poses.yaml 快照中的 `tasks.powder_weighing.home`，双手读取 `.deployment/hand_home.json` 的固定目标，一起低速到位并稳定。**当前版本双手不是保持启动时姿态。**
- 每轮模型预测 50×58，只执行索引 0–14 的 15 个动作，丢弃后 35 个，再采集真实观测并重新推理。完整预测和执行步数写入日志。
- `--num-chunks 0` 持续运行，不限于 chunk1/chunk2；任意正整数表示运行对应轮数。Ctrl-C 或观测、推理、执行失败会结束循环。没有自动判断番茄任务完成的功能。
- `--speed slow` 对应时间倍率 0.1，15 帧主体为 141 个 30 Hz 下发点，约 4.70 秒；不含首帧衔接、观测和推理。medium=0.5、fast=1.0；不指定速度时默认 fast。`--speed` 不与 `--execute-speed-scale` 混用。
- prepare 速度为 0.03 rad/s，采用关节空间插值，不包含环境碰撞规划。当前代码还默认启用手部预测小幅越界裁剪 `--prediction-clip-rad 0.05`，其范围及审计含义见 [runtime 说明](deploy/fr3_sharpa_runtime/README.md#预测动作小幅越界裁剪)。

### 停止顺序

1. 先在终端三按 Ctrl-C。当前会话管理器会结束 runtime/SDK 子进程，并按容器身份清理本次部署所属的相机、OpenPI 网关和对应机械臂控制器；不自动回 Home。
2. 再在终端二按 Ctrl-C，停止独立的模型服务。
3. 若只启动了硬件而未启动执行端，或退出后仍有部署容器遗留，在终端一显式重试清理：

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
bash deploy/fr3_sharpa_runtime/hardware.sh stop
```

`stop` 可重复运行，使用归属核验，不对整个硬件项目执行 down。SIGKILL、断电或 Docker 不可用时不能保证自动清理完成；遇到清理错误应确认实际容器状态。详细归属检查和遥操接管行为见 [runtime 说明](deploy/fr3_sharpa_runtime/README.md#现场运行)。

### 运行记录

本机默认目录：`/home/descfly/.local/state/fr3_sharpa_runtime/execution_traces/`，可用 `--execution-trace-dir` 或 `SHARPA_EXECUTION_TRACE_DIR` 覆盖。包含 chunk JSON、events.jsonl 和 run.json。

```bash
ls -lt /home/descfly/.local/state/fr3_sharpa_runtime/execution_traces/
```

`COMPLETE` 表示指令下发完成，不表示任务成功。模型加载、单元测试和离线 dry-run 的通过记录不能代替本次现场运动验收。启动及运行限制详见 [runtime 部署说明](deploy/fr3_sharpa_runtime/README.md)，模型与执行层结构见 [架构说明](deploy/fr3_sharpa_runtime/ARCHITECTURE.md)。

## 检查与验证

以下检查不连接机器人：

```bash
cd /home/descfly/LTT/franka3-sharpa-pi05-chunk-base-test
source deploy/fr3_sharpa_runtime/env.sh
"$SHARPA_PYTHON" -m pytest deploy/fr3_sharpa_runtime/tests -q
bash deploy/fr3_sharpa_runtime/hardware.sh check
bash deploy/fr3_sharpa_runtime/serve.sh --check-only
bash deploy/fr3_sharpa_runtime/serve.sh --smoke-only
```

当前 4000 步权重已通过真实 GPU 加载及假观测推理，输出有限的 `(50, 58)`。这证明模型加载与动作接口可用，不代表现场任务成功。执行记录默认位于 `~/.local/state/fr3_sharpa_runtime/execution_traces/`。

原始转换器数据为 state116/action58；训练时选择实测位置并重排为 canonical 58D。实时部署输入已经是 canonical 58D，不能再套原始数据重排或重复添加当前关节位置。数据 normal/speedup 的区别、训练代码对应关系及尚未确认的实验信息见 [runtime 说明](deploy/fr3_sharpa_runtime/README.md)。

## 代码导航

| 目录 | 用途 |
| --- | --- |
| `deploy/fr3_sharpa_runtime/` | 当前模型服务、prepare、SDK 执行和运行记录 |
| `src/openpi/policies/fr3_sharpa_policy.py` | 58D 输入输出变换 |
| `src/openpi/policies/fr3_sharpa_protocol.py` | Sharpa 关节名称和协议 |
| `examples/fr3_sharpa/` | Sharpa 离线数据转换、导出与比较工具 |
| `src/openpi/training/` | 通用训练框架及保留的 Sharpa 历史训练配置 |
| `src/openpi/models/` | OpenPI 模型实现 |
| `scripts/` | 通用训练、统计和模型服务入口 |
| `packages/openpi-client/` | WebSocket 客户端 |
| `docs/` | 通用 OpenPI 参考资料与文档索引 |

## 训练来源、当前加载配置与历史版本

本仓库负责部署。当前权重由 `deploy/fr3_sharpa_runtime/model_loader.py::build_config()` 加载：`gemma_2b` + `gemma_300m`、无 LoRA、`action_dim=58`、`action_horizon=50`、`max_token_len=256`、30 Hz、`extra_delta_transform=False`。推理配置名为 `pi05_fr3_sharpa_ab4_normal_rgb_fullft_absolute_30hz_deploy`，它不是恢复出来的原始训练实验名。

训练参考工程是 [fruit-train/tomato-openpi](https://gitlab.litchibot.top/fruit-train/tomato-openpi)，本次文档核对版本为 `68440ed8b39693b5fd206403843901187f453c1c`。本机路径 `/home/descfly/LTT/tomato-openpi` 仅是该工程的一个 clone，不是其他人必须创建的目录。具体来源如下：

| 参考工程中的文件 | 用途 |
| --- | --- |
| `training_configs/experiments/tomato_sharpa/baseline/tomato_fr3_sharpa_fullft_v1.py` | Sharpa 全参 baseline，定义任务文本、数据身份和实验配置名 |
| `training_configs/recipes/pi05_fullft_v1.py` | 选择无 LoRA 的 Gemma 变体，全参数训练，baseline 全局 batch=64 |
| `training_configs/recipes/common.py` | 模型窗口、优化器、学习率、随机种子和 checkpoint 保存规则 |
| `training_configs/embodiments/fr3_sharpa.py` | 58D、rad、absolute、30 Hz 和三路相机角色 |

**2026-10-25 已在 28 服务器找到实际 AB4 UI 任务和当次训练归档。**任务 ID 为 `2200a091b1ee48239ad11b203a4bf934`，训练源码记录为 `42f887916d22cd58c022e95a000e2cb241a8a459`；实际记录是四个 A/B normal 批次、训练全局 batch **512**、FSDP **8**、absolute、horizon **50**、计划 30,000 步、每 2,000 步保存、`epoch_shuffle`。不能将上述参考 baseline 的 batch 64，或早期对话建议的 batch 32，写成该实验的实际设置。

实际 asset_id 为 `tomato_sharpa_ab4_normal_rgb_absolute_v1_30hz`。服务器统计文件与本机 4000 步 checkpoint 自带文件的 SHA256 完全一致；归档 `.experiment.py` 的摘要也与 UI 记录一致。源码快照、UI 覆盖参数、四个数据路径及两份摘要见 [已找到的 AB4 实验记录](docs/SHARPA_HANDOFF.md#已找到的-ab4-实验记录)。原始 checkpoint 旧路径目前不存在，仍未完成模型参数文件逐项比对；训练主进程环境变量和数据文件完整性也不能仅由这两份摘要推定。

本仓库 `src/openpi/training/config.py` 中的 `pi05_fr3_sharpa`、`pi05_fr3_sharpa_tier_a_b1_20hz`、`pi05_fr3_sharpa_a04_b05_c01_30hz` 是保留的历史 LoRA 配置，包含 LoRA 变体和双臂 delta 变换。当前 `serve.sh` 使用上述独立 model_loader，不选择这些历史配置；不要只把 checkpoint 路径替换到旧配置里加载。

旧 Wuji 部署和 `experiments/weight_motion_eval` 在 `c60b4c2612020d641ed3003a82cd8ab5d143f4fc` 清理。清理前版本是 `bbe7f697e17a220ca91fae10b1812322b0741fa2`，可在不切换当前分支的情况下查阅：

```bash
git show bbe7f697e17a220ca91fae10b1812322b0741fa2:deploy/fr3_wuji_fast/README.md
git ls-tree -r --name-only bbe7f697e17a220ca91fae10b1812322b0741fa2 -- deploy/ experiments/
```

旧实现包括 Wuji 设备协议、54D/其他模型映射和 RTG/Ruckig 方案，不是当前 Sharpa 的可运行入口。当前功能与历史设计的差异见 [架构说明](deploy/fr3_sharpa_runtime/ARCHITECTURE.md)和 [Wuji 历史参考](docs/reference_wuji_architecture.md)。

权重、训练数据、运行日志和本机构建产物不提交 Git。许可证见 [LICENSE](LICENSE) 和 [LICENSE_GEMMA.txt](LICENSE_GEMMA.txt)。
