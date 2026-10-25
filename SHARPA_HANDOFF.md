# 新机器部署与研究交接

本文面向第一次拉仓库的人，区分离线模型研究、现场部署和训练复现。文档核对基线为本仓库 `b796025`；本机部署路径只是实例，迁移时使用自己的绝对路径。本文未在全新机器上重新安装环境；安装命令依据仓库的 pyproject.toml/uv.lock，已有机器的通过记录不等于新机器已验收。

## 仓库和职责

| 工程 | 来源与核对版本 | 责任与是否必需 |
| --- | --- | --- |
| 当前部署仓库 | `https://github.com/ma39nong/franka3-sharpa-pi05.git`，分支 `chunk-base-test` | 模型服务、观测、动作协议、Home、执行调度和日志；必需 |
| 训练参考 | `https://gitlab.litchibot.top/fruit-train/tomato-openpi.git`，`68440ed8b39693b5fd206403843901187f453c1c` | 理解 recipe 或开展训练时使用；仅加载已有权重不需要安装此工程 |
| 硬件工程 | `ssh://git@gitlab.litchibot.top:2222/fruit-infra/data-collection/gello-teleoperation.git`；本机核对 `4fd958fdd1de9762da2375759625da49e78bab39` | FR3 控制、安全网关、相机、消息定义和 Sharpa HAL；现场部署必需 |

GitLab 地址需要相应访问权限。硬件仓库的 `portable_deps/litchi_hardware`、供应商 SDK 和本机配置不由普通 clone 自动提供；应向设备/工程维护者取得授权的完整依赖包。不要假设拿到仓库源码就已拿到供应商运行库。硬件工程根目录的 `工程迁移与一键安装说明.md`、`一键安装环境.sh` 和 `data_collection/docs/SHARPA_ARM_RECORDING.md` 分别说明环境安装和相机数据链路。

上表 commit 是本次可核对的源码版本，不是已证明足以重建全部供应商依赖的环境镜像。移交时同时记录实际使用的部署、训练、硬件版本和本机修改。

## 交接资产清单

以下内容不在当前部署仓库 Git 中，提供者必须另行交接。接收者需要逐项取得，不能根据文档里的旧机器路径推测内容。

| 资产 | 必需内容 | 谁需要 |
| --- | --- | --- |
| 完整 checkpoint | 当前 4000 步目录的 `params/` 全部 Orbax 数据文件及 `assets/.../norm_stats.json`；不能只交 manifest | 模型推理、部署 |
| 权重身份 | 当前资产目录为 `tomato_sharpa_ab4_normal_rgb_absolute_v1_30hz`；模型服务检查 58D stats 并在握手中发送统计和 manifest 指纹 | 模型推理、部署 |
| tokenizer | PaliGemma tokenizer；首次模型变换可能从 `gs://big_vision/paligemma_tokenizer.model` 下载，受限网络应预先准备当前 OpenPI 缓存 | 模型推理 |
| 硬件依赖包 | `portable_deps/litchi_hardware`、其 Pixi ros2 环境与锁文件、Sharpa 供应商库，以及硬件工程安装所要求的其他 portable_deps | 现场部署 |
| 本机硬件配置 | 硬件仓库 `docker/.env`、workcell 配置、`data_collection/config/cameras.yaml`、左右设备身份及 SDK 绑定信息 | 现场部署 |
| 双臂 Home | `poses.yaml`，包含 `tasks.powder_weighing.home.left/right` 各 7 个具名关节角，单位 rad | 固定 Home 部署 |
| 固定双手 Home | `.deployment/hand_home.json`；`unit: rad`，`tasks.powder_weighing.left/right` 各含 22 个 `joint_names` 和对应 `positions` | 当前默认固定 Home 部署 |
| 数据与任务文本 | 有名称/频率元数据的 LeRobot 数据和 `meta/tasks.jsonl`；服务器路径不是下载地址 | 真实样本研究、效果评估 |
| 原始训练记录 | 实际实验模块、源码 commit、完整启动参数、数据清单、统计审计、日志和种子 | 重训复现 |

`train_state/` 用于训练恢复；当前推理加载 `params/` 和配套 stats，不需要 optimizer state。不要将另一实验的 stats 配给当前 checkpoint。

固定双手 Home 当前来自 2026-10-23 的现场采集：左手实测角用于左手，右手按同名关节使用相同数值，不做符号翻转。这是本次现场标定约定，不代表任何 Sharpa 装置都可直接采用。文件缺失时当前默认 Home 会失败；双臂快照同步工具不会自动生成该文件，也不会代替另一台设备的标定。

## 从零建立模型环境

先准备 uv、Git、可用的 NVIDIA 驱动；现场路径还需要 Docker Compose 和 Pixi。使用组织已有的安装方式安装这些工具，勿把另一机器的 `.venv` 目录直接复制后当作可移植环境。

```bash
git clone --branch chunk-base-test https://github.com/ma39nong/franka3-sharpa-pi05.git
cd franka3-sharpa-pi05
git submodule update --init --recursive
uv python install 3.11
uv venv --python 3.11 .venv
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
```

锁定依赖包括 JAX CUDA12 0.5.3、Flax 0.10.2、Orbax 0.11.13、Torch 2.7.1，Torch 使用项目声明的 cu128 源。不要单独更新 JAX/Flax/Orbax 来绕开安装错误；先保存具体错误并核对驱动、下载权限和锁文件。当前已有环境曾在 RTX 5090 上通过加载，但并未验证所有显卡组合。

### 只做模型研究，不准备硬件

取得完整 checkpoint 后，在同一终端显式设置路径：

```bash
export SHARPA_PYTHON="$PWD/.venv/bin/python"
export SHARPA_CHECKPOINT=/absolute/path/to/sharpa_tomato_ckpt_4000
bash deploy/fr3_sharpa_runtime/serve.sh --check-only
bash deploy/fr3_sharpa_runtime/serve.sh --smoke-only --report /tmp/sharpa-smoke.json
```

这里不需要 hardware repo、Home 或 ROS。`check-only` 检查元数据和 stats；GPU smoke 实际加载并对假观测推理。两者都不证明任务成功。不要在仅研究模型时使用 `hardware.sh up` 或 `--execute`。

### 准备现场部署

先按硬件工程的迁移说明准备完整 portable_deps、Docker 镜像 `franka-upper-body-teleop:latest`、ROS workspace 和现场配置。当前部署会使用硬件仓库 `docker/compose.yaml`、`docker/.env`、`config/modes/teleop_control.yaml`、`config/litchibot/sharpa_schema.json` 及相机配置；镜像、CPU/IRQ 布局、FR3 地址、相机序列号和设备绑定必须按接收机器配置，不能直接沿用文档作者机器的 `.env`。

在部署仓库根目录生成本机路径文件：

```bash
.venv/bin/python deploy/fr3_sharpa_runtime/setup_local.py \
  --checkpoint /absolute/path/to/sharpa_tomato_ckpt_4000 \
  --gello /absolute/path/to/gello_upper_body_teleop \
  --python "$PWD/.venv/bin/python"
source deploy/fr3_sharpa_runtime/env.sh
"$SHARPA_PYTHON" -m deploy.fr3_sharpa_runtime.home_pose \
  --source /absolute/path/to/poses.yaml --task powder_weighing
```

只有当源 Home 是 root 专属权限、且已有本项目指定镜像时，才按 runtime 说明加 `--via-docker` 进行只读同步。将维护者交接并经现场确认的固定手部文件放到 `.deployment/hand_home.json`。其默认路径由 `run.py` 的 `--hand-home` 定义，双臂快照由 `--home-poses` 定义；名称必须与 `protocol.py::SHARPA_JOINT_NAMES` 一致。

此时 `.deployment/` 至少应具备：

```text
.deployment/
  local.env                  # setup_local.py 生成，机器路径
  teleop_openpi.yaml          # 从本机网关配置派生，仅增加 openpi 允许来源
  operator_poses.yaml         # 双臂 Home 快照
  operator_poses.source.json  # 同步工具生成的来源和 SHA256
  hand_home.json             # 另行交接/现场标定的双手固定 Home
```

随后运行离线检查和消息构建：

```bash
bash deploy/fr3_sharpa_runtime/build_messages.sh
bash deploy/fr3_sharpa_runtime/hardware.sh check
bash deploy/fr3_sharpa_runtime/runtime.sh --check --speed slow
"$SHARPA_PYTHON" -m pytest -q deploy/fr3_sharpa_runtime/tests
```

这些命令不代替现场验收。确认配置和设备身份后，按 [runtime 说明](../deploy/fr3_sharpa_runtime/README.md#现场运行)依次启动硬件、模型服务及首次单轮执行。连续模式是 `--num-chunks 0`，每轮默认预测 50、执行 15 后重推，没有自动任务完成检测。当前执行终端退出会清理其所属硬件容器，模型终端独立停止；不要按早期版本的退出行为操作。

## 训练参考与权重溯源

训练参考代码可单独取得，用于阅读和建立新实验：

```bash
git clone https://gitlab.litchibot.top/fruit-train/tomato-openpi.git
cd tomato-openpi
git checkout 68440ed8b39693b5fd206403843901187f453c1c
```

| 文件 | 可核对内容 |
| --- | --- |
| `training_configs/experiments/tomato_sharpa/baseline/tomato_fr3_sharpa_fullft_v1.py` | baseline 配置名 `pi05_fr3_sharpa_fulltask_fullft_v1_absolute_30hz`，实验名 `tomato_sharpa_fulltask_fullft_v1_absolute_30hz_s42`，默认数据 `data/tomato_sharpa_fulltask_v1` |
| `training_configs/recipes/pi05_fullft_v1.py` | `gemma_2b` + `gemma_300m`、无 LoRA、全参、全局 batch 64、预算 30,000 步 |
| `training_configs/recipes/common.py` | bfloat16、58D 本体投影、horizon 50、token 256、warmup 1000、LR peak 2.5e-5 / end 2.5e-6、cosine 30000、AdamW b1=0.9/b2=0.95/eps=1e-8、weight decay 1e-10、clip 1.0、无 EMA；model/seed=42、data=42001、augmentation=42002、epoch_shuffle、workers 8、每 2000 步保存 |
| `training_configs/embodiments/fr3_sharpa.py` | 左臂7+左手22+右臂7+右手22、rad、绝对目标、30 Hz，base=cam0/left_wrist=cam2/right_wrist=cam1 |
| `scripts/run_training.py`、`scripts/python.sh` | 实验模块加载与训练入口；训练环境和数据准备遵循该工程自己的 README |

原始 116D 存储状态在部署数据适配中按 `[0:7, 28:50, 14:21, 72:94]` 提取位置，原始 action58 的重排为 `[0:7, 14:36, 7:14, 36:58]`；实时 ROS 已是 canonical 58D，不再重排。上述固定版本 baseline 直接要求 canonical 58D，它不是当前 AB4 原始数据预处理实验的完整记录。

参考 baseline 的 asset_id 是 `tomato_sharpa_fulltask_v1_absolute_30hz`；当前权重的 asset_id 是 `tomato_sharpa_ab4_normal_rgb_absolute_v1_30hz`。目前只能确认推理参数结构、配套统计和现有代码路径，不能从这两个相似名字推断它们使用同一数据、同一采样或同一训练模块。

要复现 4000 步训练，提供者还必须交出：实际 AB4 实验模块及 commit、完整 CLI/环境覆盖参数、A/B 文件清单与版本/采样、归一化审计及摘要、真实 tasks.jsonl、初始基础权重身份、训练日志及 checkpoint 恢复信息。当前仓库没有这些完整记录，因此不能给出保证复现该权重的训练命令。

## 当前加载代码与历史 LoRA 的区别

当前 `serve.sh` → `serve.py` → `model_loader.build_config()` → `create_trained_policy()`。部署配置名为 `pi05_fr3_sharpa_ab4_normal_rgb_fullft_absolute_30hz_deploy`，只用于推理身份；训练参考 clone 不需要处在某个固定绝对路径，也不参与当前服务的 import。

`src/openpi/training/config.py` 保留了历史 Sharpa LoRA 配置：`pi05_fr3_sharpa`（30 Hz）、`pi05_fr3_sharpa_a04_b05_c01_30hz`。它们包含 LoRA 参数分支、不同数据身份，以及双臂 delta 转换；直接套用当前 full/absolute 权重会产生结构或语义不匹配。研究 LoRA 应建立单独加载配置和验证统计/动作契约，不能只改路径或关闭检查。

## 查阅移除前的 Wuji 实现

清理提交：`c60b4c2612020d641ed3003a82cd8ab5d143f4fc`。清理前完整版本：`bbe7f697e17a220ca91fae10b1812322b0741fa2`。在部署仓库内只读查询，无需切分支或覆盖当前工作区：

```bash
git show bbe7f697e17a220ca91fae10b1812322b0741fa2:deploy/fr3_wuji_fast/README.md
git show bbe7f697e17a220ca91fae10b1812322b0741fa2:deploy/fr3_wuji_medium/README.md
git ls-tree -r --name-only bbe7f697e17a220ca91fae10b1812322b0741fa2 -- deploy/ experiments/
```

浅克隆缺少历史对象时需先补齐 Git 历史。旧目录包括 `deploy/fr3_wuji*` 和 `experiments/weight_motion_eval`；RTG、Ruckig、Wuji 关节映射和设备限值属于旧实现，不是当前 Sharpa 已移植能力。先阅读 [保留的设计对照](reference_wuji_architecture.md)，再按当前 Sharpa 协议开展研究。

## 最低交接验收

接收者应能在不访问作者 home 目录的条件下完成：独立 Python 环境安装、checkpoint/stats 检查、GPU smoke、真实样本 dry-run、双臂和双手 Home 资产核对、Compose 解析及消息导入。提供者应交付这些结果与三份仓库版本；现场试验另记设备身份、参数、执行 trace 和实际结果。`COMPLETE` 只表示指令下发完成，不能写成任务成功或可从任意姿态无碰撞运行。
