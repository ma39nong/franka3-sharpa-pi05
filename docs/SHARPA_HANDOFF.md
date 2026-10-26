# 新机器部署与研究交接

本文面向第一次拉仓库的人，区分离线模型研究、现场部署和训练复现。文档核对基线为本仓库 `b796025`；本机部署路径只是实例，迁移时使用自己的绝对路径。本文未在全新机器上重新安装环境；安装命令依据仓库的 pyproject.toml/uv.lock，已有机器的通过记录不等于新机器已验收。

## 仓库和职责

| 工程 | 来源与核对版本 | 责任与是否必需 |
| --- | --- | --- |
| 当前部署仓库 | `https://github.com/ma39nong/franka3-sharpa-pi05.git`，分支 `chunk-base-test` | 模型服务、观测、动作协议、Home、执行调度和日志；必需 |
| 训练工程 | `https://gitlab.litchibot.top/fruit-train/tomato-openpi.git`；实际 AB4 训练记录为 `42f887916d22cd58c022e95a000e2cb241a8a459`，本机 main 参考为 `68440ed8b39693b5fd206403843901187f453c1c` | 重建实验以实际源码快照和 UI 覆盖参数为准；仅加载已有权重不需要安装此工程 |
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
| 固定双手 Home | `.deployment/hand_home.json`；`unit: rad`，`tasks.powder_weighing.left/right` 各含 22 个 `joint_names` 和对应 `positions` | 显式 `--home` 回位时使用 |
| 数据与任务文本 | 有名称/频率元数据的 LeRobot 数据和 `meta/tasks.jsonl`；服务器路径不是下载地址 | 真实样本研究、效果评估 |
| 原始训练记录 | 28 服务器已找到 AB4 UI run.json、训练 JSON、源码快照和日志；见下方记录路径。仍需交接这些文件及其数据/环境依赖 | 重训复现 |

`train_state/` 用于训练恢复；当前推理加载 `params/` 和配套 stats，不需要 optimizer state。不要将另一实验的 stats 配给当前 checkpoint。

固定双手 Home 当前来自 2026-10-23 的现场采集：左手实测角用于左手，右手按同名关节使用相同数值，不做符号翻转。这是本次现场标定约定，不代表任何 Sharpa 装置都可直接采用。文件缺失时显式 Home 模式会失败；双臂快照同步工具不会自动生成该文件，也不会代替另一台设备的标定。

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

### 已找到的 AB4 实验记录

2026-10-25 只读检查 28 服务器后，找到了以下实际运行记录。原先“AB4 实验配置尚未取得”的结论已更新；本节的实际设置优先于上表的通用 baseline 默认值。

服务器工程为 `/home/tairan/workspace/code/openpi`，其物理路径是 `/mnt/data/users/tairan/workspace/code/openpi`。以下路径均相对此工程：

```text
logs/training_ui/2200a091b1ee48239ad11b203a4bf934/
  run.json
  state.json
  console.log

logs/pi05_fr3_sharpa_1016_2_multimodal_rgb_fullft_v1_absolute_30hz/
  tairan_tairan_tomato_sharpa_1016_2_multimodal_fullft_absolute_s42_1022_2100/
    20261022_210518_072235_3334021.json
    20261022_210518_072235_3334021.experiment.py
    20261022_210518_072235_3334021.log

assets/tomato_sharpa_ab4_normal_rgb_absolute_v1_30hz/
  norm_stats.json
  norm_metadata.json
```

UI 记录的实验模块为：

```text
training_configs.experiments.tomato_sharpa.template.tairan.tairan_tairan_tomato_sharpa_1016_2_multimodal_fullft_absolute_s42_1022_2100
```

该模块原来的 template 源文件目前不存在，使用上面的 `.experiment.py` 归档检查当次代码；不要用现有 main baseline 代替它。归档 JSON 记录的训练源码 commit 为 `42f887916d22cd58c022e95a000e2cb241a8a459`。重建实验需要该 commit、归档源码和 `run.json` 的运行覆盖值共同参与，不能只 checkout commit 就声称恢复了实验。

| 实际 UI 运行设置 | 记录值 |
| --- | --- |
| 实验名 | `tairan_tairan_tomato_sharpa_1016_2_multimodal_fullft_absolute_s42_1022_2100` |
| asset_id | `tomato_sharpa_ab4_normal_rgb_absolute_v1_30hz` |
| 动作监督 / horizon | absolute / 50 |
| 训练全局 batch / FSDP | **512 / 8** |
| 计划训练步数 / 保存间隔 | 30,000 / 2,000；不是已证明训练完成的步数 |
| 采样 | `epoch_shuffle`；不能据此写成另行指定的 A:B 加权比例 |
| LR peak / end / warmup / decay | 2.5e-5 / 2.5e-6 / 1000 / 30000 |
| seed / model_seed / data_seed / augmentation_seed | 42 / 42 / 42001 / 42002 |
| 梯度裁剪 | 1 |
| 读取时的 UI 状态 | stopped；不等同于按计划完成 30,000 步 |

UI `dataset_roots` 记录的四个完整路径如下，均为 normal：

```text
/mnt/data/users/mengxiang/workspace/data/sharpa/A/54_120_data_lerobot--batch-000001--e1f8de59549f8711/normal
/mnt/data/users/mengxiang/workspace/data/sharpa/A/121_430_data_lerobot--batch-000001--50a0ecd82e4177d5/normal
/mnt/data/users/mengxiang/workspace/data/sharpa/B/56_106_data_lerobot--batch-000001--ba765ddf94aef37c/normal
/mnt/data/users/mengxiang/workspace/data/sharpa/B/130_423_data_lerobot--batch-000001--2a48e3b9415b986c/normal
```

四个目录不是四组等概率采样的证明。应继续检查实际数据集合并、episode 数量、归一化审计及源码中的采样实现；不要从 AB4 命名推断 A:B 比例。

已完成的摘要核对：

| 核对对象 | SHA256 | 结果 |
| --- | --- | --- |
| 服务器 AB4 norm_stats.json 与本机 4000 步 checkpoint 自带统计 | `91c0295eea14bd2f1004695dd13a30eb052a96a124cbd67f65d9732a90e1032b` | 文件摘要一致 |
| 归档 experiment.py 与 UI run.json 的 config_sha256 | `0cff57cb1912243fde9ec9a77c4b3e643759aa42e30e4e7e6402b9967560d152` | 源码摘要一致 |

### 对话日志解释了什么

用户提供的历史对话记录了：main 的 58D baseline 拒绝 116D 数据；转用 `42f8879` 的 multimodal_rgb 适配；创建个人配置以便 UI 发现；早期 batch 64/FSDP 1 首步 OOM 后建议 batch 32/FSDP 8；随后仅将本地官方配置更新到 sharpa 分支，没有改动服务器既有实验。

这解释了本机新 baseline 与服务器训练源码可能不同。对话中的统计计算 batch 512 不是训练 batch 512 的证据；本次训练 batch 512 的依据是实际 UI run.json。该段对话未记录四批 AB4 normal 的最终切换及 4000 步权重导出，不能替代实际训练归档。

### 尚未闭环的复现材料

- `state.json` 记录的 checkpoint 旧路径位于 `code/openpi-sharpa-42f8879/checkpoints/.../tairan_tairan_tomato_sharpa_1016_2_multimodal_fullft_absolute_s42_1022_2100`，读取时该路径不存在。尚未将原始 step 4000 的全部参数文件与本机权重逐项比对。统计摘要相同不证明权重参数完全相同。
- 已取得四个数据根目录的记录，但本轮未验证全部数据文件和 episode 清单的完整性、内容摘要及相对采样占比。实际任务文本、基础权重身份和完整环境依赖应随归档一并核查。
- 对话提到在 UI 启动前设置 `XLA_PYTHON_CLIENT_PREALLOCATE=true` 和 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`；这只能证明当时讨论了设置方法，不能证明 AB4 训练主进程确实采用。UI 任务值、训练进程环境和统计计算 batch 需要分开记录。
- 这些归档目前保存在服务器，本文记录了位置和已核对指纹，尚未将原始文件打包进本部署仓库。交接时需取得文件本身，并核查日志/配置中的凭据或访问 token，不能只交一段服务器绝对路径。

因此已能补齐实际实验来源和关键设置，但尚不承诺逐位重训复现同一 4000 步权重。已下载权重的推理部署仍按当前 model_loader 和配套 stats 进行。

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

## 同仓库选择全量或 LoRA

`chunk-base-test` 现在共存两种模型配置；默认全量部署流程继续有效。共享入口 `serve.sh` 和 `runtime.sh` 可添加 `--model full` 或 `--model lora`，两端选择须一致；`serve.sh --list-models` 可离线查看模型注册。机器配置和 Home 共用，LoRA 独立使用 `.deployment/lora.env`，默认端口 18766。新的 30 Hz／58D absolute LoRA 尚无 checkpoint，权重接入、GPU smoke 和现场步骤见 [LoRA 部署说明](../deploy/fr3_sharpa_lora/README.md)。

## 中途恢复执行

当前 SDK direct 默认从启动后的实测状态和相机图像推理，不自动回 Home。需要回位时在执行命令后加 `--home`，到位停稳后继续任务；旧 `--initialization home` 等效。中途 Ctrl-C 的失能确认和容器清理流程保持不变，重启前需重新运行 `hardware.sh up`；从当前状态恢复的命令应去掉 `--home`、`--initialization home` 和 `--prepare-policy-first`。每次重启重新观测推理，旧 chunk 不复用。
