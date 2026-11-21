"""Hardware-free deployment recipes, checkpoint evidence and run allocation."""
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from deploy.fr3_sharpa_runtime.config import DEFAULT_PROMPT


KNOWN_STATS = {
    "91c0295eea14bd2f1004695dd13a30eb052a96a124cbd67f65d9732a90e1032b": "absolute",
    "4f760f029a44ea0c0e25d4001ccd74d7aeb0a925403d0a60e48bb03ed3a78551": "delta",
}
MODES = {"serial": "普通串行", "rtc": "RTC", "paper-rtc": "Paper RTC", "train-rtc": "Train RTC · 训练前缀条件", "train-paper-rtc": "Train Paper RTC · 训练前缀＋Paper", "fovea": "Fovea · 动作与预测框"}


def repository_info(path):
    repo = Path(path).expanduser().resolve()
    branch = subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL).strip()
    modes = ["serial", "rtc"]
    if (repo / "deploy/fr3_sharpa_runtime/fovea_serve.sh").is_file():
        modes.append("fovea")
    if (repo / "deploy/fr3_sharpa_runtime/train_rtc.py").is_file():
        modes.append("train-rtc")
    if (repo / "deploy/fr3_sharpa_runtime/train_paper_rtc.py").is_file():
        modes.append("train-paper-rtc")
    if (repo / "deploy/fr3_sharpa_runtime/paper_rtc.py").is_file():
        modes.append("paper-rtc")
    return {"branch": branch, "modes": modes}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def desktop():
    try:
        result = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, timeout=3)
        path = Path(result.stdout.strip())
        if result.returncode == 0 and path.is_absolute() and path.is_dir():
            return path
    except (OSError, subprocess.SubprocessError):
        pass
    for name in ("桌面", "Desktop"):
        if (Path.home() / name).is_dir():
            return Path.home() / name
    return Path.home() / "Desktop"


def allocate_run(root, *, now=None):
    moment = now or datetime.now().astimezone()
    day = Path(root).expanduser().resolve() / moment.strftime("%Y-%m-%d")
    day.mkdir(parents=True, exist_ok=True)
    label, attempt = moment.strftime("%H.%M"), 1
    while True:
        directory = day / (label if attempt == 1 else f"{label}-{attempt}")
        try:
            directory.mkdir()  # Atomic reservation, including concurrent launchers.
            return directory
        except FileExistsError:
            attempt += 1


def inspect_checkpoint(checkpoint, *, training_record="", semantics="auto", cache=None, train_rtc_max_delay=0, model_type="auto", image_input="auto"):
    if image_input not in ("auto", "three", "four"):
        raise ValueError("图像输入请选择自动核对、三路或四路")
    choices = {"auto": None, "full": "absolute", "full-delta": "delta", "lora": "absolute",
               "paper-delta": "delta", "train-rtc-absolute": "absolute", "train-rtc-delta": "delta",
               "train-paper-rtc": "delta", "fovea": "delta"}
    if model_type not in choices:
        raise ValueError("不支持的模型类型")
    manual = model_type != "auto"
    if manual:
        semantics = choices[model_type]
        if "train" not in model_type and train_rtc_max_delay:
            raise ValueError("训练RTC最大延迟仅适用于Train RTC模型")
    root = Path(checkpoint).expanduser().resolve()
    manifest, metadata = root / "params/manifest.ocdbt", root / "params/_METADATA"
    if not manifest.is_file() or not metadata.is_file():
        raise ValueError("请选择直接包含 params/ 和 assets/ 的完整 checkpoint 步骤目录")
    paths = list((root / "assets").rglob("norm_stats.json"))
    if len(paths) != 1:
        raise ValueError("checkpoint assets/ 必须包含唯一配套 norm_stats.json")
    stats = paths[0]
    values = json.loads(stats.read_text())["norm_stats"]
    entries = json.loads(metadata.read_text())["tree_metadata"]
    from deploy.fr3_sharpa_runtime.paper_contract import inspect_paper, validate_stats, algorithm_identity
    algorithm = inspect_paper(stats, entries)
    validate_stats(values, paper_algorithm=algorithm)
    from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile, validate_parameter_layout
    lora = any("lora" in key.lower() for key in entries)
    if algorithm and lora:
        raise ValueError("Paper目前仅支持全量delta，不能套用LoRA")
    if manual:
        if (model_type == "lora") != lora:
            raise ValueError("手动模型类型与LoRA/全量参数结构不一致")
        if (model_type in ("paper-delta", "train-paper-rtc")) != bool(algorithm):
            raise ValueError("手动模型类型与Paper归一化/噪声资产不一致")
    validate_parameter_layout(entries, get_model_profile("lora" if lora else "full"))
    projections = {}
    for entry in entries.values():
        names = [part["key"] for part in entry.get("key_metadata", ())]
        if "action_in_proj" in names and "kernel" in names:
            projections["input"] = entry["value_metadata"]["write_shape"][0]
        if "action_out_proj" in names and "bias" in names:
            projections["output"] = entry["value_metadata"]["write_shape"]
    if projections != {"input": 58, "output": [58]}:
        raise ValueError("仅支持原生58D模型投影；不能按目录名称假定模型结构")
    if (root / "assets/fovea_contract.json").is_file() or model_type == "fovea":
        if model_type not in ("auto", "fovea") or semantics == "absolute" or lora or algorithm or train_rtc_max_delay:
            raise ValueError("Fovea联合动作框权重必须使用Fovea类型，不能覆盖为普通动作模型或RTC")
        from deploy.fr3_sharpa_runtime.fovea_contract import checkpoint_contract
        contract = checkpoint_contract(root)
        if training_record:
            from deploy.fr3_sharpa_runtime.train_rtc_contract import load_training_record
            record = load_training_record(Path(training_record).expanduser(), stats)
            model_fields = record["config"]["fields"]["model"]["fields"]
            data_fields = record["config"]["fields"]["data"]["fields"]
            if (model_fields.get("execute_steps"), model_fields.get("action_dim"), model_fields.get("action_horizon"),
                    data_fields.get("extra_delta_transform"), record["inputs"].get("norm_stats_sha256")) != (15, 58, 50, True, fingerprint(stats)):
                raise ValueError("训练记录与Fovea 50/15 Delta契约不一致")
        if image_input == "three":
            raise ValueError("Fovea需要四路图像输入，不能选择三路")
        return {"tomato_model": None, "input_views": 4, "fovea": contract, "train_rtc": None, "paper_algorithm": None,
                "supported_modes": ["fovea"], "normalization": "global_quantile",
                "checkpoint": str(root), "norm_stats": str(stats), "identity": contract["fovea_identity"],
                "manifest_sha256": fingerprint(manifest), "stats_sha256": fingerprint(stats),
                "finetune_mode": "full", "semantics": "delta", "profile": "full-delta",
                "evidence": "Fovea联合动作框契约、参数结构与归一化指纹已匹配",
                "action_dim": 58, "action_horizon": 50, "action_hz": 30}
    from deploy.fr3_sharpa_runtime.train_rtc_contract import inspect_train_rtc
    trained = inspect_train_rtc(stats, root, training_record or None, paper_algorithm=algorithm,
                                manual_delay=train_rtc_max_delay, manual_semantics=semantics, model_type=model_type)
    if trained and lora:
        raise ValueError("Train RTC requires full checkpoint, excludes LoRA")
    stats_sha, manifest_sha = fingerprint(stats), fingerprint(manifest)
    from deploy.fr3_sharpa_runtime.tomato_model import identify
    tomato = identify(stats_sha, manifest_sha, fingerprint(metadata))
    # These are the input routes implemented by the loader. A path name or
    # normalization shared by two experiments cannot activate a fourth camera.
    input_views = 4 if tomato and tomato.get("online_roi_required") else 3
    if image_input != "auto" and {"three": 3, "four": 4}[image_input] != input_views:
        raise ValueError("所选图像输入与已接入的权重加载配置不匹配；四路模型需核验局部图配置，不能按数据条数推断")
    if tomato and (trained or algorithm or (manual and model_type != "full-delta") or semantics == "absolute"):
        raise ValueError("已识别的 B Coarse 必须使用普通 Full Delta，不能覆盖为 Absolute 或训练 RTC")
    identity = hashlib.sha256((manifest_sha + stats_sha + algorithm_identity(algorithm) + (trained["record_sha256"] if trained else "")).encode()).hexdigest()
    detected, source = KNOWN_STATS.get(stats_sha), "已核对的训练统计指纹"
    if tomato:
        detected, source = "delta", "已核对 B Coarse checkpoint 配套统计与模型指纹"
    if algorithm:
        detected, source = "delta", "完整Paper资产、噪声参数和双臂delta训练契约已匹配"
    if trained:
        detected, source = trained["semantics"], ("操作员手动确认训练RTC及最大延迟" if trained.get("manual_confirmation") else "匹配统计与结构的Train RTC训练记录")
    cached_confirmation = False
    if training_record:
        from deploy.fr3_sharpa_runtime.train_rtc_contract import load_training_record
        record = load_training_record(Path(training_record).expanduser(), stats, paper_algorithm=algorithm)
        config = record["config"]["fields"]
        model, data = config["model"]["fields"], config["data"]["fields"]
        if (model["action_dim"], model["action_horizon"], model["max_token_len"], data["action_hz"]) != (58, 50, 256, 30):
            raise ValueError("训练记录不是58D/50步/token256/30Hz")
        expected = ("gemma_2b_lora", "gemma_300m_lora") if lora else ("gemma_2b", "gemma_300m")
        if (model["paligemma_variant"], model["action_expert_variant"]) != expected:
            raise ValueError("训练记录模型变体与权重参数结构不一致")
        if record["inputs"].get("norm_stats_sha256") != stats_sha:
            raise ValueError("训练记录的归一化统计摘要与 checkpoint 不一致")
        if algorithm and not data["extra_delta_transform"]:
            raise ValueError("训练记录与Paper资产的delta语义冲突")
        detected = "delta" if data["extra_delta_transform"] else "absolute"
        source = "匹配统计和模型结构的训练记录"
    if manual:
        if training_record and detected and detected != semantics:
            raise ValueError("手动模型类型与提供的训练记录冲突")
        detected, source = semantics, "操作员手动选择模型类型（训练JSON可选，无需契约副本）"
    if cache and not detected:
        binding = Path(cache) / f"{identity}.json"
        if binding.is_file():
            detected = json.loads(binding.read_text())["semantics"]
            if detected not in ("absolute", "delta"):
                raise ValueError("该权重保存的语义绑定无效，请核对训练记录")
            source = "该权重此前保存的操作员确认"
            cached_confirmation = True
    if semantics not in ("auto", "absolute", "delta"):
        raise ValueError("动作语义仅支持auto/absolute/delta")
    if semantics != "auto":
        if detected and detected != semantics and not cached_confirmation:
            raise ValueError("手动选择与已有训练证据冲突，请核对训练配置")
        if not detected or cached_confirmation:
            detected, source = semantics, "操作员按训练配置手动确认"
    if detected and lora and detected == "delta":
        raise ValueError("当前新部署配置尚不支持LoRA delta；不会套用全量delta配置")
    supported_modes = ["serial"]
    if not lora:
        supported_modes += ["paper-rtc" if algorithm else "rtc"]
    if trained:
        supported_modes = ["serial", "train-paper-rtc" if algorithm else "train-rtc"]
    if tomato and tomato.get("online_roi_required"):
        supported_modes = ["serial", "rtc"]
    return {"tomato_model": tomato, "input_views": input_views, "train_rtc": trained, "supported_modes": supported_modes,
            "paper_algorithm": algorithm, "normalization": algorithm["mode"] if algorithm else "global_quantile",
            "checkpoint": str(root), "norm_stats": str(stats), "identity": identity,
            "manifest_sha256": manifest_sha, "stats_sha256": stats_sha,
            "finetune_mode": "lora" if lora else "full", "semantics": detected,
            "evidence": source if detected else "无法从参数识别动作语义：请提供训练JSON或明确确认",
            "profile": ("lora" if lora else "full-delta" if detected == "delta" else "full") if detected else None,
            "action_dim": 58, "action_horizon": 50, "action_hz": 30}


@dataclass(frozen=True)
class DeploymentSettings:
    repository: str
    checkpoint: str
    mode: str = "serial"
    initialization: str = "current"
    speed: str = "slow"
    steps: int = 15
    rounds: int = 0
    record: bool = False
    output: str = ""
    training_record: str = ""
    semantics: str = "auto"
    camera_policy: str = "exclusive"
    train_rtc_max_delay: int = 0
    model_type: str = "auto"
    rtc_initial_delay_steps: int = 0
    task_mode: str = "none"
    initial_stage: str = "pickup_1"
    recovery_confirmed: bool = False
    crop_locator: str = ""
    crop_device: str = "cuda"
    fovea_stage: str = "pickup"
    prompt: str = DEFAULT_PROMPT
    image_input: str = "auto"

    def plan(self, *, cache=None, machine_root=None):
        repo = Path(self.repository).expanduser().resolve()
        from deploy.fr3_sharpa_runtime.task_stages import stage
        task_profile = self.task_mode if self.task_mode != "none" else "tomato-b"
        stage(self.initial_stage, task_profile)
        first_stage = "pickup" if task_profile == "tomato-in-hand119" else "pickup_1"
        if self.task_mode not in ("none", "tomato-b", "tomato-in-hand119") or type(self.recovery_confirmed) is not bool:
            raise ValueError("任务模式或人工恢复确认无效")
        if not isinstance(self.prompt, str) or (self.task_mode == "none" and self.mode != "fovea" and not self.prompt.strip()):
            raise ValueError("统一 Prompt 部署需要填写训练时使用的英文指令")
        if self.initial_stage != first_stage and (self.task_mode == "none" or not self.recovery_confirmed):
            raise ValueError("非首次阶段启动需要明确人工恢复确认并重新核对配置")
        if self.camera_policy not in ("exclusive", "require_free"):
            raise ValueError("相机策略仅支持独占接管或要求设备空闲")
        if self.mode not in MODES or self.initialization not in ("home", "current") or self.speed not in ("slow", "medium", "fast"):
            raise ValueError("不支持的模式、启动姿态或速度")
        if self.mode == "fovea":
            stage(self.fovea_stage, "tomato-in-hand119")
            if self.speed != "fast" or self.steps != 15 or self.task_mode not in ("none", "tomato-in-hand119") or self.rtc_initial_delay_steps:
                raise ValueError("Fovea要求原速30Hz、15步；仅支持独立指令或五阶段Agent任务，不支持RTC")
        if type(self.rounds) is not int or self.rounds < 0 or type(self.steps) is not int or type(self.record) is not bool:
            raise ValueError("轮数/步数必须是整数，录制必须是布尔开关")
        if self.mode == "serial" and not 1 <= self.steps <= 50:
            raise ValueError("串行每轮执行动作数必须在1..50")
        if self.mode in ("rtc", "paper-rtc", "train-rtc", "train-paper-rtc") and (self.speed != "fast" or not 1 <= self.steps <= 24 or self.rounds == 1):
            raise ValueError("RTC需fast、1..24重规划触发节点，轮数0或至少2")
        if type(self.train_rtc_max_delay) is not int or not 0 <= self.train_rtc_max_delay < 50:
            raise ValueError("训练RTC最大延迟必须为0..49整数，0表示自动识别")
        if type(self.rtc_initial_delay_steps) is not int or not 0 <= self.rtc_initial_delay_steps <= 25:
            raise ValueError("RTC初始延迟必须为0..25整数，0表示使用训练上限（最多25）")
        if self.rtc_initial_delay_steps and self.mode not in ("train-rtc", "train-paper-rtc"):
            raise ValueError("RTC初始延迟仅适用于Train RTC执行模式")
        model = inspect_checkpoint(self.checkpoint, training_record=self.training_record, semantics=self.semantics,
                                   cache=cache, train_rtc_max_delay=self.train_rtc_max_delay, model_type=self.model_type,
                                   image_input=self.image_input)
        if self.crop_device not in ("cuda", "cpu"):
            raise ValueError("局部图设备仅支持cuda/cpu")
        if model.get("tomato_model") and model["tomato_model"].get("online_roi_required"):
            from deploy.fr3_sharpa_runtime.interaction_crop import locator_identity
            locator, locator_sha = locator_identity(self.crop_locator or repo / ".deployment/crop_locator")
            model["crop_locator"] = locator
            model["crop_locator_sha256"] = locator_sha
            model["identity"] = hashlib.sha256((model["identity"] + locator_sha + self.crop_device).encode()).hexdigest()
            model["supported_modes"] = ["serial", "rtc"]
        if not model["profile"]:
            raise ValueError(model["evidence"])
        if self.task_mode != "none" and self.mode != "fovea":
            if (self.mode != "rtc" or model["profile"] != "full-delta" or not model.get("tomato_model")
                    or model["train_rtc"] or model["paper_algorithm"] or self.rounds != 0):
                raise ValueError("番茄 B 任务要求已识别 B Coarse 权重、普通 Full Delta RTC、连续轮数0")
            if model["tomato_model"].get("task_profile", "tomato-b") != self.task_mode:
                raise ValueError("权重与任务流程不匹配：放回盘中调整使用七阶段，手中调整使用五阶段")
            if not (repo / "deploy/fr3_sharpa_runtime/task_execution.py").is_file():
                raise ValueError("所选仓库缺少番茄任务执行接口")
        if self.mode == "fovea" and self.task_mode != "none":
            if not model.get("fovea") or self.rounds != 0:
                raise ValueError("Fovea五阶段任务要求Fovea权重和连续轮数0")
            if not (repo / "deploy/fr3_sharpa_runtime/fovea_task.py").is_file():
                raise ValueError("所选仓库缺少Fovea Agent阶段协调器")
        if self.mode not in model["supported_modes"]:
            raise ValueError("该模型不支持所选推理模式；请按识别结果选择RTC、Paper RTC、Train RTC或Train Paper RTC；LoRA仅支持普通串行")
        if model["paper_algorithm"] and not (repo / "deploy/fr3_sharpa_runtime/paper_model.py").is_file():
            raise ValueError("所选仓库尚未同步Paper权重适配，请更新该分支")
        shared = "deploy/fr3_sharpa_runtime/"
        if not (repo / shared / "camera_startup.py").is_file():
            raise ValueError("所选部署目录未接入启动前相机闸门；请同步该分支后再使用本控制台")
        service, runtime = shared + "serve.sh", shared + "runtime.sh"
        service_args, runtime_args = [], []
        port = {"full": 18765, "lora": 18766, "full-delta": 18767}[model["profile"]]
        description = f"预测50，执行前{self.steps}，再观测推理"
        if self.mode == "fovea":
            service = shared + "fovea_serve.sh"
            port = 8010
            service_args = ["--checkpoint", model["checkpoint"], "--port", str(port)]
            runtime_args = ["--fovea", "--policy-url", f"ws://127.0.0.1:{port}",
                            "--prompt", stage(self.fovea_stage, "tomato-in-hand119").prompt]
            description = f"Fovea · {stage(self.fovea_stage, 'tomato-in-hand119').label}：预测50、执行15后更新框；停止后选择下一阶段"
        if self.mode == "fovea" and self.task_mode != "none":
            description = "Fovea五阶段：完成15步后切换；可选Agent自动推进，不使用RTC"
        if self.mode in ("rtc", "paper-rtc", "train-rtc", "train-paper-rtc"):
            service_args = [{"paper-rtc": "--enable-paper-rtc", "train-rtc": "--enable-train-rtc", "train-paper-rtc": "--enable-train-paper-rtc", "rtc": "--enable-rtc"}[self.mode]]
            runtime_args = ["--broker-mode", "rtc"]
            if self.mode in ("train-rtc", "train-paper-rtc"):
                if not (repo / shared / "train_rtc.py").is_file():
                    raise ValueError("所选仓库缺少Train RTC适配")
                if self.rtc_initial_delay_steps > min(25, model["train_rtc"]["max_delay"]):
                    raise ValueError("RTC初始延迟不能超过训练最大延迟或25")
                runtime_args += ["--train-paper-rtc" if self.mode == "train-paper-rtc" else "--train-rtc", "--rtc-initial-delay-steps", str(self.rtc_initial_delay_steps or min(25, model["train_rtc"]["max_delay"]))]
            if self.mode == "train-paper-rtc" and not (repo / shared / "train_paper_rtc.py").is_file():
                raise ValueError("所选仓库缺少Train Paper RTC适配")
            if self.mode == "paper-rtc":
                if not (repo / shared / "paper_rtc.py").is_file():
                    raise ValueError("所选仓库尚未同步Paper RTC适配类")
                runtime_args.append("--paper-rtc")
            description = f"50点时间线，{self.steps}源节点触发后台重规划（非固定执行前缀）"
        if not (repo / service).is_file() or not (repo / runtime).is_file():
            raise ValueError("所选仓库没有该模式入口，请选择对应分支的部署目录")
        if self.record:
            worker = "sdk_direct_motion_ros.py"
            if "RecordingGate" not in (repo / shared / worker).read_text():
                raise ValueError("所选仓库尚未接入chunk录制钩子，请先同步UI/录制版本")
        profile = model["profile"]
        common = ["--model", profile]
        machine = Path(machine_root).resolve() if machine_root else repo
        home_files = {}
        for argument, filename in (("--home-poses", "operator_poses.yaml"), ("--hand-home", "hand_home.json")):
            local, shared_home = repo / ".deployment" / filename, machine / ".deployment" / filename
            home_files[argument] = str(local if local.is_file() else shared_home)
        if self.initialization == "home":
            from deploy.fr3_sharpa_runtime.home_pose import load_home
            load_home(Path(home_files["--home-poses"]), "powder_weighing", hand_path=Path(home_files["--hand-home"]))
        options = [*common, *runtime_args, "--initialization", self.initialization, "--speed", self.speed,
                   "--execute-steps", str(self.steps), "--num-chunks", str(self.rounds)]
        if self.task_mode != "none":
            options += ["--task-mode", self.task_mode, "--initial-stage", self.initial_stage]
        elif self.mode != "fovea":
            options += ["--prompt", self.prompt]
        for argument, path in home_files.items():
            options += [argument, path]
        branch = subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True).strip()
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        result = {"settings": {**asdict(self), "repository": str(repo), "output": str(Path(self.output).expanduser().resolve()) if self.output else str(desktop() / "Sharpa部署记录")},
                "model": model, "branch": branch, "commit": commit, "description": description, "port": port,
                "home_files": home_files, "machine_root": str(machine),
                "home_sha256": {key: fingerprint(path) for key, path in home_files.items()} if self.initialization == "home" else {},
                "hardware": ["bash", "deploy/fr3_sharpa_runtime/hardware.sh", "up",
                             *(["--exclusive"] if self.camera_policy == "exclusive" else []),
                             *(["--defer-camera-check"] if self.initialization == "home"
                               and (repo / shared / "startup_gate.py").is_file() else [])],
                "service": ["bash", service, *([] if self.mode == "fovea" else common), *service_args],
                "check": ["bash", runtime, *options, "--check"],
                "execute": ["bash", runtime, *options, "--execute", "--enable-hardware", "--sharpa-backend", "sdk_direct"]}
        result["approval"] = hashlib.sha256(json.dumps({"settings": result["settings"], "identity": model["identity"],
            "profile": model["profile"], "commit": commit, "branch": branch, "home_sha256": result["home_sha256"]}, sort_keys=True).encode()).hexdigest()
        return result


def plan_environment(plan):
    """Explicit child settings win over machine defaults and inherited old runs."""
    from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile
    profile = get_model_profile(plan["model"]["profile"])
    env = dict(os.environ)
    env.pop("SHARPA_CAMERA_GATE_FD", None)  # Only the current hardware supervisor may grant admission.
    # Machine/SDK paths stay; inference identity and service port are per plan.
    env.update({profile.checkpoint_env: plan["model"]["checkpoint"],
                profile.norm_stats_env: plan["model"]["norm_stats"],
                profile.port_env: str(plan["port"]),
                "SHARPA_MODEL_PROFILE": profile.name,
                "SHARPA_DEPLOY_MODEL_TYPE": plan["settings"].get("model_type", "auto")})
    root = plan["settings"]["repository"]
    env.pop("SHARPA_FOVEA_CHECKPOINT", None)
    if plan["model"].get("fovea"):
        env["SHARPA_FOVEA_CHECKPOINT"] = plan["model"]["checkpoint"]
    for key in ("SHARPA_CROP_LOCATOR", "SHARPA_CROP_DEVICE", "SHARPA_CROP_STATUS_DIR", "SHARPA_SHARED_ROI_DIR", "SHARPA_FOUR_VIEW_RECORD_DIR"):
        env.pop(key, None)
    if plan["model"].get("crop_locator"):
        env.update(SHARPA_CROP_LOCATOR=plan["model"]["crop_locator"],
                   SHARPA_CROP_DEVICE=plan["settings"]["crop_device"],
                   SHARPA_CROP_STATUS_DIR=str(Path(root)/".deployment/crop_live"),
                   SHARPA_SHARED_ROI_DIR=str(Path(root)/".deployment/shared_roi"))
    env.setdefault("SHARPA_PYTHON", sys.executable)
    overlay = Path(plan["machine_root"]) / ".deployment/sharpa_ros_messages/install"
    if overlay.is_dir():
        env.setdefault("SHARPA_ROS_MESSAGES_OVERLAY", str(overlay))
    env["PYTHONPATH"] = f"{root}:{root}/src:{root}/packages/openpi-client/src"
    for key in ("SHARPA_DEPLOY_TRAINING_RECORD", "SHARPA_TRAIN_RTC_RECORD", "SHARPA_TRAIN_RTC_MAX_DELAY", "SHARPA_TRAIN_RTC_SEMANTICS"):
        env.pop(key, None)
    trained = plan["model"].get("train_rtc")
    if trained:
        if trained["record"]:
            env["SHARPA_TRAIN_RTC_RECORD"] = trained["record"]
            if plan["settings"].get("model_type", "auto") != "auto":
                env["SHARPA_DEPLOY_TRAINING_RECORD"] = trained["record"]
        if not trained["record"] or plan["settings"].get("model_type", "auto") != "auto":
            env["SHARPA_TRAIN_RTC_MAX_DELAY"] = str(trained["max_delay"])
            env["SHARPA_TRAIN_RTC_SEMANTICS"] = trained["semantics"]
    env.pop("SHARPA_RECORD_SOCKET", None)
    env.pop("SHARPA_CAMERA_LEASE_FD", None)  # Never inherit another stale run's lease.
    env.pop("SHARPA_TASK_SOCKET", None)
    env.pop("SHARPA_TASK_RUN_ID", None)
    return env
