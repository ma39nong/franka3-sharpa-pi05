"""Hardware-free deployment recipes, checkpoint evidence and run allocation."""
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


KNOWN_STATS = {
    "91c0295eea14bd2f1004695dd13a30eb052a96a124cbd67f65d9732a90e1032b": "absolute",
    "4f760f029a44ea0c0e25d4001ccd74d7aeb0a925403d0a60e48bb03ed3a78551": "delta",
}
MODES = {"serial": "普通串行", "rtc": "RTC", "paper-rtc": "Paper RTC", "train-rtc": "Train RTC · 训练前缀条件", "train-paper-rtc": "Train Paper RTC · 训练前缀＋Paper", "soft-serial": "串行 Soft", "soft-async": "异步 Soft"}
MODES.update(flash="flash · 串行", flashRTC="flashRTC · 后台推理连续执行")


def repository_info(path):
    repo = Path(path).expanduser().resolve()
    branch = subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL).strip()
    modes = ["serial", "rtc"]
    if (repo / "deploy/fr3_sharpa_runtime/flash_contract.py").is_file():
        modes.append("flash")
    if (repo / "deploy/fr3_sharpa_runtime/flash_runtime.py").is_file():
        modes.append("flashRTC")
    if (repo / "deploy/fr3_sharpa_runtime/train_rtc.py").is_file():
        modes.append("train-rtc")
    if (repo / "deploy/fr3_sharpa_runtime/train_paper_rtc.py").is_file():
        modes.append("train-paper-rtc")
    if (repo / "deploy/fr3_sharpa_runtime/paper_rtc.py").is_file():
        modes.append("paper-rtc")
    if (repo / "deploy/fr3_sharpa_full/serve_soft.sh").is_file():
        modes.append("soft-serial")
    stream = repo / "deploy/fr3_sharpa_runtime/async_soft_stream.py"
    if stream.is_file() and "sharpa-async-soft-rolling-v3" in stream.read_text():
        modes.append("soft-async")
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


def inspect_checkpoint(checkpoint, *, training_record="", semantics="auto", cache=None, train_rtc_max_delay=0, model_type="auto"):
    choices = {"auto": None, "full": "absolute", "full-delta": "delta", "lora": "absolute",
               "paper-delta": "delta", "train-rtc-absolute": "absolute", "train-rtc-delta": "delta",
               "train-paper-rtc": "delta", "flash-absolute": "absolute"}
    if model_type not in choices:
        raise ValueError("不支持的模型类型")
    manual = model_type != "auto"
    if manual:
        semantics = choices[model_type]
        if "train" not in model_type and train_rtc_max_delay:
            raise ValueError("训练RTC最大延迟仅适用于Train RTC模型")
    root = Path(checkpoint).expanduser().resolve()
    from deploy.fr3_sharpa_runtime.flash_contract import declares_flash
    declared_flash = declares_flash(root)
    if model_type == "flash-absolute" or (model_type == "auto" and declared_flash):
        if semantics == "delta" or training_record or train_rtc_max_delay:
            raise ValueError("FLASH is Absolute only; excludes Delta/RTC training records")
        from deploy.fr3_sharpa_runtime.flash_contract import inspect_flash, digest_json
        found = inspect_flash(root, backend=os.environ.get("SHARPA_FLASH_REPO"))
        return {**found, "profile": "flash-absolute", "semantics": "absolute", "finetune_mode": "full",
            "identity": digest_json({key: found[key] for key in ("flash_manifest_sha256", "flash_backend_identity", "flash_adapter_identity")}),
            "manifest_sha256": found["checkpoint_manifest_sha256"], "stats_sha256": found["normalization_sha256"],
            "normalization": "flash_internal_quantile_absolute", "evidence": "FLASH K6 stride1 NFE1 history8；直接读取原始checkpoint，无需导出契约"}
    if declared_flash:
        raise ValueError("FLASH checkpoint requires flash-absolute model type")
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
    from deploy.fr3_sharpa_runtime.train_rtc_contract import inspect_train_rtc
    trained = inspect_train_rtc(stats, root, training_record or None, paper_algorithm=algorithm,
                                manual_delay=train_rtc_max_delay, manual_semantics=semantics, model_type=model_type)
    if trained and lora:
        raise ValueError("Train RTC requires full checkpoint, excludes LoRA")
    stats_sha, manifest_sha = fingerprint(stats), fingerprint(manifest)
    identity = hashlib.sha256((manifest_sha + stats_sha + algorithm_identity(algorithm) + (trained["record_sha256"] if trained else "")).encode()).hexdigest()
    detected, source = KNOWN_STATS.get(stats_sha), "已核对的训练统计指纹"
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
        supported_modes += ["paper-rtc" if algorithm else "rtc", "soft-serial", "soft-async"]
    if trained:
        supported_modes = ["serial", "train-paper-rtc" if algorithm else "train-rtc"]
    return {"train_rtc": trained, "supported_modes": supported_modes,
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
    flash_max_plan_age_ms: float = 0
    flash_seam_position_rad: float = 0
    flash_seam_velocity_rad_s: float = 0
    flash_feedback_error_rad: float = 0

    def plan(self, *, cache=None, machine_root=None):
        repo = Path(self.repository).expanduser().resolve()
        if self.camera_policy not in ("exclusive", "require_free"):
            raise ValueError("相机策略仅支持独占接管或要求设备空闲")
        if self.mode not in MODES or self.initialization not in ("home", "current") or self.speed not in ("slow", "medium", "fast"):
            raise ValueError("不支持的模式、启动姿态或速度")
        if type(self.rounds) is not int or self.rounds < 0 or type(self.steps) is not int or type(self.record) is not bool:
            raise ValueError("轮数/步数必须是整数，录制必须是布尔开关")
        if self.mode in ("serial", "flash") and not 1 <= self.steps <= 50:
            raise ValueError("串行每轮执行动作数必须在1..50")
        if self.mode in ("rtc", "paper-rtc", "train-rtc", "train-paper-rtc") and (self.speed != "fast" or not 1 <= self.steps <= 24 or self.rounds == 1):
            raise ValueError("RTC需fast、1..24重规划触发节点，轮数0或至少2")
        if self.mode.startswith("soft-") and self.steps != 26:
            raise ValueError("Soft源执行点固定26；压缩目标20不是execute-steps")
        if self.mode == "flashRTC":
            from deploy.fr3_sharpa_runtime.flash_stream import FlashSettings
            FlashSettings(self.flash_seam_position_rad, self.flash_seam_velocity_rad_s, self.flash_feedback_error_rad)
            if self.speed != "fast" or not 1 <= self.steps <= 24:
                raise ValueError("flashRTC需fast、1..24重规划触发节点")
        if type(self.train_rtc_max_delay) is not int or not 0 <= self.train_rtc_max_delay < 50:
            raise ValueError("训练RTC最大延迟必须为0..49整数，0表示自动识别")
        if type(self.rtc_initial_delay_steps) is not int or not 0 <= self.rtc_initial_delay_steps <= 25:
            raise ValueError("RTC初始延迟必须为0..25整数，0表示使用训练上限（最多25）")
        if self.rtc_initial_delay_steps and self.mode not in ("train-rtc", "train-paper-rtc"):
            raise ValueError("RTC初始延迟仅适用于Train RTC执行模式")
        if (type(self.flash_max_plan_age_ms) not in (float, int)
                or not math.isfinite(self.flash_max_plan_age_ms) or self.flash_max_plan_age_ms < 0):
            raise ValueError("FLASH计划年龄上限必须为非负有限毫秒数（0关闭附加检查）")
        model = inspect_checkpoint(self.checkpoint, training_record=self.training_record, semantics=self.semantics,
                                   cache=cache, train_rtc_max_delay=self.train_rtc_max_delay, model_type=self.model_type)
        if self.flash_max_plan_age_ms and model["profile"] != "flash-absolute":
            raise ValueError("FLASH计划年龄参数仅适用于FLASH模型")
        if not model["profile"]:
            raise ValueError(model["evidence"])
        if self.mode not in model["supported_modes"]:
            raise ValueError("该模型不支持所选推理模式；请按识别结果选择RTC、Paper RTC、Train RTC或Train Paper RTC；LoRA仅支持普通串行")
        if self.mode.startswith("soft-") and model["finetune_mode"] != "full":
            raise ValueError("Soft目前仅支持全量absolute/delta")
        if model["paper_algorithm"] and not (repo / "deploy/fr3_sharpa_runtime/paper_model.py").is_file():
            raise ValueError("所选仓库尚未同步Paper权重适配，请更新该分支")
        shared = "deploy/fr3_sharpa_runtime/"
        if not (repo / shared / "camera_startup.py").is_file():
            raise ValueError("所选部署目录未接入启动前相机闸门；请同步该分支后再使用本控制台")
        full = "deploy/fr3_sharpa_full/"
        service, runtime = shared + "serve.sh", shared + "runtime.sh"
        service_args, runtime_args = [], []
        port = {"full": 18765, "lora": 18766, "full-delta": 18767, "flash-absolute": 18768}[model["profile"]]
        description = f"预测50，执行前{self.steps}，再观测推理"
        if model["profile"] == "flash-absolute":
            service = shared + "serve_flash.sh"
            description = f"FLASH K6/stride1/NFE1/history8；预测50，串行执行前{self.steps}；支持连续多轮"
            runtime_args = ["--broker-mode", "flash"]
            if self.mode == "flashRTC":
                runtime_args = ["--broker-mode", "flashRTC", "--flash-seam-position-rad", str(self.flash_seam_position_rad),
                    "--flash-seam-velocity-rad-s", str(self.flash_seam_velocity_rad_s),
                    "--flash-feedback-error-rad", str(self.flash_feedback_error_rad)]
                description = f"flashRTC：{self.steps}节点触发后台推理；30Hz按观测相位切换；无RTC模型前缀条件；最后一代执行到触发进度结束"
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
        elif self.mode == "soft-serial":
            service = full + "serve_soft.sh"
            runtime_args = ["--inference-preset", "paper-rolling", "--inpainting", "soft", "--compression", "cubic", "--broker-mode", "serial"]
            description = "50→30→26+4；26→20 cubic（原串行实现可能回退26）"
        elif self.mode == "soft-async":
            service, runtime = full + "serve_async_soft.sh", full + "runtime_async_soft.sh"
            port += 100
            description = "50→30→26+4→20，固定PCHIP，后台预测，段边界切换"
            stream = repo / shared / "async_soft_stream.py"
            if not stream.is_file() or "sharpa-async-soft-rolling-v3" not in stream.read_text():
                raise ValueError("需要Soft分支rolling-v3；拒绝旧50点/15触发异步入口")
        if not (repo / service).is_file() or not (repo / runtime).is_file():
            raise ValueError("所选仓库没有该模式入口，请选择对应分支的部署目录")
        if self.record:
            worker = "async_soft_motion_ros.py" if self.mode == "soft-async" else "sdk_direct_motion_ros.py"
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
                               and self.mode != "soft-async" and (repo / shared / "startup_gate.py").is_file() else [])],
                "service": ["bash", service, *common, *service_args],
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
                profile.port_env: str(plan["port"] if plan["settings"]["mode"] != "soft-async" else plan["port"] - 100),
                "SHARPA_MODEL_PROFILE": profile.name,
                "SHARPA_DEPLOY_MODEL_TYPE": plan["settings"].get("model_type", "auto")})
    root = plan["settings"]["repository"]
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
    env["SHARPA_FLASH_MAX_PLAN_AGE_MS"] = str(plan["settings"].get("flash_max_plan_age_ms", 0))
    env.pop("SHARPA_RECORD_SOCKET", None)
    env.pop("SHARPA_CAMERA_LEASE_FD", None)  # Never inherit another stale run's lease.
    return env
