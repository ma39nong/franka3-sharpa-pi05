"""Hardware-free deployment recipes, checkpoint evidence and run allocation."""
from dataclasses import asdict
from dataclasses import dataclass
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
MODES = {"serial": "普通串行", "rtc": "RTC", "paper-rtc": "Paper RTC", "train-rtc": "Train RTC · 训练前缀条件", "train-paper-rtc": "Train Paper RTC · 训练前缀＋Paper"}


def repository_info(path):
    repo = Path(path).expanduser().resolve()
    branch = subprocess.check_output(["git", "-C", str(repo), "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL).strip()
    modes = ["serial", "rtc"]
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


def inspect_checkpoint(checkpoint, *, training_record="", semantics="auto", cache=None, train_rtc_max_delay=0, model_type="auto", image_input="auto", task_mode="none"):
    if semantics not in ("auto", "absolute", "delta"):
        raise ValueError("动作语义仅支持 auto/absolute/delta")
    if type(train_rtc_max_delay) is not int or not 0 <= train_rtc_max_delay < 50:
        raise ValueError("训练 RTC 最大延迟必须为0..49整数")
    if image_input not in ("auto", "three", "four"):
        raise ValueError("图像输入请选择自动核对、三路或四路")
    choices = {"auto": None, "full": "absolute", "full-delta": "delta", "lora": "absolute",
               "paper-delta": "delta", "train-rtc-absolute": "absolute", "train-rtc-delta": "delta",
               "train-paper-rtc": "delta"}
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
    from deploy.fr3_sharpa_runtime.paper_contract import algorithm_identity
    from deploy.fr3_sharpa_runtime.paper_contract import inspect_paper
    from deploy.fr3_sharpa_runtime.paper_contract import validate_stats
    algorithm = inspect_paper(stats, entries)
    validate_stats(values, paper_algorithm=algorithm)
    from deploy.fr3_sharpa_runtime.model_profiles import get_model_profile
    from deploy.fr3_sharpa_runtime.model_profiles import validate_parameter_layout
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
    from deploy.fr3_sharpa_runtime.deployment_options import discover
    from deploy.fr3_sharpa_runtime.deployment_options import input_route
    found = discover(root, training_record)
    warnings = found.pop("warnings")
    # Manual settings always win over optional historical evidence.
    detected = semantics if semantics != "auto" else ("delta" if algorithm else found.get("semantics") or KNOWN_STATS.get(fingerprint(stats)))
    if manual:
        detected = choices[model_type]
    delay = train_rtc_max_delay or (found.get("rtc_max_delay", 0) if not manual else 0)
    is_train = model_type.startswith("train-") if manual else bool(delay)
    if is_train and not 0 < delay < 50:
        raise ValueError("请填写训练 RTC 最大延迟（1..49）")
    if lora and (is_train or detected == "delta"):
        raise ValueError("当前加载器支持 LoRA Absolute 串行部署")
    stats_sha, manifest_sha = fingerprint(stats), fingerprint(manifest)
    from deploy.fr3_sharpa_runtime.tomato_model import identify
    try:
        historical = identify(stats_sha, manifest_sha, fingerprint(metadata))
    except ValueError:
        historical = None
    views = image_input if image_input != "auto" else found.get("image_input")
    if views is None and historical:
        views = "four" if historical.get("online_roi_required") else "three"
    route = input_route(views, task_mode, historical)
    trained = None
    if is_train:
        if not detected:
            warnings.append("请选择 Absolute 或 Delta 动作类型")
        trained = {"max_delay": delay, "record": None, "manual_confirmation": True,
                   "semantics": detected, "record_sha256": hashlib.sha256(f"{manifest_sha}:{stats_sha}:{delay}:{detected}".encode()).hexdigest()}
    supported_modes = ["serial"]
    if not lora:
        supported_modes.append("train-paper-rtc" if trained and algorithm else "train-rtc" if trained else "paper-rtc" if algorithm else "rtc")
    identity = hashlib.sha256(json.dumps([manifest_sha, stats_sha, algorithm_identity(algorithm), trained, route], sort_keys=True).encode()).hexdigest()
    if views is None:
        warnings.append("无法识别图像路数，请手动选择三路或四路")
    if not detected:
        warnings.append("无法识别动作语义，请手动选择模型类型")
    return {"tomato_model": route, "input_views": {"three": 3, "four": 4}.get(views),
            "train_rtc": trained, "supported_modes": supported_modes,
            "paper_algorithm": algorithm, "normalization": algorithm["mode"] if algorithm else "global_quantile",
            "checkpoint": str(root), "norm_stats": str(stats), "identity": identity,
            "manifest_sha256": manifest_sha, "stats_sha256": stats_sha,
            "finetune_mode": "lora" if lora else "full", "semantics": detected,
            "evidence": "手动配置" if manual else "参数、算法资产及可选配置中的可识别字段",
            "warnings": warnings, "suggestions": found,
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
    prompt: str = DEFAULT_PROMPT
    image_input: str = "auto"
    dagger: bool = False
    dagger_output: str = ""
    dagger_recording_format: str = "mcap"
    dagger_head_depth: bool = False
    dagger_tactile: bool = False

    def plan(self, *, cache=None, machine_root=None):
        if type(self.dagger_head_depth) is not bool or type(self.dagger_tactile) is not bool:
            raise ValueError("深度/触觉录制开关必须为布尔值")
        if type(self.dagger) is not bool or (self.dagger and not self.record):
            raise ValueError('DAgger必须明确开启并同时录制样本与标签')
        if self.dagger_recording_format not in ("mcap", "rosbag"):
            raise ValueError("样本保存方式仅支持mcap或rosbag")
        repo = Path(self.repository).expanduser().resolve()
        if self.dagger and not all((repo / 'deploy/fr3_sharpa_runtime' / name).is_file()
                                   for name in ('dagger_execution.py','dagger_owner.py')):
            raise ValueError('所选部署仓库缺少DAgger执行器，请选择franka-sharpa-dagger分支')
        from deploy.fr3_sharpa_runtime.task_stages import stage
        task_profile = self.task_mode if self.task_mode != "none" else "tomato-b"
        stage(self.initial_stage, task_profile)
        first_stage = "pickup" if task_profile == "tomato-in-hand119" else "pickup_1"
        if self.task_mode not in ("none", "tomato-b", "tomato-in-hand119") or type(self.recovery_confirmed) is not bool:
            raise ValueError("任务模式或人工恢复确认无效")
        if not isinstance(self.prompt, str) or (self.task_mode == "none" and not self.prompt.strip()):
            raise ValueError("统一 Prompt 部署需要填写训练时使用的英文指令")
        if self.initial_stage != first_stage and (self.task_mode == "none" or not self.recovery_confirmed):
            raise ValueError("非首次阶段启动需要明确人工恢复确认并重新核对配置")
        if self.camera_policy not in ("exclusive", "require_free"):
            raise ValueError("相机策略仅支持独占接管或要求设备空闲")
        if self.mode not in MODES or self.initialization not in ("home", "current") or self.speed not in ("slow", "medium", "fast"):
            raise ValueError("不支持的模式、启动姿态或速度")
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
                                   image_input=self.image_input, task_mode=self.task_mode)
        if self.crop_device not in ("cuda", "cpu"):
            raise ValueError("局部图设备仅支持cuda/cpu")
        if model.get("tomato_model") and model["tomato_model"].get("online_roi_required"):
            from deploy.fr3_sharpa_runtime.interaction_crop import locator_identity
            locator, locator_sha = locator_identity(self.crop_locator or repo / ".deployment/crop_locator")
            model["crop_locator"] = locator
            model["crop_locator_sha256"] = locator_sha
            model["identity"] = hashlib.sha256((model["identity"] + locator_sha + self.crop_device).encode()).hexdigest()

        if not model["profile"]:
            raise ValueError(model["evidence"])
        if model["input_views"] is None:
            raise ValueError("请选择三路或四路图像输入后核对配置")
        if self.task_mode != "none":
            if self.mode == "serial" or model["profile"] != "full-delta" or self.rounds != 0:
                raise ValueError("分阶段任务要求 Full Delta、RTC 类执行模式和连续轮数0")
            if not (repo / "deploy/fr3_sharpa_runtime/task_execution.py").is_file():
                raise ValueError("所选仓库缺少番茄任务执行接口")
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
        else:
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
                profile.port_env: str(plan["port"]),
                "SHARPA_MODEL_PROFILE": profile.name,
                "SHARPA_DEPLOY_MODEL_TYPE": plan["settings"].get("model_type", "auto"),
                "SHARPA_DEPLOY_IMAGE_INPUT": "four" if plan["model"]["input_views"] == 4 else "three",
                "SHARPA_DEPLOY_TASK_MODE": plan["settings"].get("task_mode", "none")})
    root = plan["settings"]["repository"]
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
    for key in ('SHARPA_DAGGER_SOCKET', 'SHARPA_DAGGER_RUN_ID', 'SHARPA_DAGGER_DIRECTORY'):
        env.pop(key, None)
    env['SHARPA_RECORD_HEAD_DEPTH'] = '1' if plan['settings'].get('dagger') and plan['settings'].get('dagger_head_depth', False) else '0'
    env['SHARPA_RECORD_TACTILE'] = '1' if plan['settings'].get('dagger') and plan['settings'].get('dagger_tactile', False) else '0'
    env['SHARPA_DAGGER_ENABLED'] = '1' if plan['settings'].get('dagger') else '0'
    env['SHARPA_RECORD_FORMAT'] = (plan['settings'].get('dagger_recording_format', 'rosbag')
                                  if plan['settings'].get('dagger') else 'rosbag')
    return env
