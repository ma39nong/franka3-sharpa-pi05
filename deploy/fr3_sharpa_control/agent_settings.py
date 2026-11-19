"""Portable per-machine Agent settings. Development never requires ROS or sudo."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import subprocess
import json

TAIRAN_PROVIDER = "tairan"
AGENT_PROVIDERS = ("mengxiang", TAIRAN_PROVIDER)
IN_HAND_NORM = Path(".deployment/checkpoints/tomato-in-hand119/20000/assets") / (
    "tomato_in_hand119_rgb_crop_v3_r1_delta_30hz/norm_stats.json"
)


@dataclass
class AgentSettings:
    provider: str = "mengxiang"
    profile: str = "development"
    repository: str = ""
    checkpoint: str = "runs/tomato/optimization_20261108/bundles/independent_h15_s42.pt"
    python: str = ".venv-tomato/bin/python"
    pretrained: str = "pretrained/vjepa2-vitl-fpc64-256"
    sample_npz: str = "runs/tomato/optimization_20261108/validation_observation.npz"
    gello_repository: str = ""
    ros_overlay: str = ".deployment/sharpa_ros_messages/install"

    @classmethod
    def defaults(cls):
        return cls(
            repository=os.environ.get("SHARPA_AGENT_REPOSITORY", str(Path.home() / "vjepa2_training")),
            gello_repository=os.environ.get("SHARPA_GELLO_REPO", str(Path.home() / "gello_upper_body_teleop")),
        )

    @classmethod
    def parse(cls, values, deployment_root):
        defaults = asdict(cls.defaults())
        values = dict(values)
        # Keep existing workstation settings readable across the provider rename.
        handshake = values.pop("handshake_config", "")
        if values.get("provider") == "inhand119_four":
            values["provider"] = TAIRAN_PROVIDER
        elif values.get("provider") == TAIRAN_PROVIDER and isinstance(handshake, str) and handshake.endswith(".json"):
            # The old three-camera bundle used a JSON handshake and different model paths.
            for name in ("repository", "checkpoint", "pretrained", "python", "sample_npz", "profile"):
                values.pop(name, None)
        if values.get("provider") == TAIRAN_PROVIDER:
            defaults.update(
                profile="workstation",
                repository=str(Path(deployment_root) / ".deployment/agents/inhand119_four"),
                checkpoint="head/best_head.pt",
                python=str(Path(deployment_root) / ".venv/bin/python"),
                pretrained="encoder",
                sample_npz="",
            )
        unknown = set(values) - set(defaults)
        if unknown:
            raise ValueError("未知 Agent 配置：" + ", ".join(sorted(unknown)))
        defaults.update(values)
        if any(not isinstance(v, str) for v in defaults.values()):
            raise ValueError("Agent 配置必须为文本")
        obj = cls(**defaults)
        if obj.provider not in AGENT_PROVIDERS:
            raise ValueError("请选择已支持的 Agent 模型")
        if obj.profile not in ("development", "workstation"):
            raise ValueError("请选择开发机自检或工作机实时模式")
        for name in ("repository", "gello_repository"):
            setattr(obj, name, str(Path(getattr(obj, name)).expanduser().resolve()))
        for name in ("checkpoint", "python", "pretrained", "sample_npz", "ros_overlay"):
            value = Path(getattr(obj, name)).expanduser()
            parent = Path(deployment_root) if name == "ros_overlay" else Path(obj.repository)
            absolute = value if value.is_absolute() else parent / value
            # Python locates pyvenv.cfg using its invocation path. Resolving the
            # interpreter symlink would silently escape the selected venv.
            setattr(obj, name, os.path.abspath(absolute) if name == "python" else str(absolute.resolve()))
        return obj

    def check(self, deployment_root=None):
        checks = []

        def add(name, ok, detail):
            checks.append(dict(name=name, ok=bool(ok), detail=str(detail)))

        entry = {
            "mengxiang": "tools/serve_tomato_agent.py",
            TAIRAN_PROVIDER: "tools/tomato_phase_model.py",
        }[self.provider]
        for name, path in [
            ("Agent仓库", Path(self.repository) / entry),
            ("Python环境", Path(self.python)),
            ("Agent权重", Path(self.checkpoint)),
            ("编码器权重", Path(self.pretrained) / "model.safetensors"),
            ("编码器配置", Path(self.pretrained) / "config.json"),
        ]:
            add(name, path.is_file(), path)
        if self.provider == TAIRAN_PROVIDER:
            manifest = Path(self.checkpoint).with_name("manifest.json")
            add("119 条训练清单", manifest.is_file(), manifest)
            root = Path(deployment_root) if deployment_root else Path(__file__).resolve().parents[2]
            norm = root / IN_HAND_NORM
            locator = root / ".deployment/crop_locator"
            add("119 条 ROI 归一化", norm.is_file(), norm)
            add("119 条 ROI 定位器", all((locator / name).is_file() for name in
                                       ("model.pt", "primary/model.pt", "receipt.json")), locator)
        add("Python可执行", os.access(self.python, os.X_OK), self.python)
        if os.access(self.python, os.X_OK):
            try:
                env = dict(os.environ)
                env.pop("PYTHONPATH", None)
                result = subprocess.run(
                    [
                        self.python,
                        "-c",
                        "import json,numpy,torch,transformers; print(json.dumps(dict(torch=str(torch.__version__),cuda=torch.cuda.is_available(),bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported())))",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                info = json.loads(result.stdout) if result.returncode == 0 else {}
                add(
                    "模型依赖与CUDA BF16",
                    info.get("cuda") and info.get("bf16"),
                    info if info else result.stderr[-1200:],
                )
            except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                add("模型依赖与CUDA BF16", False, error)
        if self.profile == "development" and self.provider == TAIRAN_PROVIDER:
            script = Path(self.repository) / "offline_check.py"
            add("119 条实时模式", False, f"请选工作机实时模式；离线回放入口为 {script}")
            note = "四路观察的实时输入来自工作机。"
        elif self.profile == "development":
            add("离线观测样本", Path(self.sample_npz).is_file(), self.sample_npz)
            note = "开发机仅加载模型并读取离线NPZ自检；不启动ROS、不检查DDS、不连接任务执行端。回放请求使用测试时间，不能当作真实采集延迟。"
        else:
            gello = Path(self.gello_repository)
            for name, path in [
                ("ROS项目", gello / "portable_deps/litchi_hardware/pixi.toml"),
                ("ROS消息overlay", Path(self.ros_overlay) / "setup.bash"),
                ("DDS配置", gello / "config/cyclonedds.xml"),
                ("ROS domain配置", gello / "docker/.env"),
            ]:
                add(name, path.is_file(), path)
            add("Pixi", shutil.which("pixi") is not None, shutil.which("pixi") or "未找到pixi")
            try:
                maximum = int(Path("/proc/sys/net/core/rmem_max").read_text())
            except (OSError, ValueError):
                maximum = 0
            command = "sudo bash " + str(gello / "ops/setup/install_dds_buffers.sh")
            add(
                "工作机DDS接收缓冲",
                maximum >= 8388608,
                f"当前 {maximum}；要求8388608。仅在连接机器人的工作机执行：{command}",
            )
            note = "工作机实时模式：Agent与部署控制台在同一工作机运行。检查通过后仍需真实三路相机和关节流；本页不启动硬件。跨机器Agent通信尚未实现。"
        return dict(
            provider=self.provider,
            profile=self.profile,
            ready=all(c["ok"] for c in checks),
            checks=checks,
            note=note,
            settings=asdict(self),
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--deployment-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    values = json.loads(args.config.read_text()) if args.config else {}
    report = AgentSettings.parse(values, args.deployment_root).check(args.deployment_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
