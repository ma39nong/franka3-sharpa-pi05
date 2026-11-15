"""Portable per-machine Agent settings. Development never requires ROS or sudo."""

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import subprocess
import json


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
    handshake_config: str = ""
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
        if values.get("provider") == "tairan":
            base = Path(deployment_root) / ".deployment/agents/tairan"
            defaults.update(
                repository=str(base / "source"),
                checkpoint=str(base / "artifacts/joint/best_head.pt"),
                python=str(Path.home() / "vjepa2_training/.venv-tomato/bin/python"),
                pretrained=str(base / "encoder"),
                sample_npz=str(
                    Path.home() / "vjepa2_training/runs/tomato/optimization_20261108/validation_observation.npz"
                ),
                handshake_config="configs/tairan_handshake.json",
            )
        unknown = set(values) - set(defaults)
        if unknown:
            raise ValueError("未知 Agent 配置：" + ", ".join(sorted(unknown)))
        defaults.update(values)
        if any(not isinstance(v, str) for v in defaults.values()):
            raise ValueError("Agent 配置必须为文本")
        obj = cls(**defaults)
        if obj.provider not in ("mengxiang", "tairan"):
            raise ValueError("请选择 mengxiang 或 tairan")
        if obj.profile not in ("development", "workstation"):
            raise ValueError("请选择开发机自检或工作机实时模式")
        for name in ("repository", "gello_repository"):
            setattr(obj, name, str(Path(getattr(obj, name)).expanduser().resolve()))
        for name in ("checkpoint", "python", "pretrained", "sample_npz", "ros_overlay", "handshake_config"):
            value = Path(getattr(obj, name)).expanduser()
            parent = Path(deployment_root) if name == "ros_overlay" else Path(obj.repository)
            absolute = value if value.is_absolute() else parent / value
            # Python locates pyvenv.cfg using its invocation path. Resolving the
            # interpreter symlink would silently escape the selected venv.
            setattr(obj, name, os.path.abspath(absolute) if name == "python" else str(absolute.resolve()))
        return obj

    def check(self):
        checks = []

        def add(name, ok, detail):
            checks.append(dict(name=name, ok=bool(ok), detail=str(detail)))

        for name, path in [
            (
                "Agent仓库",
                Path(self.repository)
                / (
                    "tools/serve_tomato_agent.py"
                    if self.provider == "mengxiang"
                    else "tools/verify_tairan_deployment.py"
                ),
            ),
            ("Python环境", Path(self.python)),
            ("Agent权重", Path(self.checkpoint)),
            ("编码器权重", Path(self.pretrained) / "model.safetensors"),
            ("编码器配置", Path(self.pretrained) / "config.json"),
        ]:
            add(name, path.is_file(), path)
        if self.provider == "tairan":
            add("tairan握手配置", Path(self.handshake_config).is_file(), self.handshake_config)
            for name in (
                "manifest.json",
                "split.json",
                "training_config.json",
                "validation_lock.json",
                "validation_receipt.json",
            ):
                add(
                    "tairan " + name,
                    (Path(self.checkpoint).parent / name).is_file(),
                    Path(self.checkpoint).parent / name,
                )
            if all(c["ok"] for c in checks) and os.access(self.python, os.X_OK):
                try:
                    env = dict(os.environ)
                    env.pop("PYTHONPATH", None)
                    verified = subprocess.run(
                        [
                            self.python,
                            "-m",
                            "tools.verify_tairan_deployment",
                            "--run-dir",
                            str(Path(self.checkpoint).parent),
                            "--encoder-dir",
                            self.pretrained,
                        ],
                        cwd=self.repository,
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    add(
                        "tairan九文件与来源校验",
                        verified.returncode == 0,
                        "PASS：正式权重、归一化、源码与完整配置匹配"
                        if verified.returncode == 0
                        else verified.stderr[-1600:],
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    add("tairan九文件与来源校验", False, error)
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
        if self.profile == "development":
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
    report = AgentSettings.parse(values, args.deployment_root).check()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
