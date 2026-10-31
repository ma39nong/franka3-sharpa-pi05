"""Startup-only camera ownership transfer; never kill unknown processes or reset devices."""
# ruff: noqa: RUF001 -- Chinese operator diagnostics use Chinese punctuation.

import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml

from .camera_lease import CameraLease
from .hardware_session import HardwareSession

ROLES = {"cam0": "cam_high", "cam1": "cam_right_wrist", "cam2": "cam_left_wrist"}
KNOWN = {"teleop-cameras", "openpi-cameras"}


def camera_contract(path):
    data = yaml.safe_load(Path(path).read_text())["camera_bringup"]
    result = {}
    for name, role in ROLES.items():
        entry = data[name]
        serial = str(entry.get("serial_number", "")).strip()
        if not serial or serial.startswith("REPLACE_") or entry.get("semantic") != role:
            raise ValueError(f"{name} 相机序列号或左右语义未正确配置（应为{role}）")
        result[name] = {"serial_number": serial, "semantic": role, "net_device_ip": str(entry.get("net_device_ip", ""))}
    if len({v["serial_number"] for v in result.values()}) != 3:
        raise ValueError("三路相机序列号必须唯一")
    return result


def mapped_path(item, path):
    matches = []
    for mount in item.get("Mounts", []):
        destination = Path(mount["Destination"])
        try:
            relative = Path(path).relative_to(destination)
        except ValueError:
            continue
        if mount.get("Type") == "bind":
            matches.append((len(destination.parts), Path(mount["Source"]) / relative))
    if not matches:
        raise ValueError("相机配置没有可核对的主机只读来源")
    return max(matches, key=lambda pair: pair[0])[1]


def classify(item, expected):
    command = item["Config"].get("Cmd") or []
    service = (item["Config"].get("Labels") or {}).get("com.docker.compose.service", "")
    text = " ".join(command)
    if service not in KNOWN and not any(word in text.lower() for word in ("camera", "orbbec", "obsensor")):
        return "unrelated"
    arguments = [part.split(":=", 1)[1] for part in command if part.startswith("deployment_config:=")]
    observed = None
    if len(arguments) == 1:
        with suppress(OSError, ValueError, KeyError, yaml.YAMLError):
            observed = camera_contract(mapped_path(item, arguments[0]))
    if observed:

        def serials(contract):
            return {v["serial_number"] for v in contract.values()}

        if not serials(observed) & serials(expected):
            return "unrelated"
    if service in KNOWN and "triple_camera.launch.py" in command and observed == expected:
        return "known"
    raise RuntimeError(
        f"未知或无法核对的相机占用：{item.get('Name', item['Id'])} service={service!r} command={text!r}；不会自动停止"
    )


class CameraStartup:
    def __init__(self, hardware=None):
        self.hardware = hardware or HardwareSession()
        self.config = Path(os.environ["SHARPA_GELLO_REPO"]) / "data_collection/config/cameras.yaml"
        self.expected = camera_contract(self.config)
        self.started = None
        self.stopped = []

    @staticmethod
    def command(arguments, **kwargs):
        result = subprocess.run(
            arguments, check=False, capture_output=True, text=True, timeout=kwargs.pop("timeout", 45), **kwargs
        )
        if result.returncode:
            raise RuntimeError(f"命令失败（退出码{result.returncode}）：{' '.join(arguments)}\n{result.stderr[-8000:]}")
        return result.stdout

    def containers(self):
        ids = self.command(["docker", "ps", "-q"]).split()
        return json.loads(self.command(["docker", "inspect", *ids])) if ids else []

    def process_owners(self, excluded=()):
        """Best-effort host USB/network/driver diagnostics, never signal a PID."""
        excluded = set(excluded) | {os.getpid()}
        devices = set()
        serials = {entry["serial_number"] for entry in self.expected.values()}
        for device in Path("/sys/bus/usb/devices").glob("*"):
            try:
                if (device / "serial").read_text().strip() in serials:
                    bus, number = int((device / "busnum").read_text()), int((device / "devnum").read_text())
                    devices.add(f"/dev/bus/usb/{bus:03}/{number:03}")
            except OSError:
                continue
        owners = []
        for process in Path("/proc").glob("[0-9]*"):
            pid = int(process.name)
            if pid in excluded:
                continue
            try:
                words = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
                text = " ".join(words)
                active = any(
                    word in words
                    for word in (
                        "deploy.fr3_sharpa_runtime.sdk_direct_motion_ros",
                        "deploy.fr3_sharpa_runtime.async_soft_motion_ros",
                    )
                )
                driver = any(
                    Path(word).name in ("OrbbecViewer", "camera_node", "ob_camera_node", "ob_camera_node_driver")
                    for word in words
                )
                held = []
                for fd in (process / "fd").glob("*"):
                    try:
                        target = os.readlink(fd)
                        if target in devices:
                            held.append(target)
                    except OSError:
                        continue
                if active or driver or held:
                    owners.append({"pid": pid, "command": text[:500], "usb_handles": held, "active_deployment": active})
            except (OSError, ValueError):
                continue
        # Network camera users may have no USB handles.
        result = subprocess.run(["ss", "-H", "-uapn"], capture_output=True, text=True, timeout=5, check=False)
        ips = {entry["net_device_ip"] for entry in self.expected.values()} - {""}
        for line in result.stdout.splitlines():
            if any(ip + ":" in line for ip in ips):
                for pid in map(int, re.findall(r"pid=(\d+)", line)):
                    if pid not in excluded and not any(owner["pid"] == pid for owner in owners):
                        owners.append({"pid": pid, "network_socket": line.strip()})
        return owners

    def container_pids(self, items):
        result = set()
        for item in items:
            rows = self.command(["docker", "top", item["Id"], "-eo", "pid"]).splitlines()[1:]
            result.update(int(row.strip()) for row in rows if row.strip().isdigit())
        return result

    def inspect(self):
        items = self.containers()
        known = [item for item in items if classify(item, self.expected) == "known"]
        owners = self.process_owners(self.container_pids(known))
        if owners:
            raise RuntimeError("相机存在未知占用或活动执行端；不会杀进程：" + json.dumps(owners, ensure_ascii=False))
        return known

    def stop(self, item):
        current = json.loads(self.command(["docker", "inspect", item["Id"]]))[0]
        if classify(current, self.expected) != "known":
            raise RuntimeError("停止前相机身份发生变化；拒绝接管")
        self.command(["docker", "stop", "--time", "30", item["Id"]], timeout=40)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = json.loads(self.command(["docker", "inspect", item["Id"]]))[0]["State"]
            if not state["Running"] and not state.get("Pid"):
                self.stopped.append(item.get("Name", item["Id"]))
                return
            time.sleep(0.1)
        raise TimeoutError(f"相机驱动未退出：{item['Id']}；停止启动流程")

    def release(self):
        deadline, quiet = time.monotonic() + 5, None
        owners = []
        while time.monotonic() < deadline:
            owners = self.process_owners()
            if not owners:
                quiet = time.monotonic() if quiet is None else quiet
                if time.monotonic() - quiet >= 0.5:
                    return
            else:
                quiet = None
            time.sleep(0.1)
        raise RuntimeError("等待相机设备释放超时；占用信息：" + json.dumps(owners, ensure_ascii=False))

    def probe(self):
        script = Path(__file__).with_name("camera_probe.sh")
        output = self.command(["bash", str(script), str(self.config)], timeout=55)
        report = json.loads(output.splitlines()[-1])
        if report.get("status") != "PASS":
            raise RuntimeError("三路相机启动验证失败：" + output)
        return report

    def up(self, *, exclusive=False):
        self.command(
            ["bash", str(Path(__file__).with_name("camera_probe.sh")), str(self.config), "--check"], timeout=15
        )
        known = self.inspect()  # Fully validate all candidates BEFORE stopping any.
        if known and not exclusive:
            raise RuntimeError(
                "相机已有已知服务占用；选择部署独占接管后再启动："
                + ", ".join(item.get("Name", item["Id"]) for item in known)
            )
        try:
            for item in known:
                print(f"[camera-startup] 停止同组已知服务 {item.get('Name', item['Id'])}", file=sys.stderr, flush=True)
                self.stop(item)
            print("[camera-startup] 等待驱动退出与设备释放", file=sys.stderr, flush=True)
            self.release()
            # Catch a replacement camera launcher racing with release.
            if self.inspect():
                raise RuntimeError("相机释放期间出现新的驱动；拒绝重复启动")
            self.command([*self.hardware.compose, "up", "-d", "--no-deps", "openpi-cameras"])
            items = [
                item
                for item in self.containers()
                if (item["Config"].get("Labels") or {}).get("com.docker.compose.service") == "openpi-cameras"
                and self.hardware.is_openpi(item, "openpi-cameras")
            ]
            if len(items) != 1:
                raise RuntimeError("未发现唯一的本部署相机容器")
            self.started = items[0]
            print("[camera-startup] 核对实际序列号、三路RGB、连续频率及时间戳", file=sys.stderr, flush=True)
            readiness = self.probe()
            # Only start arm/Gateway dependencies AFTER cameras are verified.
            self.command([*self.hardware.compose, "up", "-d", "franka-control", "teleop-control"])
            return {
                "status": "PASS",
                "policy": "exclusive" if exclusive else "require_free",
                "stopped": self.stopped,
                "camera_container": self.started["Id"],
                "readiness": readiness,
            }
        except BaseException as error:
            error.add_note("本次已停止的已知相机服务：" + json.dumps(self.stopped, ensure_ascii=False))
            if self.started is not None:
                # Immutable ID; never stop a replacement service or restart the old owner.
                try:
                    self.stop(self.started)
                except Exception as cleanup_error:
                    error.add_note(f"本次相机清理失败：{cleanup_error}")
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inspect", action="store_true", help="Read-only ownership audit; do not stop/start containers"
    )
    parser.add_argument(
        "--exclusive", action="store_true", help="Stop verified known same-camera services before startup"
    )
    args = parser.parse_args()
    with CameraLease():
        startup = CameraStartup()
        result = (
            {
                "status": "INSPECTED",
                "cameras": startup.expected,
                "known_owners": [item.get("Name", item["Id"]) for item in startup.inspect()],
            }
            if args.inspect
            else startup.up(exclusive=args.exclusive)
        )
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
