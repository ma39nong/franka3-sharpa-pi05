"""Read-only ROS camera gate: actual serial identity, unique publishers and fresh RGB."""
# ruff: noqa: RUF001 -- Chinese operator diagnostics use Chinese punctuation.

import argparse
from collections import deque
import json
import os
import sys
import time

from .camera_alignment_recovery import DEFAULT_RECOVERY_SEC
from .camera_alignment_recovery import AlignmentRecovery
from .camera_startup import camera_contract
from .ros_observation import TOPICS
from .ros_observation import ObservationProvider
from .ros_observation import select_aligned_observation
from .ros_observation import stamp_ns


def check_serials(expected, observed):
    for name, entry in expected.items():
        if observed.get(name) != entry["serial_number"]:
            raise ValueError(f"{name} 实际序列号{observed.get(name)!r}与配置{entry['serial_number']!r}不符；禁止Home")


def probe(path, timeout=45, *, watch=False, alignment_recovery_sec=DEFAULT_RECOVERY_SEC):
    recovery = AlignmentRecovery(alignment_recovery_sec)
    from orbbec_camera_msgs.srv import GetDeviceInfo
    import rclpy
    from rclpy.qos import DurabilityPolicy
    from rclpy.qos import HistoryPolicy
    from rclpy.qos import QoSProfile
    from rclpy.qos import ReliabilityPolicy
    from sensor_msgs.msg import Image

    expected = camera_contract(path)
    rclpy.init()
    node = rclpy.create_node("sharpa_deployment_camera_gate")
    # Reuse the exact runtime freshness/history/stability rules, but require cameras only.
    provider = ObservationProvider.__new__(ObservationProvider)
    provider.latest = {}
    provider.histories = {key: deque(maxlen=16) for key in expected}
    provider.camera_windows = {}
    provider.startup_stable_s = 3.0
    provider.startup_ready = False
    issues, serials = {}, {}
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )

    def remember(key, message):
        if (
            message.encoding not in ("rgb8", "bgr8")
            or message.height <= 0
            or message.width <= 0
            or message.step < message.width * 3
            or len(message.data) != message.height * message.step
        ):
            issues[key] = "图像不是有效RGB/BGR帧"
            return
        issues.pop(key, None)
        provider.remember(key, message)

    subscriptions = [
        node.create_subscription(Image, TOPICS[key], lambda msg, key=key: remember(key, msg), qos) for key in expected
    ]
    deadline = time.monotonic() + timeout
    detail = "等待相机驱动与图像"
    ready = False
    last_report = 0.0
    publisher_ids = None
    last_warning = None
    try:
        while (watch and ready) or time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            services = node.get_service_names_and_types()
            for key in expected:
                matches = [
                    name
                    for name, types in services
                    if name.startswith(f"/{key}/")
                    and name.endswith("/get_device_info")
                    and "orbbec_camera_msgs/srv/GetDeviceInfo" in types
                ]
                if len(matches) > 1:
                    raise RuntimeError(f"{key}存在多个设备身份服务：{matches}；可能仍有相机占用")
                if key in serials or len(matches) != 1:
                    continue
                client = node.create_client(GetDeviceInfo, matches[0])
                try:
                    future = client.call_async(GetDeviceInfo.Request())
                    rclpy.spin_until_future_complete(
                        node, future, timeout_sec=min(2, max(0.01, deadline - time.monotonic()))
                    )
                    if future.done() and future.exception() is None and future.result().success:
                        serials[key] = future.result().info.serial_number
                finally:
                    node.destroy_client(client)
            if len(serials) != 3:
                detail = f"等待实际设备序列号，应答缺失：{sorted(set(expected) - serials.keys())}"
                continue
            check_serials(expected, serials)
            if issues or len(provider.latest) != 3 or not provider.cameras_ready(time.monotonic_ns()):
                if ready:
                    raise RuntimeError("运行中相机图像无效；停止动作，禁止自动重启驱动")
                detail = f"等待三路连续3秒出图（cam0≥18Hz，双腕≥25Hz，间隔≤150ms）：{issues}"
                continue
            publishers = {key: node.get_publishers_info_by_topic(TOPICS[key]) for key in expected}
            if any(len(owners) != 1 for owners in publishers.values()):
                if ready:
                    raise RuntimeError("运行中相机发布者缺失或重复；停止动作，禁止抢占")
                detail = "三路相机必须各有唯一发布者：" + str({key: len(owners) for key, owners in publishers.items()})
                continue
            if any(not owners[0].node_namespace.startswith(f"/{key}") for key, owners in publishers.items()):
                raise RuntimeError("相机发布者命名空间与身份服务不匹配")
            current_ids = {key: tuple(owners[0].endpoint_gid) for key, owners in publishers.items()}
            if publisher_ids is not None and publisher_ids != current_ids:
                raise RuntimeError("运行中相机发布者身份发生变化；停止动作，禁止自动接管")
            publisher_ids = current_ids
            try:
                selected, alignment = select_aligned_observation(
                    provider.latest,
                    provider.histories,
                    now_ros=node.get_clock().now().nanoseconds,
                    now_mono=time.monotonic_ns(),
                )
            except ValueError as error:
                if ready:
                    try:
                        diagnostic = recovery.wait(
                            error,
                            provider.latest,
                            now_ros=node.get_clock().now().nanoseconds,
                            now_mono=time.monotonic_ns(),
                        )
                    except RuntimeError:
                        print(
                            json.dumps({"event": "camera_alignment_failed", "reason": str(error)}, ensure_ascii=False),
                            file=sys.stderr,
                            flush=True,
                        )
                        raise
                    now = time.monotonic()
                    if last_warning is None or now - last_warning >= 0.1:
                        diagnostic["history"] = {
                            key: {
                                "count": len(history),
                                "oldest_stamp_ns": min((stamp_ns(e[0]) for e in history), default=None),
                                "newest_stamp_ns": max((stamp_ns(e[0]) for e in history), default=None),
                            }
                            for key, history in provider.histories.items()
                        }
                        print(json.dumps(diagnostic, ensure_ascii=False), file=sys.stderr, flush=True)
                        last_warning = now
                detail = str(error)
                continue
            recovered = recovery.recovered(time.monotonic_ns())
            if recovered is not None:
                print(json.dumps(recovered), file=sys.stderr, flush=True)
                last_warning = None
            report = {
                "status": "PASS",
                "checked_monotonic_ns": time.monotonic_ns(),
                "serials": serials,
                "contract": expected,
                "stable_seconds": 3,
                "alignment": alignment,
                "source_stamps_ns": {key: stamp_ns(value[0]) for key, value in selected.items()},
                "command_publishers_created": 0,
                "alignment_recovery_sec": recovery.seconds,
            }
            if not watch:
                return report
            if not ready or time.monotonic() - last_report >= 0.2:
                print(json.dumps(report, ensure_ascii=False), flush=True)
                ready = True
                last_report = time.monotonic()
        raise TimeoutError("相机启动验证超时：" + detail + "；未允许连接执行端或Home")
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument(
        "--watch", action="store_true", help="Monitor continuously; bounded retry for fresh cross-camera misalignment"
    )
    parser.add_argument(
        "--alignment-recovery-sec",
        type=float,
        default=os.environ.get("SHARPA_CAMERA_ALIGNMENT_RECOVERY_SEC", str(DEFAULT_RECOVERY_SEC)),
        help="Fresh alignment retry window in [0, 0.5] seconds (default 0.3); 0 fails immediately",
    )
    parser.add_argument("--check", action="store_true", help="Import message types only; no ROS node or devices")
    args = parser.parse_args()
    try:
        AlignmentRecovery(args.alignment_recovery_sec)
    except ValueError as error:
        parser.error(str(error))
    if args.check:
        from orbbec_camera_msgs.srv import GetDeviceInfo
        from sensor_msgs.msg import Image

        GetDeviceInfo.Request()
        Image()
        camera_contract(args.config)
        print(json.dumps({"status": "PASS", "scope": "camera probe dependencies; no ROS node"}))
        return
    try:
        result = probe(args.config, watch=args.watch, alignment_recovery_sec=args.alignment_recovery_sec)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
