import argparse
from itertools import pairwise
import json
from pathlib import Path
import statistics
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

p = argparse.ArgumentParser()
p.add_argument("--duration", type=float, default=60)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
rclpy.init()
n = rclpy.create_node("camera_source_clock_probe")
start = time.monotonic()
warmup = 10
stats = {
    k: {"received": [], "age": [], "prev_stamp": None, "backwards": 0, "bad_examples": []}
    for k in ("cam0", "cam1", "cam2")
}
subs = []


def receive(k, m):
    now = time.monotonic()
    if now - start < warmup:
        return
    stamp = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
    s = stats[k]
    age = (n.get_clock().now().nanoseconds - stamp) / 1e6
    if s["prev_stamp"] is not None and stamp < s["prev_stamp"]:
        s["backwards"] += 1
    s["prev_stamp"] = stamp
    s["received"].append(now)
    s["age"].append(age)
    if not 0 <= age <= 200 and len(s["bad_examples"]) < 3:
        s["bad_examples"].append({"elapsed_s": now - start, "age_ms": age})


subs = [
    n.create_subscription(Image, f"/{k}/color/image_raw", lambda m, k=k: receive(k, m), qos_profile_sensor_data)
    for k in stats
]
next_print = warmup + 15
try:
    while time.monotonic() - start < warmup + a.duration:
        rclpy.spin_once(n, timeout_sec=0.03)
        if time.monotonic() - start >= next_print:
            print(
                json.dumps(
                    {
                        "elapsed_s": round(time.monotonic() - start, 1),
                        "streams": {
                            k: {
                                "frames": len(s["age"]),
                                "backwards": s["backwards"],
                                "bad_age": sum(not 0 <= x <= 200 for x in s["age"]),
                            }
                            for k, s in stats.items()
                        },
                    }
                ),
                flush=True,
            )
            next_print += 15
    result = {}
    for k, s in stats.items():
        ts = s["received"]
        ages = s["age"]
        gaps = [b - c for c, b in pairwise(ts)]
        result[k] = {
            "frames": len(ts),
            "fps": (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 1 else None,
            "source_age_ms": {"min": min(ages), "median": statistics.median(ages), "max": max(ages)} if ages else None,
            "source_backwards": s["backwards"],
            "bad_age_frames": sum(not 0 <= x <= 200 for x in ages),
            "max_receive_gap_ms": max(gaps) * 1000 if gaps else None,
            "bad_examples": s["bad_examples"],
            "publishers": [i.node_name for i in n.get_publishers_info_by_topic(f"/{k}/color/image_raw")],
        }
    a.output.write_text(json.dumps({"duration_s": a.duration, "warmup_s": warmup, "streams": result}, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
finally:
    n.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
