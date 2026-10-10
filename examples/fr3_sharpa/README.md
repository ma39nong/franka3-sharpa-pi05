# FR3 + Sharpa 离线数据转换

本目录只读 ROS 2 bag，生成中间的 `episode.npz`、JPEG 图像和完整报告；不连接
ROS graph，不创建 publisher，也不向机器人发送命令。运行需要 Sharpa 工程的
ROS 2 pixi 环境（提供 `rosbag2_py` 和消息类型）。

```bash
cd /home/user/gello_upper_body_teleop/portable_deps/litchi_hardware
pixi run -e ros2 python /home/user/sharpa-pi05/examples/fr3_sharpa/convert_rosbag.py \
  --bag /home/user/franka_teleop_data/bags/gello_sharpa_tactile/episode0 \
  --output /home/user/franka_teleop_data/converted/fr3_sharpa_episode0
```

The output is intentionally outside Git. `report.json` is the acceptance report;
`episode.npz` contains measured `state`, commanded `action`, and their source times.
Images are stored as `images/{cam0,cam1,cam2}/*.jpg`. A frame is emitted only when
all four state/action streams and all three RGB cameras have a source sample within
the configured tolerance; no missing robot state or action is fabricated.

Export the intermediate into a standard LeRobot v2.1 dataset with the FPS recorded
in `report.json` (the exporter cross-checks metadata, parquet timestamps, MP4 frame
rates, and representative LeRobotDataset samples):

```bash
uv run python /home/user/sharpa-pi05/examples/fr3_sharpa/export_lerobot.py \
  --source /home/user/franka_teleop_data/converted/fr3_sharpa_episode0 \
  --repo-id fr3_sharpa/episode0 \
  --output /home/user/franka_teleop_data/converted/fr3_sharpa_episode0/lerobot
```

Batch conversion writes one manifest and can optionally export all accepted
episodes into one multi-episode LeRobot dataset. The ROS conversion stage runs in
the ROS 2 environment; the optional exporter additionally needs the repository's
LeRobot/uv environment:

```bash
PYTHONPATH=/home/user/sharpa-pi05/src pixi run -e ros2 python batch_convert.py \
  --input-root /home/user/franka_teleop_data/bags/gello_sharpa_tactile \
  --output-root /home/user/franka_teleop_data/converted/batch_30hz_causal \
  --hz 30 --causal
```

Crop intervals are supplied as relative seconds in a JSON file, for example
`{"episode284": [[12.0, 45.0], [50.0, 80.0]]}`. Each interval becomes a separate
LeRobot episode, so action chunks cannot cross a crop boundary. Static frames are
never removed automatically; the source bag and full intermediate remain intact.

Compare two causal conversions and save the age, repetition, and action-event
report with:

```bash
uv run python compare_causal.py \
  --low /path/to/episode284_20hz_causal \
  --high /path/to/episode284_30hz_causal \
  --output /path/to/causal_comparison.json
```
