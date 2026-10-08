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
