# 七阶段在线切换实验报告

数据来源：**synthetic**。生成时间：2026-11-11T02:50:01.868767+00:00。

synthetic 为模拟数据；hardware_recording 仅表示实验清单声明的实机记录，未经本工具认证。未知或混合来源不可作为纯实机结论。

本报告的成功仅指软件命令流接入。抓握稳定性、接触力、采摘成功率：**缺失**，不从阶段按钮、COMPLETED或位置插值推断。

归档数：1；在线/未知范围请求：6；确认接入：6。
全部请求接入比例：1.000；有明确结果的接入比例：1.000（分母6）。

## 延迟统计

仅统计已确认接入的切换；每个指标单独排除缺失值，不能把缺失当0。

| 指标 | n | P50 ms | P95 ms | P99 ms |
|---|---:|---:|---:|---:|
| click_to_registered_ms | 6 | 5.000 | 5.000 | 5.000 |
| registered_to_observation_ms | 6 | 24.000 | 24.000 | 24.000 |
| old_future_wait_ms | 6 | 14.500 | 14.500 | 14.500 |
| new_prompt_inference_ms | 6 | 60.000 | 60.000 | 60.000 |
| inference_to_armed_ms | 6 | 6.000 | 6.000 | 6.000 |
| bridge_start_to_applied_ms | 6 | 200.000 | 200.000 | 200.000 |
| click_to_applied_ms | 6 | 305.000 | 305.000 | 305.000 |
| registered_to_applied_ms | 6 | 300.000 | 300.000 | 300.000 |
| observation_age_at_inference_ms | 6 | 3.000 | 3.000 | 3.000 |
| observation_age_at_result_ms | 6 | 63.000 | 63.000 | 63.000 |
| observation_age_at_armed_ms | 6 | 68.000 | 68.000 | 68.000 |

小样本分位数是经验插值，不构成尾延迟保证。来源混合时请使用JSON中by_source_kind分组。

## 每次切换

| run / request_id / version | 源→目标阶段 | 结果 | Click→Applied ms | 登记→Applied ms | 旧phase | 原因 |
|---|---|---|---:|---:|---:|---|
| synthetic-seven-stage / demo-1 / 1 | pickup_1→harvest_1 | applied | 305.000 | 300.000 | 10.000 | 缺失 |
| synthetic-seven-stage / demo-2 / 2 | harvest_1→return_truss | applied | 305.000 | 300.000 | 15.000 | 缺失 |
| synthetic-seven-stage / demo-3 / 3 | return_truss→adjust | applied | 305.000 | 300.000 | 25.000 | 缺失 |
| synthetic-seven-stage / demo-4 / 4 | adjust→pickup_2 | applied | 305.000 | 300.000 | 20.000 | 缺失 |
| synthetic-seven-stage / demo-5 / 5 | pickup_2→harvest_2 | applied | 305.000 | 300.000 | 15.000 | 缺失 |
| synthetic-seven-stage / demo-6 / 6 | harvest_2→return_stem | applied | 305.000 | 300.000 | 25.000 | 缺失 |

## 详细时间线

绝对monotonic时间只在各自文件段或声明的同一host/boot域内有意义，不跨域拼接。

### demo-1（harvest_1，v1）

| 时间点 | monotonic s | 时钟域 | 距登记 ms | 文件:行 |
|---|---:|---|---:|---|
| registered | 12.000 | synthetic-host/boot-demo | 0.000 | task-events.jsonl:3 |
| boundary | 12.003 | synthetic-host/boot-demo | 3.000 | task-events.jsonl:5 |
| observation_start | 12.004 | synthetic-host/boot-demo | 4.000 | task-events.jsonl:6 |
| observation_complete | 12.024 | synthetic-host/boot-demo | 24.000 | task-events.jsonl:9 |
| inference_start | 12.025 | synthetic-host/boot-demo | 25.000 | task-events.jsonl:10 |
| inference_complete | 12.085 | synthetic-host/boot-demo | 85.000 | task-events.jsonl:11 |
| inference_failure | 缺失 | 缺失 | 缺失 | 缺失 |
| armed | 12.091 | synthetic-host/boot-demo | 91.000 | task-events.jsonl:12 |
| scheduled_bridge_start | 12.100 | synthetic-host/boot-demo | 100.000 | control-synthetic.jsonl:9 |
| worker_applied | 12.300 | synthetic-host/boot-demo | 300.000 | control-synthetic.jsonl:16 |
| parent_ack | 12.305 | synthetic-host/boot-demo | 305.000 | task-events.jsonl:13 |
| ordinary_rtc_resumed | 12.600 | synthetic-host/boot-demo | 600.000 | task-events.jsonl:15 |

缺失指标：无
实际接入时间证据：worker_post_send_field；桥接有效帧：7；旧phase证据：prior_worker_frame。

### demo-2（return_truss，v2）

| 时间点 | monotonic s | 时钟域 | 距登记 ms | 文件:行 |
|---|---:|---|---:|---|
| registered | 14.000 | synthetic-host/boot-demo | 0.000 | task-events.jsonl:16 |
| boundary | 14.003 | synthetic-host/boot-demo | 3.000 | task-events.jsonl:18 |
| observation_start | 14.004 | synthetic-host/boot-demo | 4.000 | task-events.jsonl:19 |
| observation_complete | 14.024 | synthetic-host/boot-demo | 24.000 | task-events.jsonl:22 |
| inference_start | 14.025 | synthetic-host/boot-demo | 25.000 | task-events.jsonl:23 |
| inference_complete | 14.085 | synthetic-host/boot-demo | 85.000 | task-events.jsonl:24 |
| inference_failure | 缺失 | 缺失 | 缺失 | 缺失 |
| armed | 14.091 | synthetic-host/boot-demo | 91.000 | task-events.jsonl:25 |
| scheduled_bridge_start | 14.100 | synthetic-host/boot-demo | 100.000 | control-synthetic.jsonl:26 |
| worker_applied | 14.300 | synthetic-host/boot-demo | 300.000 | control-synthetic.jsonl:33 |
| parent_ack | 14.305 | synthetic-host/boot-demo | 305.000 | task-events.jsonl:26 |
| ordinary_rtc_resumed | 14.600 | synthetic-host/boot-demo | 600.000 | task-events.jsonl:28 |

缺失指标：无
实际接入时间证据：worker_post_send_field；桥接有效帧：7；旧phase证据：prior_worker_frame。

### demo-3（adjust，v3）

| 时间点 | monotonic s | 时钟域 | 距登记 ms | 文件:行 |
|---|---:|---|---:|---|
| registered | 16.000 | synthetic-host/boot-demo | 0.000 | task-events.jsonl:29 |
| boundary | 16.003 | synthetic-host/boot-demo | 3.000 | task-events.jsonl:31 |
| observation_start | 16.004 | synthetic-host/boot-demo | 4.000 | task-events.jsonl:32 |
| observation_complete | 16.024 | synthetic-host/boot-demo | 24.000 | task-events.jsonl:35 |
| inference_start | 16.025 | synthetic-host/boot-demo | 25.000 | task-events.jsonl:36 |
| inference_complete | 16.085 | synthetic-host/boot-demo | 85.000 | task-events.jsonl:37 |
| inference_failure | 缺失 | 缺失 | 缺失 | 缺失 |
| armed | 16.091 | synthetic-host/boot-demo | 91.000 | task-events.jsonl:38 |
| scheduled_bridge_start | 16.100 | synthetic-host/boot-demo | 100.000 | control-synthetic.jsonl:43 |
| worker_applied | 16.300 | synthetic-host/boot-demo | 300.000 | control-synthetic.jsonl:50 |
| parent_ack | 16.305 | synthetic-host/boot-demo | 305.000 | task-events.jsonl:39 |
| ordinary_rtc_resumed | 16.600 | synthetic-host/boot-demo | 600.000 | task-events.jsonl:41 |

缺失指标：无
实际接入时间证据：worker_post_send_field；桥接有效帧：7；旧phase证据：prior_worker_frame。

### demo-4（pickup_2，v4）

| 时间点 | monotonic s | 时钟域 | 距登记 ms | 文件:行 |
|---|---:|---|---:|---|
| registered | 18.000 | synthetic-host/boot-demo | 0.000 | task-events.jsonl:42 |
| boundary | 18.003 | synthetic-host/boot-demo | 3.000 | task-events.jsonl:44 |
| observation_start | 18.004 | synthetic-host/boot-demo | 4.000 | task-events.jsonl:45 |
| observation_complete | 18.024 | synthetic-host/boot-demo | 24.000 | task-events.jsonl:48 |
| inference_start | 18.025 | synthetic-host/boot-demo | 25.000 | task-events.jsonl:49 |
| inference_complete | 18.085 | synthetic-host/boot-demo | 85.000 | task-events.jsonl:50 |
| inference_failure | 缺失 | 缺失 | 缺失 | 缺失 |
| armed | 18.091 | synthetic-host/boot-demo | 91.000 | task-events.jsonl:51 |
| scheduled_bridge_start | 18.100 | synthetic-host/boot-demo | 100.000 | control-synthetic.jsonl:60 |
| worker_applied | 18.300 | synthetic-host/boot-demo | 300.000 | control-synthetic.jsonl:67 |
| parent_ack | 18.305 | synthetic-host/boot-demo | 305.000 | task-events.jsonl:52 |
| ordinary_rtc_resumed | 18.600 | synthetic-host/boot-demo | 600.000 | task-events.jsonl:54 |

缺失指标：无
实际接入时间证据：worker_post_send_field；桥接有效帧：7；旧phase证据：prior_worker_frame。

### demo-5（harvest_2，v5）

| 时间点 | monotonic s | 时钟域 | 距登记 ms | 文件:行 |
|---|---:|---|---:|---|
| registered | 20.000 | synthetic-host/boot-demo | 0.000 | task-events.jsonl:55 |
| boundary | 20.003 | synthetic-host/boot-demo | 3.000 | task-events.jsonl:57 |
| observation_start | 20.004 | synthetic-host/boot-demo | 4.000 | task-events.jsonl:58 |
| observation_complete | 20.024 | synthetic-host/boot-demo | 24.000 | task-events.jsonl:61 |
| inference_start | 20.025 | synthetic-host/boot-demo | 25.000 | task-events.jsonl:62 |
| inference_complete | 20.085 | synthetic-host/boot-demo | 85.000 | task-events.jsonl:63 |
| inference_failure | 缺失 | 缺失 | 缺失 | 缺失 |
| armed | 20.091 | synthetic-host/boot-demo | 91.000 | task-events.jsonl:64 |
| scheduled_bridge_start | 20.100 | synthetic-host/boot-demo | 100.000 | control-synthetic.jsonl:77 |
| worker_applied | 20.300 | synthetic-host/boot-demo | 300.000 | control-synthetic.jsonl:84 |
| parent_ack | 20.305 | synthetic-host/boot-demo | 305.000 | task-events.jsonl:65 |
| ordinary_rtc_resumed | 20.600 | synthetic-host/boot-demo | 600.000 | task-events.jsonl:67 |

缺失指标：无
实际接入时间证据：worker_post_send_field；桥接有效帧：7；旧phase证据：prior_worker_frame。

### demo-6（return_stem，v6）

| 时间点 | monotonic s | 时钟域 | 距登记 ms | 文件:行 |
|---|---:|---|---:|---|
| registered | 22.000 | synthetic-host/boot-demo | 0.000 | task-events.jsonl:68 |
| boundary | 22.003 | synthetic-host/boot-demo | 3.000 | task-events.jsonl:70 |
| observation_start | 22.004 | synthetic-host/boot-demo | 4.000 | task-events.jsonl:71 |
| observation_complete | 22.024 | synthetic-host/boot-demo | 24.000 | task-events.jsonl:74 |
| inference_start | 22.025 | synthetic-host/boot-demo | 25.000 | task-events.jsonl:75 |
| inference_complete | 22.085 | synthetic-host/boot-demo | 85.000 | task-events.jsonl:76 |
| inference_failure | 缺失 | 缺失 | 缺失 | 缺失 |
| armed | 22.091 | synthetic-host/boot-demo | 91.000 | task-events.jsonl:77 |
| scheduled_bridge_start | 22.100 | synthetic-host/boot-demo | 100.000 | control-synthetic.jsonl:94 |
| worker_applied | 22.300 | synthetic-host/boot-demo | 300.000 | control-synthetic.jsonl:101 |
| parent_ack | 22.305 | synthetic-host/boot-demo | 305.000 | task-events.jsonl:78 |
| ordinary_rtc_resumed | 22.600 | synthetic-host/boot-demo | 600.000 | task-events.jsonl:80 |

缺失指标：无
实际接入时间证据：worker_post_send_field；桥接有效帧：7；旧phase证据：prior_worker_frame。

## 七阶段记录

| run | 阶段 | Prompt类别 | 请求数 | 已接纳Chunk数 | 软件生效证据 | 采摘/抓握 |
|---|---|---|---:|---:|---|---|
| synthetic-seven-stage | 首次拿起 | pickup | 0 | 1 | active | 缺失 |
| synthetic-seven-stage | 首次采摘 | harvest | 1 | 1 | active | 缺失 |
| synthetic-seven-stage | 放回整串 | return | 1 | 1 | active | 缺失 |
| synthetic-seven-stage | 调整 | adjust | 1 | 1 | active | 缺失 |
| synthetic-seven-stage | 再次拿起 | pickup | 1 | 1 | active | 缺失 |
| synthetic-seven-stage | 再次采摘 | harvest | 1 | 1 | active | 缺失 |
| synthetic-seven-stage | 放回裸杆 | return | 1 | 1 | active | 缺失 |

## 各类阶段切换

| 分组 | 请求数 | 接入数 | 全部请求接入比例 | 登记→Applied P50 ms |
|---|---:|---:|---:|---:|
| adjust->pickup_2 | 1 | 1 | 1.000 | 300.000 |
| harvest_1->return_truss | 1 | 1 | 1.000 | 300.000 |
| harvest_2->return_stem | 1 | 1 | 1.000 | 300.000 |
| pickup_1->harvest_1 | 1 | 1 | 1.000 | 300.000 |
| pickup_2->harvest_2 | 1 | 1 | 1.000 | 300.000 |
| return_truss->adjust | 1 | 1 | 1.000 | 300.000 |

## 四类Prompt切换

| 分组 | 请求数 | 接入数 | 全部请求接入比例 | 登记→Applied P50 ms |
|---|---:|---:|---:|---:|
| adjust->pickup | 1 | 1 | 1.000 | 300.000 |
| harvest->return | 2 | 2 | 1.000 | 300.000 |
| pickup->harvest | 2 | 2 | 1.000 | 300.000 |
| return->adjust | 1 | 1 | 1.000 | 300.000 |

## 不同旧Chunk相位

| 分组 | 请求数 | 接入数 | 全部请求接入比例 | 登记→Applied P50 ms |
|---|---:|---:|---:|---:|
| [10,15) | 1 | 1 | 1.000 | 300.000 |
| [15,20) | 2 | 2 | 1.000 | 300.000 |
| [20,25) | 1 | 1 | 1.000 | 300.000 |
| [25,30) | 2 | 2 | 1.000 | 300.000 |

## 关节误差与桥接动态

误差为同一日志帧内command与measured的差，未校正反馈延迟。导数使用实际日志间隔；不是硬件编码器同步采样或动态安全认证。

| session / 范围 | 关节组 | 误差RMSE rad | 误差P95 rad | 命令速度峰值 rad/s | 命令加速度峰值 rad/s² | 命令jerk峰值 rad/s³ |
|---|---|---:|---:|---:|---:|---:|
| 6fd4ed3dfaab / 全程 | left_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / 全程 | left_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / 全程 | right_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / 全程 | right_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-1 | left_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-1 | left_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-1 | right_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-1 | right_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-2 | left_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-2 | left_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-2 | right_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-2 | right_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-3 | left_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-3 | left_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-3 | right_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-3 | right_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-4 | left_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-4 | left_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-4 | right_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-4 | right_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-5 | left_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-5 | left_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-5 | right_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-5 | right_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-6 | left_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-6 | left_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-6 | right_arm | 0.001 | 0.001 | 0.003 | 0.000 | 0.000 |
| 6fd4ed3dfaab / demo-6 | right_hand | 0.003 | 0.003 | 0.003 | 0.000 | 0.000 |

## 错误与数据质量

| 类型 | 次数 |
|---|---:|
| 数据质量：frame_gap_or_nonpositive_dt | 5 |

缺失或截断的日志不能自动判成超时/安全停止。快照仅作终态证据，不补造时间点。
浏览器wall clock没有已验证偏移时，Click相关指标保持缺失；跨文件没有同一host/boot声明时，不计算跨文件monotonic差值。
所有详细数据、字段定义、来源校验摘要和质量问题见summary.json及CSV。
