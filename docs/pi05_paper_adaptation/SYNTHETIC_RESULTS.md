# 本轮离线验证结果

开发基线 `64f8f66`，新分支 `pi05-rolling-inpainting`。全部检查在本地进行，没有机器人连接、远程训练服务器访问、真实权重加载或训练。

## 验证命令

隔离 CPU 环境 `/tmp/sharpa-paper-tests`，核心版本及最初基线缺依赖情况见 [审计记录](WP0_AUDIT.md)。

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src:$PWD"
export JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1
/tmp/sharpa-paper-tests/bin/python -m pytest deploy/fr3_sharpa_runtime/tests -q
uv lock --check --offline --python /tmp/sharpa-paper-tests/bin/python
```

最终完整 runtime 测试：**240 passed**。包含原有初始化、Home、速度档、SDK slew、Delta、停止/失能确认及清理回归，以及32项新增测试。新增验证包括：

- 三轮以上参考切片与提交、缓存过期、模型/统计变化、服务重启及部分执行拒绝；正常每次新建socket不清空参考。
- 非零状态下真实 training data transform 与参考编码结果相同；协方差工具使用当前 Delta/quantile 资产，完整矩阵加载及错误身份拒绝、拒绝覆盖。
- CPU上用小型假向量场执行**实际 Pi0.sample_actions 的 Euler 和 module_jit 路径**，验证无约束基线等价性、更新后的时间阈值、后期释放、identity传播退化及过大修正审计。不是完整Gemma模型加载。
- 实际 Policy/TimedPolicy 接口以及真实 msgpack 编解码的假socket协议测试，验证迟到响应不会返回可执行结果、启动UUID更换和CLI参数。
- 26→20端点、手部变化取消压缩、严格样条检查失败退回26点、原始序列仍无效时停止；慢速时间伸缩与bridge数量分别记录。
- **真实 SDK worker 主循环配合完全模拟的ROS/SDK对象**，验证消费26个模型点但下发20个压缩点、最终目标、逐点审计、第三点发送失败时只记录两次成功发送，以及独立退出确认。未导入或连接供应商SDK。
- 新增纯模块及新增测试通过 Ruff 检查；`git diff --check`通过；离线lock校验通过，没有升级原依赖图中的版本，仅将已锁定的SciPy声明为直接依赖。

## 固定 seed 合成对照

```bash
/tmp/sharpa-paper-tests/bin/python -m deploy.fr3_sharpa_runtime.paper_replay \
  --seed 42 --rounds 5 --report /tmp/sharpa-paper-fast.json
/tmp/sharpa-paper-tests/bin/python -m deploy.fr3_sharpa_runtime.paper_replay \
  --seed 42 --rounds 5 --speed-scale 0.1 --report /tmp/sharpa-paper-slow.json
```

完整原始输出：[fast](synthetic_fast.json)、[slow](synthetic_slow.json)。各模式使用相同随机噪声数组，SHA256写入报告。相关传播使用解析的合成协方差，不是本任务训练集统计。合成停顿固定60ms，时间轴包含该停顿和bridge；不是工控机实际观测/通信延迟。

下表取左臂的最大目标交接跳变；其他三组关节及玩具目标RMSE、实际指令时间轴速度/加速度见JSON。

| 模式 | 消费模型点/轮 | 最大跳变 rad | fast合成总时长 s | slow合成总时长 s |
|---|---:|---:|---:|---:|
| 原50/15 | 15 | 0.001732 | 3.407 | 24.407 |
| rolling，无约束 | 26 | 0.002108 | 5.173 | 42.673 |
| Soft | 26 | 0.002003 | 5.173 | 42.673 |
| 相关传播 | 26 | 0.001899 | 5.140 | 42.640 |
| Soft + cubic | 26 | 0.002003 | 4.173 | 32.673 |
| 相关传播 + cubic | 26 | 0.001899 | 4.140 | 32.640 |

此玩具向量场中Soft与相关传播的目标交接跳变略有下降，但**不证明真实模型有相同收益**。原50/15与26点模式的观测起点/重规划间隔不同，不能用此表给它们排名。压缩减少指令播放时长，不减少采样计算本身；更平滑、目标RMSE变化或更快均不能直接解释为成功率提升。CPU计算时间含测试/JIT开销，不是部署延迟评测。

## 待完成验收

1. 工控机完整环境安装、checkpoint及norm stats身份检查；实验sampler实际GPU预热和有限输出。
2. 真实留出数据下相同观测/噪声的50/15基线、rolling、Soft消融；不同模型分别使用自己的统计。
3. 如需相关传播，从当前任务训练窗口准备独立资产，再做同条件对照；不将旧权重噪声分布改成相关噪声。
4. 现场首先有限轮数，检查设备身份、观测时效、接触阶段、目标/反馈误差、压缩回退、停止与失能确认。真实机器人表现与番茄采摘成功率仍待验证。
5. 相关噪声训练、逐时间点归一化、异步推理、延迟补偿均不在本轮实现中。
