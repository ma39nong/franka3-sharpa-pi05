# 已核对的 RTC 训练证据

这里只保存精简训练配置和文件摘要，不包含模型权重、统计数组或数据集。

- `train-rtc-7999.json`：2026-10-31 的普通全量 Delta Train RTC，训练提交 `c135c79`。
- `train-paper-rtc-7999.json`：2026-10-31 的 Paper Delta Train RTC，训练提交 `e93a3aa`。

2026-11-02 从对应训练日志读取。`source.training_record_sha256`保留原始训练JSON的摘要；config只保留部署所需字段，inputs保留相关统计和Paper资产摘要。

自动识别同时匹配 checkpoint 的 `params/manifest.ocdbt` 和 `params/_METADATA` 摘要，再核对统计和训练算法。目录名称、7999步数或相同统计本身不能触发该绑定。保持checkpoint内容不变时可以移动、改名；重新导出或重新保存的权重需要重新核验并提供匹配训练记录。显式指定的训练记录或checkpoint根目录的 `train_rtc_record.json` 优先。

新增记录前必须独立核对训练代码、训练JSON、checkpoint元数据、统计摘要和Paper资产；不要复制现有记录后只改路径或步数。

- `train-paper-rtc-1102-6000.json` / `train-paper-rtc-1102-7999.json`：2026-11-02 `…1102_1309_paper_t3` 的两个步骤，按服务器运行 JSON、checkpoint manifest/metadata 与统计/Paper bundle绑定。后续实验优先携带 effective_config 归档，或由操作员在 GUI 明确填写训练 RTC 最大延迟，无需扩展内置名单。
