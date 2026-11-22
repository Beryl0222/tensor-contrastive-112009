# 图文向量交换格式

本项目资料在 `fixtures/session.json`，采用合成数据。记录字段和数值口径见 [资料协议](contract.md)。

使用 Python 3.11 和固定依赖。新环境先执行 `python3 -m pip install -r requirements.txt`。

运行 `python3 inspect_fixture.py` 可以核对资料文件、内容指纹和顶层记录结构。执行 `python3 -m unittest discover -s tests -v` 检查资料中的逻辑引用与输入关系。

## 双向对比损失算子

`bidirect_contrast.py` 是训练作业内使用的双向多正例对比损失算子：

- `run_step(...)` 按本步冻结的 `StepManifest`（可见分组、显式排除、有效标记、`matches` 正例、`candidate_only`）计算图文两方向损失，返回损失、两侧有效锚点分母、方向报告与冻结步 `FrozenStep`；
- `backward_step(frozen, refetch=...)` 晚到也按原步的正负例、向量版本与归一权重求梯度，按 `(侧, 生产分片)` 交付归属明确的 `GradientPiece`；缺块、版本/向量身份冲突时 `can_update=False` 且不交付半边梯度，缺块绝不补零；
- 全程 FP64 稳定 log-sum-exp，不构建也不保存全量交叉打分矩阵，临时分数/归约/中间梯度受 `ScratchBudget` 约束（只读向量快照与最终梯度不计预算）；
- `build_synthetic_workload(profile)` 按 `load_profile` 生成等价合成工作量，用于全规模预算与确定性核验。

隔离模式下运行测试：`python3 -I -m unittest discover -s tests -v`。
