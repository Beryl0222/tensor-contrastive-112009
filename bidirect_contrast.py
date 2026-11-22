"""双向多正例对比损失算子（图文两侧对称，分片归属明确）。

设计要点（与 ``contract.md`` 和运行约定一一对应）：

* 每条向量以 ``(资料身份, 版本)`` 进入本步，并记录其生产分片；重发的同身份同版本
  是同一逻辑向量，展品相同但身份不同的向量仍是不同候选；缺块绝不补零，缺失直接
  把该步标记为不可更新。
* 允许集合 = 同可见分组 ∩ 两侧均有效 ∩ 不在显式排除对中；正例只能来自本步冻结
  清单的 ``matches``，不按展品名自动扩展。``candidate_only`` 的资料只做对侧候选、
  不发起本侧查询。
* 每方向仅对“有候选且有已确认正例”的锚点求平均；无候选锚点从该方向均值剔除；
  整张图没有正例时不贡献锚点损失，但作为他人候选仍会收到梯度。
* 生产规模下不构建、也不保存两侧全量交叉打分矩阵：前向按锚点流式打分，只缓存
  冻结的元数据和两个稳定口径下的 log-sum-exp 标量；读向量快照不计临时预算，
  反向需要的分数从快照重算。
* 反向请求晚到时，按冻结步重取原 ``(身份, 版本)`` 的向量；取不到或身份冲突时
  整步调为不可更新，不允许只更新收到梯度的半边。
* 锚点按身份排序后以固定顺序处理，分片抵达顺序与划分变化不改变数学结果；
  全部数值在 FP64 下用 max-shift 的稳定 log-sum-exp 计算。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = [
    "Record",
    "StepManifest",
    "ScratchBudget",
    "DirectionReport",
    "StepOutcome",
    "FrozenStep",
    "BackwardOutcome",
    "GradientPiece",
    "StepIssue",
    "record_from_row",
    "manifest_from_session",
    "run_step",
    "backward_step",
    "build_synthetic_workload",
    "ATOL",
    "RTOL",
]

# 训练契约：FP64 稳定口径下的比对容差。
ATOL = 1e-9
RTOL = 1e-7

IMAGE = "image"
TEXT = "text"
_SIDES = (IMAGE, TEXT)
_OPPOSITE = {IMAGE: TEXT, TEXT: IMAGE}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """一侧的一条向量记录，对应交换资料的一行。"""

    material_id: str
    version: int
    shard: str
    exhibit: str
    group: str
    valid: bool
    vector: np.ndarray

    @property
    def key(self) -> tuple[str, int]:
        return self.material_id, self.version


def record_from_row(row: Sequence) -> Record:
    """把 JSON 行 ``[身份, 版本, 分片, 展品, 可见分组, 有效, 向量]`` 转为记录。

    向量一律转成 FP64 连续数组；调用方也可直接构造 :class:`Record`。
    """
    material_id, version, shard, exhibit, group, valid, vector = row
    return Record(
        material_id=str(material_id),
        version=int(version),
        shard=str(shard),
        exhibit=str(exhibit),
        group=str(group),
        valid=bool(valid),
        vector=np.asarray(vector, dtype=np.float64),
    )


@dataclass(frozen=True)
class StepManifest:
    """本步冻结的配对清单。配对更正在步中到达后应构建下一步的清单。"""

    matches: frozenset[tuple[str, str]]
    exclusions: frozenset[tuple[str, str]]
    candidate_only: frozenset[str]
    expected_versions: Mapping[str, int]

    @classmethod
    def create(
        cls,
        matches: Iterable[tuple[str, str]],
        exclusions: Iterable[tuple[str, str]] = (),
        candidate_only: Iterable[str] = (),
        expected_versions: Optional[Mapping[str, int]] = None,
    ) -> "StepManifest":
        return cls(
            matches=frozenset((str(i), str(t)) for i, t in matches),
            exclusions=frozenset((str(i), str(t)) for i, t in exclusions),
            candidate_only=frozenset(str(x) for x in candidate_only),
            expected_versions=dict(expected_versions or {}),
        )


@dataclass(frozen=True)
class StepIssue:
    """使本步不可更新的身份冲突/缺块（无效记录不属于此类，仅被排除）。"""

    code: str
    material_id: str
    detail: str


@dataclass
class ScratchBudget:
    """临时分数、归约与中间梯度的字节预算（不含只读快照与最终梯度）。"""

    limit_bytes: int
    high_water_bytes: int = field(default=0, init=False)
    _current: int = field(default=0, init=False)

    class _Charge:
        def __init__(self, budget: "ScratchBudget", shape, dtype) -> None:
            self.budget = budget
            self.nbytes = math.prod(shape) * np.dtype(dtype).itemsize
            self.shape = shape
            self.dtype = dtype
            self.array: Optional[np.ndarray] = None

        def __enter__(self) -> np.ndarray:
            self.budget._current += self.nbytes
            if self.budget._current > self.budget.high_water_bytes:
                self.budget.high_water_bytes = self.budget._current
            if self.budget._current > self.budget.limit_bytes:
                self.budget._current -= self.nbytes
                raise MemoryError(
                    f"算子临时工作区超出预算：{self.budget._current + self.nbytes} > "
                    f"{self.budget.limit_bytes} 字节"
                )
            self.array = np.empty(self.shape, dtype=self.dtype)
            return self.array

        def __exit__(self, *exc) -> None:
            self.budget._current -= self.nbytes
            self.array = None

    def temp_array(self, shape: tuple[int, ...], dtype=np.float64):
        """在预算内申请一个临时数组，离开 ``with`` 即归还额度。"""
        return self._Charge(self, shape, dtype)


# 不限制预算的默认工作区（仅小步使用；生产步请传入 load_profile 的预算）。
_UNLIMITED = ScratchBudget(limit_bytes=2**63 - 1)


# ---------------------------------------------------------------------------
# 前向冻结包与结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _AnchorPlan:
    """单个锚点在本步冻结的计算计划（不含任何分数矩阵）。"""

    anchor_id: str
    version: int
    shard: str
    candidate_ids: tuple[str, ...]
    positive_ids: tuple[str, ...]


@dataclass(frozen=True)
class _SideTable:
    records: Mapping[str, Record]
    order: tuple[str, ...]
    invalid_ids: frozenset[str]


@dataclass(frozen=True)
class DirectionReport:
    """一个方向的交付信息。"""

    side: str
    available: bool
    # 有效锚点分母：参与该方向均值的锚点数（有候选且有正例）。
    anchor_denominator: int
    # 该方向锚点损失之和（均值 = loss_sum / anchor_denominator）。
    loss_sum: float
    # 发起查询的锚点数（candidate_only 不发起）。
    anchors_seen: int
    # 有候选但没有任何已确认正例、因此不贡献锚点损失的锚点。
    anchors_without_positive: tuple[str, ...]
    # 一个候选都没有、从该方向均值剔除的锚点。
    anchors_without_candidate: tuple[str, ...]


@dataclass(frozen=True)
class FrozenStep:
    """前向后冻结的步。反向晚到时仍按这里的正负例、版本与归一权重计算。"""

    step_id: int
    inverse_temperature: float
    plans: Mapping[str, tuple[_AnchorPlan, ...]]
    snapshots: Mapping[str, Mapping[str, Optional[np.ndarray]]]
    fingerprints: Mapping[str, Mapping[str, np.ndarray]]
    shards: Mapping[str, Mapping[str, str]]
    versions: Mapping[str, Mapping[str, int]]
    weight_per_anchor: float
    reports: Mapping[str, DirectionReport]
    loss: Optional[float]
    can_update: bool
    issues: tuple[StepIssue, ...]
    missing_ids: tuple[str, ...]
    invalid_ids: Mapping[str, tuple[str, ...]]
    manifest: StepManifest


@dataclass(frozen=True)
class GradientPiece:
    """归属明确的一片梯度：属于哪一侧、哪个身份版本、由哪个分片生产。"""

    side: str
    material_id: str
    version: int
    shard: str
    gradient: np.ndarray


@dataclass(frozen=True)
class StepOutcome:
    """调用端拿到的前向结果。"""

    loss: Optional[float]
    anchor_denominators: Mapping[str, int]
    reports: Mapping[str, DirectionReport]
    frozen: FrozenStep
    can_update: bool
    issues: tuple[StepIssue, ...]
    missing_ids: tuple[str, ...]
    invalid_ids: Mapping[str, tuple[str, ...]]
    budget: ScratchBudget


@dataclass(frozen=True)
class BackwardOutcome:
    """反向交付：两侧归属明确的梯度与本步可更新性。"""

    can_update: bool
    gradients: tuple[GradientPiece, ...]
    issues: tuple[StepIssue, ...]
    loss: Optional[float]
    anchor_denominators: Mapping[str, int]
    by_shard: Mapping[tuple[str, str], tuple["GradientPiece", ...]]


# ---------------------------------------------------------------------------
# 稳定数值原语
# ---------------------------------------------------------------------------


def _logsumexp_inplace(work: np.ndarray, budget: "ScratchBudget") -> float:
    """对 ``work`` 就地做 FP64 max-shift 稳定 log-sum-exp；数组本身已计入预算。"""
    m = float(np.max(work))
    work -= m
    np.exp(work, out=work)
    return m + math.log(float(np.sum(work)))


def _anchor_loss(
    anchor_vec: np.ndarray,
    cand_vecs: np.ndarray,
    positive_idx: np.ndarray,
    tau: float,
    budget: "ScratchBudget",
) -> float:
    """单锚点多正例损失：全候选 log-sum-exp 与正例 log-sum-exp 之差。

    所有临时分数与归约都在 ``budget`` 内申请，峰值约 3 个候选长度的 FP64 数组。
    """
    n = cand_vecs.shape[0]
    with budget.temp_array((n,)) as scores, budget.temp_array((n,)) as denom_buf, \
            budget.temp_array((positive_idx.shape[0],)) as numer_buf:
        np.dot(cand_vecs, anchor_vec, out=scores)
        scores *= tau

        denom_buf[:] = scores
        denom_lse = _logsumexp_inplace(denom_buf, budget)

        numer_buf[:] = scores[positive_idx]
        numer_lse = _logsumexp_inplace(numer_buf, budget)
    return denom_lse - numer_lse


# ---------------------------------------------------------------------------
# 侧表构建与冲突检测
# ---------------------------------------------------------------------------


def _build_side_table(
    side: str,
    rows: Iterable[Sequence | Record],
    expected_versions: Mapping[str, int],
    issues: list[StepIssue],
) -> _SideTable:
    by_id: dict[str, Record] = {}
    invalid: list[str] = []

    for raw in rows:
        rec = raw if isinstance(raw, Record) else record_from_row(raw)
        if rec.vector.dtype != np.float64:
            rec = Record(
                material_id=rec.material_id,
                version=rec.version,
                shard=rec.shard,
                exhibit=rec.exhibit,
                group=rec.group,
                valid=rec.valid,
                vector=np.asarray(rec.vector, dtype=np.float64),
            )

        prior = by_id.get(rec.material_id)
        if prior is None:
            by_id[rec.material_id] = rec
            if not rec.valid:
                invalid.append(rec.material_id)
            continue

        # 同一身份重发：同版本必须是同一逻辑向量（含同一生产分片归属）。
        same_version = prior.version == rec.version
        same_shard = prior.shard == rec.shard
        same_shape = prior.vector.shape == rec.vector.shape
        same_values = same_shape and np.array_equal(prior.vector, rec.vector)
        if same_version and same_shard and same_values:
            continue  # 重复传输，折叠为一条。
        code = "duplicate_conflict" if same_version else "version_conflict"
        issues.append(
            StepIssue(
                code,
                rec.material_id,
                f"{side} 侧身份 {rec.material_id} 重发内容不一致："
                f"已有 v{prior.version}@{prior.shard}，收到 v{rec.version}@{rec.shard}",
            )
        )
        # 冲突身份不折叠：保留先到者，但该步已不可更新。

    # 清单声明的版本必须与实际抵达版本一致（不同版可并存，但本步只认声明版）。
    for material_id, rec in by_id.items():
        declared = expected_versions.get(material_id)
        if declared is not None and declared != rec.version:
            issues.append(
                StepIssue(
                    "version_conflict",
                    material_id,
                    f"{side} 侧身份 {material_id} 声明 v{declared}，实际抵达 v{rec.version}",
                )
            )

    order = tuple(sorted(by_id))
    return _SideTable(
        records=by_id,
        order=order,
        invalid_ids=frozenset(invalid),
    )


def _check_declared_presence(
    tables: Mapping[str, _SideTable],
    manifest: StepManifest,
) -> list[StepIssue]:
    issues: list[StepIssue] = []
    present = {side: set(t.records) for side, t in tables.items()}
    for material_id, declared in manifest.expected_versions.items():
        if not any(material_id in present[side] for side in _SIDES):
            # 缺块：绝不补零，直接标记不可更新。
            issues.append(
                StepIssue(
                    "missing",
                    material_id,
                    f"清单声明身份 {material_id} v{declared} 未在任一分片抵达",
                )
            )
    # matches / exclusions 引用的身份也必须存在，否则清单与向量集合不自洽。
    for image_id, text_id in (*manifest.matches, *manifest.exclusions):
        if image_id not in present[IMAGE]:
            issues.append(StepIssue("missing", image_id, f"清单引用图像身份 {image_id} 缺失"))
        if text_id not in present[TEXT]:
            issues.append(StepIssue("missing", text_id, f"清单引用文本身份 {text_id} 缺失"))
    return issues


# ---------------------------------------------------------------------------
# 允许集合与锚点计划
# ---------------------------------------------------------------------------


def _is_allowed(image: Record, text: Record, manifest: StepManifest) -> bool:
    if not (image.valid and text.valid):
        return False
    if image.group != text.group:
        return False
    if (image.material_id, text.material_id) in manifest.exclusions:
        return False
    return True


def _build_plans(
    tables: Mapping[str, _SideTable],
    manifest: StepManifest,
    budget: ScratchBudget,
) -> dict[str, tuple[_AnchorPlan, ...]]:
    plans: dict[str, tuple[_AnchorPlan, ...]] = {}

    for anchor_side in _SIDES:
        own = tables[anchor_side]
        opp = tables[_OPPOSITE[anchor_side]]
        side_plans: list[_AnchorPlan] = []

        for anchor_id in own.order:
            if anchor_id in manifest.candidate_only:
                continue  # 仅参加对侧候选，不发起本侧查询。
            anchor = own.records[anchor_id]
            if not anchor.valid:
                continue  # 无效资料不发起查询（仍可能因无效而根本不进任何允许集合）。

            candidate_ids: list[str] = []
            positive_ids: list[str] = []
            for opp_id in opp.order:
                cand = opp.records[opp_id]
                if anchor_side == IMAGE:
                    allowed = _is_allowed(anchor, cand, manifest)
                    paired = (anchor_id, opp_id) in manifest.matches
                else:
                    allowed = _is_allowed(cand, anchor, manifest)
                    paired = (opp_id, anchor_id) in manifest.matches
                if not allowed:
                    continue
                candidate_ids.append(opp_id)
                # 只有 matches 中登记且仍然允许的组合才是正例，不按展品名扩展。
                if paired:
                    positive_ids.append(opp_id)

            side_plans.append(
                _AnchorPlan(
                    anchor_id=anchor_id,
                    version=anchor.version,
                    shard=anchor.shard,
                    candidate_ids=tuple(candidate_ids),
                    positive_ids=tuple(positive_ids),
                )
            )
        plans[anchor_side] = tuple(side_plans)
    return plans


# ---------------------------------------------------------------------------
# 前向
# ---------------------------------------------------------------------------


def run_step(
    step_id: int,
    images: Iterable[Sequence | Record],
    texts: Iterable[Sequence | Record],
    manifest: StepManifest,
    inverse_temperature: float,
    budget: Optional[ScratchBudget] = None,
) -> StepOutcome:
    """按本步冻结清单计算全局双向对比损失（前向）。

    分片抵达顺序不影响结果：记录先按身份折叠排序，锚点与候选均按身份升序处理。
    向量快照是只读输入，不计入 ``budget``；临时分数/归约计入。
    """
    budget = budget or _UNLIMITED
    tau = float(inverse_temperature)
    issues: list[StepIssue] = []

    tables = {
        IMAGE: _build_side_table(IMAGE, images, manifest.expected_versions, issues),
        TEXT: _build_side_table(TEXT, texts, manifest.expected_versions, issues),
    }
    issues.extend(_check_declared_presence(tables, manifest))

    blocking = tuple(sorted(issues, key=lambda x: (x.code, x.material_id)))
    can_update = len(blocking) == 0

    plans = _build_plans(tables, manifest, budget)

    # 只读快照（不计预算）：反向晚到时从这里重取原版；取不到再交给 refetch 回调。
    # 指纹用于核对分片重发的是否确为原步同身份同版本向量，冲突即整步作废。
    snapshots: dict[str, dict[str, Optional[np.ndarray]]] = {s: {} for s in _SIDES}
    fingerprints: dict[str, dict[str, np.ndarray]] = {s: {} for s in _SIDES}
    shards: dict[str, dict[str, str]] = {s: {} for s in _SIDES}
    versions: dict[str, dict[str, int]] = {s: {} for s in _SIDES}
    for side in _SIDES:
        for mid in tables[side].order:
            rec = tables[side].records[mid]
            snapshots[side][mid] = rec.vector.copy()
            fingerprints[side][mid] = rec.vector.copy()
            shards[side][mid] = rec.shard
            versions[side][mid] = rec.version

    available_sides: list[str] = []
    reports: dict[str, DirectionReport] = {}
    total_loss = 0.0

    for side in _SIDES:
        seen = 0
        no_positive: list[str] = []
        no_candidate: list[str] = []
        loss_sum = 0.0
        denominator = 0

        for plan in plans[side]:
            seen += 1
            if not plan.candidate_ids:
                no_candidate.append(plan.anchor_id)
                continue  # 锚点没有候选：从该方向均值剔除。
            anchor_vec = snapshots[side][plan.anchor_id]
            n_cand = len(plan.candidate_ids)
            # 候选堆叠是单锚点临时量，计入预算；绝不形成两侧全量交叉矩阵。
            with budget.temp_array((n_cand, anchor_vec.shape[0])) as cand_vecs:
                for j, cand_id in enumerate(plan.candidate_ids):
                    cand_vecs[j] = snapshots[_OPPOSITE[side]][cand_id]
                positive_idx = np.asarray(
                    [plan.candidate_ids.index(p) for p in plan.positive_ids],
                    dtype=np.int64,
                )
                if positive_idx.size == 0:
                    # 没有可用正例：不贡献自己的锚点损失，但向量仍是别人的候选。
                    no_positive.append(plan.anchor_id)
                    continue
                anchor_loss = _anchor_loss(
                    anchor_vec, cand_vecs, positive_idx, tau, budget
                )
            loss_sum += anchor_loss
            denominator += 1

        available = denominator > 0
        if available:
            available_sides.append(side)
            total_loss += loss_sum
        reports[side] = DirectionReport(
            side=side,
            available=available,
            anchor_denominator=denominator,
            loss_sum=float(loss_sum),
            anchors_seen=seen,
            anchors_without_positive=tuple(no_positive),
            anchors_without_candidate=tuple(no_candidate),
        )

    if not available_sides:
        loss: Optional[float] = None  # 两方向均空：报告无方向损失。
        weight_per_anchor = 0.0
    else:
        # 两方向都可用时各占 1/2；仅一方向可用时取该方向（每个锚点权重为 1）。
        weight_per_anchor = 0.5 if len(available_sides) == 2 else 1.0
        loss = weight_per_anchor * total_loss

    frozen = FrozenStep(
        step_id=step_id,
        inverse_temperature=tau,
        plans=plans,
        snapshots=snapshots,
        fingerprints=fingerprints,
        shards=shards,
        versions=versions,
        weight_per_anchor=weight_per_anchor,
        reports=reports,
        loss=loss,
        can_update=can_update,
        issues=blocking,
        missing_ids=tuple(sorted({i.material_id for i in blocking if i.code == "missing"})),
        invalid_ids={
            side: tuple(sorted(tables[side].invalid_ids)) for side in _SIDES
        },
        manifest=manifest,
    )
    return StepOutcome(
        loss=loss,
        anchor_denominators={s: reports[s].anchor_denominator for s in _SIDES},
        reports=reports,
        frozen=frozen,
        can_update=can_update,
        issues=blocking,
        missing_ids=frozen.missing_ids,
        invalid_ids=frozen.invalid_ids,
        budget=budget,
    )


# ---------------------------------------------------------------------------
# 反向
# ---------------------------------------------------------------------------


def _revalidate_vector(
    side: str,
    material_id: str,
    expected_version: int,
    expected_shard: str,
    stored: Optional[np.ndarray],
    fingerprint: np.ndarray,
    refetch: Optional[Callable[[str, str, int, str], Optional[np.ndarray]]],
    issues: list[StepIssue],
) -> Optional[np.ndarray]:
    """按原步身份版本取回向量。

    * 工作区仍持有原版只读快照时直接复用（原步版本、正负例与权重不变）；
    * 快照丢失时由分片重发；重发内容必须与冻结指纹在契约容差内一致，否则按
      向量身份冲突登记，整步作废——绝不拿冲突向量或零向量顶替；
    * 分片确认原版无法取回（回调返回 ``None``）时登记缺块。
    """
    if stored is not None:
        return stored
    if refetch is None:
        issues.append(
            StepIssue("missing", material_id, f"{side} 侧原版向量已无法取回且无重发通道")
        )
        return None
    got = refetch(side, material_id, expected_version, expected_shard)
    if got is None:
        issues.append(
            StepIssue("missing", material_id, f"{side} 侧分片 {expected_shard} 确认原版无法取回")
        )
        return None
    vec = np.asarray(got, dtype=np.float64)
    if vec.shape != fingerprint.shape or not np.allclose(
        vec, fingerprint, atol=ATOL, rtol=RTOL
    ):
        issues.append(
            StepIssue(
                "vector_conflict",
                material_id,
                f"{side} 侧分片 {expected_shard} 重发的 {material_id} v{expected_version} "
                "与冻结指纹不一致，向量身份冲突",
            )
        )
        return None
    return vec


def backward_step(
    frozen: FrozenStep,
    refetch: Optional[Callable[[str, str, int, str], Optional[np.ndarray]]] = None,
    budget: Optional[ScratchBudget] = None,
) -> BackwardOutcome:
    """对冻结步求反向梯度并按生产分片归属送回。

    晚到的反向请求继续使用原步的正负例、向量版本与归一权重；若缺块或向量身份
    冲突，整步标记不可更新，不会只交付半边梯度。
    """
    budget = budget or _UNLIMITED
    if not frozen.can_update:
        return BackwardOutcome(
            can_update=False,
            gradients=(),
            issues=frozen.issues,
            loss=frozen.loss,
            anchor_denominators={s: r.anchor_denominator for s, r in frozen.reports.items()},
            by_shard={},
        )

    tau = frozen.inverse_temperature
    w = frozen.weight_per_anchor
    issues: list[StepIssue] = []

    # 最终梯度不在临时预算内（契约：交付给训练端的最终梯度不计）。
    # 用冻结指纹定形：即使工作区快照已被清空，梯度容器仍按原步身份建立。
    grads: dict[str, dict[str, np.ndarray]] = {
        side: {
            mid: np.zeros_like(frozen.fingerprints[side][mid])
            for mid in frozen.fingerprints[side]
        }
        for side in _SIDES
    }

    for anchor_side in _SIDES:
        report = frozen.reports[anchor_side]
        if not report.available:
            continue
        opp_side = _OPPOSITE[anchor_side]

        for plan in frozen.plans[anchor_side]:
            if not plan.positive_ids:
                continue
            anchor_vec = _revalidate_vector(
                anchor_side,
                plan.anchor_id,
                plan.version,
                plan.shard,
                frozen.snapshots[anchor_side].get(plan.anchor_id),
                frozen.fingerprints[anchor_side][plan.anchor_id],
                refetch,
                issues,
            )
            if anchor_vec is None:
                continue

            cand_vectors: list[np.ndarray] = []
            for cand_id in plan.candidate_ids:
                vec = _revalidate_vector(
                    opp_side,
                    cand_id,
                    frozen.versions[opp_side][cand_id],
                    frozen.shards[opp_side][cand_id],
                    frozen.snapshots[opp_side].get(cand_id),
                    frozen.fingerprints[opp_side][cand_id],
                    refetch,
                    issues,
                )
                if vec is None:
                    cand_vectors = []
                    break
                cand_vectors.append(vec)
            if not cand_vectors:
                continue

            width = anchor_vec.shape[0]
            n_cand = len(cand_vectors)
            n_pos = len(plan.positive_ids)
            # 单锚点工作集：候选堆叠 + 分数/softmax/系数 + 正例工作数组 + 宽度向量。
            # 峰值只随该锚点的允许候选数增长，与两侧全量交叉矩阵无关。
            with budget.temp_array((n_cand, width)) as cand_mat, \
                    budget.temp_array((n_cand,)) as scores, \
                    budget.temp_array((n_cand,)) as probs, \
                    budget.temp_array((n_cand,)) as coeff, \
                    budget.temp_array((n_pos,)) as pos_work, \
                    budget.temp_array((width,)) as anchor_grad:
                np.stack(cand_vectors, out=cand_mat)
                np.dot(cand_mat, anchor_vec, out=scores)
                scores *= tau

                np.subtract(scores, float(np.max(scores)), out=probs)
                np.exp(probs, out=probs)
                probs /= float(np.sum(probs))

                positive_idx = np.asarray(
                    [plan.candidate_ids.index(p) for p in plan.positive_ids], dtype=np.int64
                )
                pos_work[:] = scores[positive_idx]
                pos_work -= float(np.max(pos_work))
                np.exp(pos_work, out=pos_work)
                pos_work /= float(np.sum(pos_work))

                # 先累计未归一的单锚点原始导数；方向分母与组合权重在汇总后统一
                # 按“向量所属侧”缩放一次，避免跨侧候选被重复缩放。
                np.copyto(coeff, probs)
                coeff[positive_idx] -= pos_work
                coeff *= tau

                # 锚点梯度 = Σ_j coeff_j * 候选_j（中间梯度，计入预算）。
                anchor_grad.fill(0.0)
                np.dot(coeff, cand_mat, out=anchor_grad)
                grads[anchor_side][plan.anchor_id] += anchor_grad

                # 候选侧按身份 scatter-add：coeff_j * 锚点向量。
                for j, cand_id in enumerate(plan.candidate_ids):
                    grads[opp_side][cand_id] += coeff[j] * anchor_vec

    if issues:
        blocking = tuple(sorted(issues, key=lambda x: (x.code, x.material_id)))
        return BackwardOutcome(
            can_update=False,
            gradients=(),
            issues=blocking,
            loss=frozen.loss,
            anchor_denominators={s: r.anchor_denominator for s, r in frozen.reports.items()},
            by_shard={},
        )

    # 按向量所属侧统一归一：有效锚点分母 + 两方向各占 1/2 的组合权重。
    for side in _SIDES:
        report = frozen.reports[side]
        if not report.available:
            continue
        side_scale = w / report.anchor_denominator
        for mid in grads[side]:
            grads[side][mid] *= side_scale

    pieces: list[GradientPiece] = []
    by_shard: dict[tuple[str, str], list[GradientPiece]] = {}
    for side in _SIDES:
        for mid in sorted(grads[side]):
            piece = GradientPiece(
                side=side,
                material_id=mid,
                version=frozen.versions[side][mid],
                shard=frozen.shards[side][mid],
                gradient=grads[side][mid],
            )
            pieces.append(piece)
            by_shard.setdefault((side, piece.shard), []).append(piece)

    shard_packages = {key: tuple(bundle) for key, bundle in sorted(by_shard.items())}
    return BackwardOutcome(
        can_update=True,
        gradients=tuple(pieces),
        issues=(),
        loss=frozen.loss,
        anchor_denominators={s: r.anchor_denominator for s, r in frozen.reports.items()},
        # 训练端按 (侧, 分片) 取走归属明确的梯度包，包内带身份版本可核对。
        by_shard=shard_packages,
    )


# ---------------------------------------------------------------------------
# 会话装载与合成工作量
# ---------------------------------------------------------------------------


def manifest_from_session(session: Mapping, candidate_only: Optional[Iterable[str]] = None) -> StepManifest:
    """从交换资料 JSON 构建本步冻结清单。

    ``candidate_only`` 缺省取会话顶层字段；``asymmetric_trial`` 等场景可显式传入。
    ``next_add`` 一类配对更正在步中到达时不要塞进本清单，应构建下一步清单。
    """
    co = session["candidate_only"] if candidate_only is None else list(candidate_only)
    return StepManifest.create(
        matches=session["matches"],
        exclusions=session["exclusions"],
        candidate_only=co,
        expected_versions=session["step_versions"],
    )


def build_synthetic_workload(profile: Mapping) -> tuple[list[list], list[list], StepManifest]:
    """按 ``load_profile`` 生成等价合成工作量（契约给定的确定性构造）。

    向量第 j 个分量为 ``(((i+1)*(j+3)*17) % 113 - 56)/16``，文本侧用 ``i+3``；
    可见分组为 ``i % 8``；正例为同组相同 i 或 i+8（后者须在目标侧范围内）；
    分片为 ``i % shards``。
    """
    n_img = int(profile["images"])
    n_txt = int(profile["texts"])
    width = int(profile["width"])
    shards = int(profile["shards"])

    def vec(i: int, offset: int) -> list[float]:
        return [(((i + offset) * (j + 3) * 17) % 113 - 56) / 16.0 for j in range(width)]

    images: list[list] = []
    texts: list[list] = []
    expected: dict[str, int] = {}
    for i in range(n_img):
        mid = f"i-{i}"
        images.append([mid, 1, f"shard{i % shards}", f"ex{i}", f"g{i % 8}", True, vec(i, 1)])
        expected[mid] = 1
    for i in range(n_txt):
        mid = f"t-{i}"
        texts.append([mid, 1, f"shard{i % shards}", f"ex{i}", f"g{i % 8}", True, vec(i, 3)])
        expected[mid] = 1

    matches: set[tuple[str, str]] = set()
    exclusions: frozenset[tuple[str, str]] = frozenset()
    for i in range(n_img):
        for k in (i, i + 8):
            if k < n_txt and k % 8 == i % 8:
                matches.add((f"i-{i}", f"t-{k}"))
    manifest = StepManifest.create(
        matches=matches,
        exclusions=exclusions,
        candidate_only=(),
        expected_versions=expected,
    )
    return images, texts, manifest
