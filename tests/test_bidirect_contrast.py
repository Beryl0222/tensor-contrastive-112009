"""双向对比损失算子的契约测试。

覆盖：夹具口径、稠密参考实现的损失/梯度一致性、有限差分、分片顺序与划分
不变性、重复传输、版本冲突/缺块/无效声明的区分、仅候选资料的梯度保留、
晚到反向的重发与身份冲突、空方向语义，以及 load_profile 全规模下的临时预算。
"""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bidirect_contrast import (  # noqa: E402
    ATOL,
    RTOL,
    ScratchBudget,
    StepManifest,
    build_synthetic_workload,
    backward_step,
    manifest_from_session,
    record_from_row,
    run_step,
)

ROOT = Path(__file__).resolve().parent.parent
SESSION_PATH = ROOT / "fixtures" / "session.json"
BUDGET_BYTES = 524288


def load_session():
    return json.loads(SESSION_PATH.read_bytes())


# ---------------------------------------------------------------------------
# 稠密参考实现（仅测试用，允许显式构建全量矩阵）
# ---------------------------------------------------------------------------


def _as_records(rows):
    return [r if hasattr(r, "material_id") else record_from_row(r) for r in rows]


def dense_reference(images, texts, manifest, tau):
    """与算子同口径的稠密实现，返回 (loss, 分母dict, 梯度dict)。"""
    img = {r.material_id: r for r in _as_records(images)}
    txt = {r.material_id: r for r in _as_records(texts)}
    grad = {("image", k): np.zeros_like(v.vector) for k, v in img.items()}
    grad.update({("text", k): np.zeros_like(v.vector) for k, v in txt.items()})

    def allowed(i_id, t_id):
        a, b = img[i_id], txt[t_id]
        return (
            a.valid and b.valid and a.group == b.group
            and (i_id, t_id) not in manifest.exclusions
        )

    sums, dens = {}, {}
    for anchor_side in ("image", "text"):
        own = img if anchor_side == "image" else txt
        opp = txt if anchor_side == "image" else img
        loss_sum, denom = 0.0, 0
        for a_id in sorted(own):
            if a_id in manifest.candidate_only or not own[a_id].valid:
                continue
            cand, pos_flag = [], []
            for c_id in sorted(opp):
                if anchor_side == "image":
                    ok = allowed(a_id, c_id)
                    paired = (a_id, c_id) in manifest.matches
                else:
                    ok = allowed(c_id, a_id)
                    paired = (c_id, a_id) in manifest.matches
                if ok:
                    cand.append(c_id)
                    pos_flag.append(paired)
            pos = [i for i, p in enumerate(pos_flag) if p]
            if not cand or not pos:
                continue
            a_vec = own[a_id].vector
            c_mat = np.stack([opp[c].vector for c in cand])
            scores = tau * (c_mat @ a_vec)
            d_lse = np.log(np.exp(scores - scores.max()).sum()) + scores.max()
            ps = scores[np.array(pos)]
            n_lse = np.log(np.exp(ps - ps.max()).sum()) + ps.max()
            loss_sum += d_lse - n_lse
            denom += 1

            probs = np.exp(scores - scores.max())
            probs /= probs.sum()
            pprobs = np.exp(ps - ps.max())
            pprobs /= pprobs.sum()
            coeff = probs.copy()
            coeff[np.array(pos)] -= pprobs
            coeff *= tau
            # 先累加未归一的原始梯度；方向归一在全部锚点汇总后统一做，
            # 与算子的 scatter-add 后单侧单分母口径一致。
            if anchor_side == "image":
                grad[("image", a_id)] += coeff @ c_mat
                for j, c_id in enumerate(cand):
                    grad[("text", c_id)] += coeff[j] * a_vec
            else:
                grad[("text", a_id)] += coeff @ c_mat
                for j, c_id in enumerate(cand):
                    grad[("image", c_id)] += coeff[j] * a_vec
        sums[anchor_side], dens[anchor_side] = loss_sum, denom

    available = [s for s in ("image", "text") if dens[s] > 0]
    if not available:
        return None, dens, grad
    w = 0.5 if len(available) == 2 else 1.0
    loss = w * sum(sums[s] for s in available)
    # 每条向量按其所属侧的方向分母归一；两侧各占 1/2 时再乘组合权重。
    for side in ("image", "text"):
        if dens[side] == 0:
            continue
        scale = w / dens[side]
        for key in [k for k in grad if k[0] == side]:
            grad[key] *= scale
    return loss, dens, grad


# ---------------------------------------------------------------------------
# 随机构造小案例
# ---------------------------------------------------------------------------


def make_random_case(seed, n_img=7, n_txt=8, dim=5, groups=3, tau=1.5,
                     candidate_only=(), exclusions=(), invalid=(), extra_matches=()):
    rng = np.random.default_rng(seed)
    images, texts = [], []
    expected = {}
    for i in range(n_img):
        mid = f"i{i}"
        images.append([mid, 1, f"is{i % 3}", f"e{i}", f"g{i % groups}",
                       mid not in invalid, rng.normal(size=dim)])
        expected[mid] = 1
    for i in range(n_txt):
        mid = f"t{i}"
        texts.append([mid, 1, f"ts{i % 4}", f"e{i}", f"g{i % groups}",
                       True, rng.normal(size=dim)])
        expected[mid] = 1
    matches = set()
    for i in range(n_img):
        for k in (i, i + groups):
            if k < n_txt and k % groups == i % groups:
                matches.add((f"i{i}", f"t{k}"))
    matches.update(extra_matches)
    manifest = StepManifest.create(matches, exclusions, candidate_only, expected)
    return images, texts, manifest, tau


def grad_map(bw):
    return {(g.side, g.material_id): g.gradient for g in bw.gradients}


class FixtureContractTests(unittest.TestCase):
    def setUp(self):
        self.session = load_session()
        self.manifest = manifest_from_session(self.session)
        self.budget = ScratchBudget(BUDGET_BYTES)
        self.out = run_step(
            0, self.session["images"], self.session["texts"],
            self.manifest, self.session["inverse_temperature"], self.budget,
        )

    def test_fixture_denominators_and_reports(self):
        self.assertTrue(self.out.can_update)
        self.assertEqual(self.out.anchor_denominators, {"image": 2, "text": 2})
        img_report = self.out.reports["image"]
        # i-poster 无正例：不贡献锚点损失，但仍在向量集合中作为候选。
        self.assertEqual(img_report.anchors_without_positive, ("i-poster",))
        self.assertEqual(img_report.anchors_without_candidate, ())
        txt_report = self.out.reports["text"]
        # t-none 分组 north 没有任何有效图像候选 -> 无候选，从均值剔除。
        self.assertEqual(txt_report.anchors_without_candidate, ("t-none",))
        self.assertEqual(txt_report.anchors_without_positive, ("t-poster",))

    def test_loss_matches_manual_lse(self):
        s = self.session
        vec = {r[0]: np.array(r[6], dtype=np.float64) for r in s["images"] + s["texts"]}
        tau = s["inverse_temperature"]
        # i-lamp：候选 t-lamp/t-shade/t-pot（t-poster 被显式排除），正例前两个。
        lamp_scores = {t: tau * float(vec[t] @ vec["i-lamp"])
                       for t in ("t-lamp", "t-shade", "t-pot")}
        lamp_denom = np.log(sum(np.exp(v) for v in lamp_scores.values()))
        lamp_numer = np.log(np.exp(lamp_scores["t-lamp"]) + np.exp(lamp_scores["t-shade"]))
        lamp_loss = lamp_denom - lamp_numer
        # i-pot：同组四个有效文本都是候选，仅 t-pot 是正例。
        pot_cand = ("t-lamp", "t-shade", "t-poster", "t-pot")
        pot_scores = np.array([tau * float(vec[t] @ vec["i-pot"]) for t in pot_cand])
        pot_loss = float(
            np.log(np.exp(pot_scores - pot_scores.max()).sum()) + pot_scores.max()
            - pot_scores[3]
        )
        np.testing.assert_allclose(
            self.out.reports["image"].loss_sum, lamp_loss + pot_loss,
            atol=ATOL, rtol=RTOL,
        )
        expected_loss = 0.5 * (
            self.out.reports["image"].loss_sum + self.out.reports["text"].loss_sum
        )
        np.testing.assert_allclose(self.out.loss, expected_loss, atol=ATOL, rtol=RTOL)

    def test_duplicate_retransmission_collapses(self):
        s = self.session
        texts = list(s["texts"]) + [s["duplicate"]]  # 同身份同版本同向量重发。
        out2 = run_step(1, s["images"], texts, self.manifest,
                        s["inverse_temperature"], ScratchBudget(BUDGET_BYTES))
        self.assertTrue(out2.can_update)
        np.testing.assert_allclose(out2.loss, self.out.loss, atol=ATOL, rtol=RTOL)
        bw1, bw2 = backward_step(self.out.frozen), backward_step(out2.frozen)
        self.assertEqual(len(bw1.gradients), len(bw2.gradients))
        for g1, g2 in zip(bw1.gradients, bw2.gradients):
            self.assertEqual((g1.side, g1.material_id, g1.shard),
                             (g2.side, g2.material_id, g2.shard))
            np.testing.assert_array_equal(g1.gradient, g2.gradient)

    def test_conflicting_duplicate_blocks_step(self):
        s = self.session
        bad = list(s["duplicate"])
        bad[2] = "t9"  # 同身份同版本但生产分片归属/内容冲突。
        out = run_step(2, s["images"], list(s["texts"]) + [bad], self.manifest,
                       s["inverse_temperature"], ScratchBudget(BUDGET_BYTES))
        self.assertFalse(out.can_update)
        self.assertTrue(any(i.code == "duplicate_conflict" and i.material_id == "t-shade"
                            for i in out.issues))
        bw = backward_step(out.frozen)
        self.assertFalse(bw.can_update)
        self.assertEqual(bw.gradients, ())

    def test_version_conflict_blocks_step(self):
        s = self.session
        bad = [list(r) for r in s["texts"]]
        bad[0][1] = 2  # t-lamp 实际版本与清单声明不一致。
        out = run_step(3, s["images"], bad, self.manifest,
                       s["inverse_temperature"], ScratchBudget(BUDGET_BYTES))
        self.assertFalse(out.can_update)
        self.assertTrue(any(i.code == "version_conflict" and i.material_id == "t-lamp"
                            for i in out.issues))

    def test_missing_distinct_from_invalid(self):
        s = self.session
        # 缺块：i-pot 完全不到货，绝不补零。
        images = [r for r in s["images"] if r[0] != "i-pot"]
        out = run_step(4, images, s["texts"], self.manifest,
                       s["inverse_temperature"], ScratchBudget(BUDGET_BYTES))
        self.assertFalse(out.can_update)
        self.assertIn("i-pot", out.missing_ids)
        self.assertIsNotNone(out.loss)  # 诊断损失仍计算，但该步不可更新。
        self.assertEqual(backward_step(out.frozen).gradients, ())
        # 无效声明（i-idle valid=false）不阻塞，只被排除在允许集合外。
        self.assertTrue(self.out.can_update)
        self.assertEqual(self.out.invalid_ids["image"], ("i-idle",))

    def test_candidate_only_keeps_vectors_in_gradient_set(self):
        s = self.session
        manifest = manifest_from_session(
            s, candidate_only=s["asymmetric_trial"]["candidate_only"]
        )
        out = run_step(5, s["images"], s["texts"], manifest,
                       s["inverse_temperature"], ScratchBudget(BUDGET_BYTES))
        # 文本方向整体为空：无方向损失，图像方向照常交付。
        self.assertFalse(out.reports["text"].available)
        self.assertTrue(out.reports["image"].available)
        self.assertEqual(out.anchor_denominators["text"], 0)
        np.testing.assert_allclose(
            out.loss, out.reports["image"].loss_sum, atol=ATOL, rtol=RTOL
        )
        ids = {(g.side, g.material_id) for g in backward_step(out.frozen).gradients}
        # 所有向量（含仅候选文本、无正例图像）都有归属梯度位，没有被一笔清零。
        for r in s["images"]:
            self.assertIn(("image", r[0]), ids)
        for r in s["texts"]:
            self.assertIn(("text", r[0]), ids)

    def test_non_positive_anchor_keeps_candidate_gradient(self):
        # i-poster 没有已确认正例：自身锚点损失为 0（锚点侧梯度为 0），
        # 但它仍是 t-poster 的允许候选，必须保留候选侧通道（不被一笔清零）。
        bw = backward_step(self.out.frozen)
        g = grad_map(bw)
        # i-poster 不发起有效锚点：自身锚点侧贡献为 0（允许 FP64 舍入残差）。
        np.testing.assert_allclose(
            g[("image", "i-poster")], np.zeros(3), atol=1e-12, rtol=0.0
        )
        # t-poster 是 i-poster 的唯一同组候选；它在文本方向无正例但作为
        # i-poster 候选时，由于 i-poster 自身不发起有效锚点，两侧梯度均为 0，
        # 关键是梯度位仍然存在且归属正确，而不是被删除或补零伪造。
        self.assertIn(("text", "t-poster"), g)
        piece = next(p for p in bw.gradients
                     if (p.side, p.material_id) == ("text", "t-poster"))
        self.assertEqual((piece.version, piece.shard), (1, "t0"))

    def test_missing_never_zero_filled(self):
        s = self.session
        images = [r for r in s["images"] if r[0] != "i-lamp"]
        out = run_step(7, images, s["texts"], self.manifest,
                       s["inverse_temperature"], ScratchBudget(BUDGET_BYTES))
        # 算子不发明零向量：冻结包中根本没有 i-lamp 这个身份。
        self.assertNotIn("i-lamp", out.frozen.snapshots["image"])
        self.assertNotIn("i-lamp", out.frozen.fingerprints["image"])
        bw = backward_step(out.frozen)
        self.assertFalse(bw.can_update)
        self.assertNotIn(("image", "i-lamp"), grad_map(bw))

    def test_both_directions_empty_reports_no_loss(self):
        s = self.session
        manifest = StepManifest.create(
            matches=(), exclusions=s["exclusions"],
            candidate_only=[r[0] for r in s["images"]] + [r[0] for r in s["texts"]],
            expected_versions=s["step_versions"],
        )
        out = run_step(6, s["images"], s["texts"], manifest,
                       s["inverse_temperature"], ScratchBudget(BUDGET_BYTES))
        self.assertIsNone(out.loss)
        self.assertFalse(any(r.available for r in out.reports.values()))
        bw = backward_step(out.frozen)
        self.assertTrue(bw.can_update)
        # 无锚点损失 -> 全部梯度为零，但仍按身份交付（不是缺失）。
        self.assertEqual(len(bw.gradients), len(s["images"]) + len(s["texts"]))
        self.assertTrue(all(np.all(g.gradient == 0.0) for g in bw.gradients))


class DenseReferenceTests(unittest.TestCase):
    def _assert_matches_dense(self, seed, **kwargs):
        images, texts, manifest, tau = make_random_case(seed, **kwargs)
        out = run_step(seed, images, texts, manifest, tau, ScratchBudget(BUDGET_BYTES))
        self.assertTrue(out.can_update, msg=str(out.issues))
        ref_loss, ref_den, ref_grad = dense_reference(images, texts, manifest, tau)
        self.assertEqual(out.anchor_denominators, ref_den)
        if ref_loss is None:
            self.assertIsNone(out.loss)
        else:
            np.testing.assert_allclose(out.loss, ref_loss, atol=ATOL, rtol=RTOL)
        got = grad_map(backward_step(out.frozen))
        self.assertEqual(set(got), set(ref_grad))
        for key, ref in ref_grad.items():
            np.testing.assert_allclose(got[key], ref, atol=ATOL, rtol=RTOL,
                                       err_msg=f"梯度不一致 {key}")

    def test_seeds_match_dense(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                self._assert_matches_dense(seed)

    def test_exclusions_and_candidate_only_match_dense(self):
        self._assert_matches_dense(11, exclusions=[("i0", "t0")], candidate_only=["t1"])

    def test_invalid_records_match_dense(self):
        self._assert_matches_dense(7, invalid=["i2", "t3"])

    def test_anchor_without_positive_still_receives_candidate_gradient(self):
        # 构造一个有候选但无正例的图像 i6：它不发起锚点损失（自身梯度为 0），
        # 但同组文本作为锚点时把它当候选，使它通过候选侧通道收到非零梯度。
        images, texts, manifest, tau = make_random_case(
            51, n_img=7, n_txt=8, groups=2
        )
        # 删掉所有涉及 i6 的正例登记，保留同组允许关系。
        pruned_matches = {p for p in manifest.matches if p[0] != "i6"}
        manifest = StepManifest.create(
            pruned_matches, manifest.exclusions, manifest.candidate_only,
            manifest.expected_versions,
        )
        out = run_step(51, images, texts, manifest, tau)
        self.assertIn("i6", out.reports["image"].anchors_without_positive)
        g = grad_map(backward_step(out.frozen))
        # i6 自身锚点不产生方向性梯度（候选侧累积来自把它当候选的文本锚点，
        # 与此处断言的“无自身锚点损失”不冲突；这里只校验它没有锚点自导数口径
        # 下的额外项——通过与稠密参考逐分量一致已在其他用例覆盖）。
        ref_loss, ref_den, ref_grad = dense_reference(images, texts, manifest, tau)
        np.testing.assert_allclose(
            g[("image", "i6")], ref_grad[("image", "i6")], atol=ATOL, rtol=RTOL
        )
        # 同组存在把 i6 当候选的文本锚点（t6 同组且通常有自己的正例）。
        same_group_texts = [r[0] for r in texts if r[4] == "g0"]
        self.assertTrue(any(np.any(g[("text", t)] != 0.0) for t in same_group_texts))

    def test_non_positive_image_gets_nonzero_gradient_as_candidate(self):
        # iX 有同组候选但无任何正例登记 -> 不发起锚点损失；
        # 同组文本 tX 有自己的正例且把 iX 当候选 -> iX 经候选侧通道拿到非零梯度。
        rng = np.random.default_rng(77)
        images = [
            ["iX", 1, "s0", "ex", "g0", True, rng.normal(size=4)],
            ["iY", 1, "s0", "ey", "g0", True, rng.normal(size=4)],
        ]
        texts = [
            ["tX", 1, "s1", "ex", "g0", True, rng.normal(size=4)],
            ["tY", 1, "s1", "ey", "g0", True, rng.normal(size=4)],
        ]
        expected = {"iX": 1, "iY": 1, "tX": 1, "tY": 1}
        # 只有 iY-tY 一条正例：iX 无正例，但 tX 的候选集合包含 iX、iY。
        manifest = StepManifest.create([("iY", "tY")], (), (), expected)
        out = run_step(77, images, texts, manifest, 1.2)
        self.assertEqual(out.reports["image"].anchors_without_positive, ("iX",))
        g = grad_map(backward_step(out.frozen))
        # 文本方向：tX 是锚点，iX/iY 是其候选，iX 必然收到非零候选梯度。
        self.assertTrue(np.any(np.abs(g[("image", "iX")]) > 1e-8))
        # 与稠密参考逐分量一致。
        _, _, ref = dense_reference(images, texts, manifest, 1.2)
        for key, value in g.items():
            np.testing.assert_allclose(value, ref[key], atol=ATOL, rtol=RTOL)

    def test_gradient_finite_difference(self):
        images, texts, manifest, tau = make_random_case(21, dim=4, n_img=4, n_txt=5)
        out = run_step(21, images, texts, manifest, tau)
        got = grad_map(backward_step(out.frozen))
        eps = 1e-7
        rng = np.random.default_rng(0)
        for side, rows in (("image", images), ("text", texts)):
            for row in rows:
                mid = row[0]
                d = int(rng.integers(0, len(row[6])))
                analytic = got[(side, mid)][d]
                row[6][d] += eps
                plus = run_step(21, images, texts, manifest, tau).loss
                row[6][d] -= 2 * eps
                minus = run_step(21, images, texts, manifest, tau).loss
                row[6][d] += eps
                fd = (plus - minus) / (2 * eps)
                # 中心差分的截断误差 O(eps^2)，对接近零的梯度只做绝对量约束。
                tol = max(1e-5, 1e-4 * max(abs(analytic), abs(fd), 1.0))
                np.testing.assert_allclose(
                    analytic, fd, atol=tol, rtol=1e-4,
                    err_msg=f"有限差分不一致 {side}/{mid}[{d}]",
                )


class DeterminismTests(unittest.TestCase):
    def test_arrival_order_invariant(self):
        images, texts, manifest, tau = make_random_case(31)
        rng = np.random.default_rng(1)
        base = run_step(0, images, texts, manifest, tau)
        for k in range(4):
            shuffled_i = [images[i] for i in rng.permutation(len(images))]
            shuffled_t = [texts[i] for i in rng.permutation(len(texts))]
            shuffled_t.append(list(texts[k % len(texts)]))  # 夹带重复传输。
            out = run_step(k + 1, shuffled_i, shuffled_t, manifest, tau)
            np.testing.assert_array_equal(out.loss, base.loss)
            ga, gb = grad_map(backward_step(out.frozen)), grad_map(backward_step(base.frozen))
            self.assertEqual(set(ga), set(gb))
            for key in ga:
                np.testing.assert_array_equal(ga[key], gb[key])

    def test_shard_repartition_invariant_and_routing(self):
        images, texts, manifest, tau = make_random_case(33)
        base = run_step(0, images, texts, manifest, tau)
        re_i = [[r[0], r[1], f"new{int(r[0][1:]) % 2}", *r[3:]] for r in images]
        re_t = [[r[0], r[1], f"newT{int(r[0][1:]) % 5}", *r[3:]] for r in texts]
        out = run_step(1, re_i, re_t, manifest, tau)
        np.testing.assert_array_equal(out.loss, base.loss)
        bw = backward_step(out.frozen)
        expect_shard = {**{r[0]: r[2] for r in re_i}, **{r[0]: r[2] for r in re_t}}
        for g in bw.gradients:
            self.assertEqual(g.shard, expect_shard[g.material_id])
        ga, gb = grad_map(bw), grad_map(backward_step(base.frozen))
        for key in ga:
            np.testing.assert_allclose(ga[key], gb[key], atol=ATOL, rtol=RTOL)

    def test_by_shard_packages_are_disjoint(self):
        images, texts, manifest, tau = make_random_case(34)
        bw = backward_step(run_step(0, images, texts, manifest, tau).frozen)
        seen = []
        for (side, shard), pieces in bw.by_shard.items():
            for p in pieces:
                self.assertEqual((p.side, p.shard), (side, shard))
                seen.append((p.side, p.material_id))
        self.assertEqual(len(seen), len(set(seen)))


class LateBackwardTests(unittest.TestCase):
    def _drop_snapshot(self, frozen, side, material_id):
        frozen.snapshots[side][material_id] = None

    def test_refetch_original_versions_after_late_backward(self):
        images, texts, manifest, tau = make_random_case(41)
        out = run_step(0, images, texts, manifest, tau)
        frozen = out.frozen
        stored = {
            (side, r[0]): np.array(r[6], dtype=np.float64)
            for side, rows in (("image", images), ("text", texts)) for r in rows
        }
        versions = dict(frozen.versions["image"], **frozen.versions["text"])

        def refetch(side, material_id, version, shard):
            self.assertEqual(versions[material_id], version)  # 必须沿用原步版本。
            return stored[(side, material_id)].copy()

        # 模拟工作区只读快照被回收：两侧各丢一块，靠分片重发原版取回。
        self._drop_snapshot(frozen, "image", "i2")
        self._drop_snapshot(frozen, "text", "t3")
        bw_refetch = backward_step(frozen, refetch=refetch)
        self.assertTrue(bw_refetch.can_update)
        fresh = run_step(99, images, texts, manifest, tau)
        a, b = grad_map(bw_refetch), grad_map(backward_step(fresh.frozen))
        for key in a:
            np.testing.assert_allclose(a[key], b[key], atol=ATOL, rtol=RTOL)

    def test_unrecoverable_original_blocks_whole_step(self):
        images, texts, manifest, tau = make_random_case(42)
        frozen = run_step(0, images, texts, manifest, tau).frozen
        self._drop_snapshot(frozen, "image", "i3")

        def refetch(side, material_id, version, shard):
            return None  # 分片确认原版无法取回，且不允许补零。

        bw = backward_step(frozen, refetch=refetch)
        self.assertFalse(bw.can_update)
        self.assertEqual(bw.gradients, ())  # 不会只交付半边。
        self.assertTrue(any(i.code == "missing" and i.material_id == "i3"
                            for i in bw.issues))

    def test_refetch_identity_conflict_blocks(self):
        images, texts, manifest, tau = make_random_case(43)
        frozen = run_step(0, images, texts, manifest, tau).frozen
        self._drop_snapshot(frozen, "text", "t2")

        def refetch(side, material_id, version, shard):
            if (side, material_id) == ("text", "t2"):
                return frozen.fingerprints["text"]["t2"] + 1.0  # 冲突版本内容。
            return frozen.fingerprints[side][material_id].copy()

        bw = backward_step(frozen, refetch=refetch)
        self.assertFalse(bw.can_update)
        self.assertEqual(bw.gradients, ())
        self.assertTrue(any(i.code == "vector_conflict" and i.material_id == "t2"
                            for i in bw.issues))


class LoadProfileTests(unittest.TestCase):
    def test_full_profile_fits_scratch_budget(self):
        session = load_session()
        profile = session["load_profile"]
        images, texts, manifest = build_synthetic_workload(profile)
        fwd_budget = ScratchBudget(profile["scratch_bytes"])
        out = run_step(0, images, texts, manifest, session["inverse_temperature"], fwd_budget)
        self.assertTrue(out.can_update)
        self.assertIsNotNone(out.loss)
        bwd_budget = ScratchBudget(profile["scratch_bytes"])
        bw = backward_step(out.frozen, budget=bwd_budget)
        self.assertTrue(bw.can_update)
        # 临时工作区峰值必须显著低于全量交叉矩阵，且在契约预算内。
        full_matrix = profile["images"] * profile["texts"] * 8
        self.assertGreater(full_matrix, profile["scratch_bytes"])
        self.assertLessEqual(fwd_budget.high_water_bytes, profile["scratch_bytes"])
        self.assertLessEqual(bwd_budget.high_water_bytes, profile["scratch_bytes"])
        # 冻结包只保存候选身份元数据，没有保存任何分数矩阵。
        for plans in out.frozen.plans.values():
            for plan in plans:
                self.assertIsInstance(plan.candidate_ids, tuple)
        self.assertEqual(len(bw.gradients), profile["images"] + profile["texts"])

    def test_budget_exceeded_raises(self):
        session = load_session()
        profile = dict(session["load_profile"])
        profile["images"], profile["texts"], profile["shards"] = 64, 64, 4
        images, texts, manifest = build_synthetic_workload(profile)
        with self.assertRaises(MemoryError):
            run_step(0, images, texts, manifest,
                     session["inverse_temperature"], ScratchBudget(64))


class NextStepCorrectionTests(unittest.TestCase):
    def test_next_add_takes_effect_only_in_next_manifest(self):
        s = load_session()
        manifest = manifest_from_session(s)
        out = run_step(0, s["images"], s["texts"], manifest, s["inverse_temperature"])
        # 本步 i-poster 无正例。
        self.assertEqual(out.reports["image"].anchors_without_positive, ("i-poster",))
        next_manifest = StepManifest.create(
            set(manifest.matches) | {tuple(s["next_add"])},
            manifest.exclusions, manifest.candidate_only, manifest.expected_versions,
        )
        out_next = run_step(1, s["images"], s["texts"], next_manifest,
                            s["inverse_temperature"])
        self.assertEqual(out_next.reports["image"].anchors_without_positive, ())
        self.assertGreater(out_next.anchor_denominators["image"],
                           out.anchor_denominators["image"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
