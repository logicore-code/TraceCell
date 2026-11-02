"""阈值标定。

Jev 给的是 0~1 的分数，但这个分数**不是**"识别正确的概率"。官方模型卡
反复提醒：AUROC 93.6 不等于 93.6% 的正确率，更不等于自动通过的部分
有 93.6% 是对的。

要把它变成一个能上生产的闸门，唯一可靠的办法是拿自己业务里的数据标
一次。本模块提供三样东西：

* **AUROC**：衡量分数能不能把对错分开（秩和实现，不依赖 sklearn）。
* **保序回归（PAVA）**：把分数校准成概率，回答"0.8 分到底对不对的概率
  是多少"。
* **阈值搜索**：在"自动通过准确率不低于目标"的前提下，最大化自动通过率，
  并给出对应的免检率与漏检规模。

用法：

    samples = load_labelled_samples("samples.json")
    result = calibrate(samples, target_accuracy=0.995)
    print(result.threshold, result.accept_rate, result.accept_accuracy)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class CalibrationSample:
    """一条人工标注过的样本。"""

    score: float
    correct: bool
    category: str = "text"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "correct": self.correct,
            "category": self.category,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CalibrationSample":
        return cls(
            score=float(data["score"]),
            correct=bool(data["correct"]),
            category=data.get("category", "text"),
            note=data.get("note", ""),
        )


@dataclass
class ThresholdResult:
    """标定结果。"""

    threshold: float
    accept_rate: float
    accept_accuracy: float
    accepted: int
    rejected: int
    false_accepts: int
    auroc: float
    total: int
    note: str = ""
    curve: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 4),
            "accept_rate": round(self.accept_rate, 4),
            "accept_accuracy": round(self.accept_accuracy, 4),
            "accepted": self.accepted,
            "rejected": self.rejected,
            "false_accepts": self.false_accepts,
            "auroc": round(self.auroc, 4),
            "total": self.total,
            "note": self.note,
            "curve": self.curve,
        }


def load_samples(path: str | Path) -> list[CalibrationSample]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [CalibrationSample.from_dict(item) for item in data]


def auroc(samples: Sequence[CalibrationSample]) -> float:
    """AUROC：用秩和算，正确处理并列。

    分数越高越可能正确。返回值是随机取一个正样本和一个负样本时，正样本
    得分更高的概率。
    """

    positives = [s.score for s in samples if s.correct]
    negatives = [s.score for s in samples if not s.correct]
    n_pos, n_neg = len(positives), len(negatives)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # 并列取平均秩
    ordered = sorted([(s.score, s.correct) for s in samples], key=lambda item: item[0])
    ranks: list[float] = [0.0] * len(ordered)
    index = 0
    while index < len(ordered):
        end = index
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[index][0]:
            end += 1
        average_rank = (index + end) / 2 + 1
        for position in range(index, end + 1):
            ranks[position] = average_rank
        index = end + 1

    rank_sum_positive = sum(rank for rank, (_, correct) in zip(ranks, ordered) if correct)
    return (rank_sum_positive - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def isotonic_regression(samples: Sequence[CalibrationSample]) -> tuple[list[float], list[float]]:
    """保序回归，把分数映射成校准后的概率。

    返回 (分数断点, 对应概率)。用池相邻违规者算法（PAVA）：先按分数排序，
    逐段合并违反单调性的相邻块，直到剩余概率序列单调不减。

    校准后的概率回答了"Jev 给了 0.8 分的样本，实际正确的比例是多少"。
    """

    if not samples:
        return [], []

    ordered = sorted(samples, key=lambda s: s.score)
    xs = [s.score for s in ordered]
    ys = [1.0 if s.correct else 0.0 for s in ordered]
    weights = [1.0] * len(ys)

    # 每一块：[值和, 权重和, 起点, 终点]
    blocks: list[list[float]] = []
    for value, weight in zip(ys, weights):
        blocks.append([value, weight, len(blocks), len(blocks)])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            right = blocks.pop()
            left = blocks.pop()
            merged = [
                left[0] + right[0],
                left[1] + right[1],
                left[2],
                right[3],
            ]
            blocks.append(merged)

    knot_x: list[float] = []
    knot_y: list[float] = []
    for start, end, value_sum, weight_sum in [(b[2], b[3], b[0], b[1]) for b in blocks]:
        probability = value_sum / weight_sum if weight_sum else 0.0
        for position in range(start, end + 1):
            knot_x.append(xs[position])
            knot_y.append(probability)

    # 折叠成断点
    folded_x: list[float] = []
    folded_y: list[float] = []
    for x, y in zip(knot_x, knot_y):
        if folded_y and abs(folded_y[-1] - y) < 1e-12:
            continue
        folded_x.append(x)
        folded_y.append(y)
    return folded_x, folded_y


def apply_isotonic(knot_x: list[float], knot_y: list[float], score: float) -> float:
    """用保序回归的结果把一个分数换算成校准概率。"""

    if not knot_x:
        return score
    if score <= knot_x[0]:
        return knot_y[0]
    if score >= knot_x[-1]:
        return knot_y[-1]
    for index in range(1, len(knot_x)):
        if score <= knot_x[index]:
            span = knot_x[index] - knot_x[index - 1]
            if span <= 0:
                return knot_y[index]
            ratio = (score - knot_x[index - 1]) / span
            return knot_y[index - 1] + ratio * (knot_y[index] - knot_y[index - 1])
    return knot_y[-1]


def threshold_curve(
    samples: Sequence[CalibrationSample], grid: Iterable[float] | None = None
) -> list[dict[str, Any]]:
    """扫描阈值，画出"免检率 vs 免检准确率"的曲线。"""

    if not samples:
        return []
    if grid is None:
        grid = [round(value / 100, 2) for value in range(0, 101, 1)]
    curve: list[dict[str, Any]] = []
    total = len(samples)
    for threshold in grid:
        accepted = [s for s in samples if s.score >= threshold]
        if not accepted:
            continue
        correct = sum(1 for s in accepted if s.correct)
        curve.append(
            {
                "threshold": threshold,
                "accepted": len(accepted),
                "accept_rate": len(accepted) / total,
                "accept_accuracy": correct / len(accepted),
                "false_accepts": len(accepted) - correct,
                "missed": total - len(accepted),
            }
        )
    return curve


def calibrate(
    samples: Sequence[CalibrationSample],
    target_accuracy: float = 0.995,
    detection_target: float = 0.95,
) -> ThresholdResult:
    """在"自动通过准确率 ≥ target_accuracy"的前提下找最优阈值。

    目标准确率定高一点是有意的：漏掉一处高危字段的代价，通常远大于多
    送几处去人工。返回值里的 accept_rate 告诉你这个闸门能省多少人工。
    """

    if not samples:
        return ThresholdResult(
            threshold=1.0,
            accept_rate=0.0,
            accept_accuracy=0.0,
            accepted=0,
            rejected=0,
            false_accepts=0,
            auroc=float("nan"),
            total=0,
            note="没有样本，无法标定",
        )

    curve = threshold_curve(samples)
    total = len(samples)
    qualifying = [point for point in curve if point["accept_accuracy"] >= target_accuracy]
    note = ""

    if qualifying:
        # 在满足准确率约束的点里选免检率最高的。
        best = max(qualifying, key=lambda p: (p["accept_rate"], p["threshold"]))
        threshold = best["threshold"]
    else:
        # 达不到目标，就把最严的阈值用上（全部转人工）。
        strictest = max(curve, key=lambda p: p["threshold"])
        threshold = strictest["threshold"]
        note = (
            f"在 {len(samples)} 条样本上无法同时满足自动通过准确率 ≥ {target_accuracy:.3%}，"
            f"已退化为最严格阈值，建议补充标注样本或调整目标"
        )

    accepted = [s for s in samples if s.score >= threshold]
    correct = sum(1 for s in accepted if s.correct)
    return ThresholdResult(
        threshold=threshold,
        accept_rate=len(accepted) / total,
        accept_accuracy=correct / len(accepted) if accepted else 0.0,
        accepted=len(accepted),
        rejected=total - len(accepted),
        false_accepts=len(accepted) - correct,
        auroc=auroc(samples),
        total=total,
        note=note,
        curve=curve,
    )


def per_category_stats(samples: Sequence[CalibrationSample]) -> dict[str, dict[str, Any]]:
    """按 text / formula / table 分别统计。

    官方模型卡里三类区域的错误形态差别很大（错字、上下标、行列错位），
    分开看才有意义。
    """

    buckets: dict[str, list[CalibrationSample]] = {}
    for sample in samples:
        buckets.setdefault(sample.category, []).append(sample)
    stats: dict[str, dict[str, Any]] = {}
    for category, group in buckets.items():
        accepted = [s for s in group if s.score >= 0.5]
        correct = sum(1 for s in accepted if s.correct)
        stats[category] = {
            "count": len(group),
            "mean_score": sum(s.score for s in group) / len(group),
            "auroc": auroc(group),
            "accept_rate_at_0.5": len(accepted) / len(group),
            "accept_accuracy_at_0.5": correct / len(accepted) if accepted else None,
        }
    return stats


def recommend_threshold(
    samples: Sequence[CalibrationSample],
    human_review_cost: float,
    error_cost: float,
    candidate_grid: Iterable[float] | None = None,
) -> dict[str, Any]:
    """按成本模型推荐阈值。

    把人工复核的成本摊进来：阈值定低一点，多自动通过一些，但每个错误放行
    的代价是 error_cost；阈值定高一点，人工量上升，每个复核的代价是
    human_review_cost。取总期望成本最小的那个阈值。
    """

    curve = threshold_curve(samples, candidate_grid)
    if not curve:
        return {}

    total = len(samples)
    best: dict[str, Any] | None = None
    for point in curve:
        # 放行的错误要付 error_cost；转人工的数量乘以人工复核成本。
        expected_cost = (
            point["false_accepts"] * error_cost
            + point["missed"] * human_review_cost
        ) / total
        candidate = {
            "threshold": point["threshold"],
            "accept_rate": point["accept_rate"],
            "accept_accuracy": point["accept_accuracy"],
            "expected_cost": expected_cost,
        }
        if best is None or candidate["expected_cost"] < best["expected_cost"]:
            best = candidate
    return best or {}
