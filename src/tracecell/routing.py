"""文档级路由：这份文件要不要人看？人看的话看哪儿？

粒度降到文档层，问题的形式就变了：不再是"这一格可不可信"，而是
"这份文档能不能免检、要不要抽检、哪些格子必须人工过目"。

TraceCell 给出的结论是三档：

* **免检（auto_pass）**：没有必须复核项，抽检也检不出有意义的东西。
* **抽检（spot_check）**：整体可信，但按抽样验收的算法抽一部分看。
* **全检（full_review）**：高危字段大面积缺证据，或者候选之间大面积打架。

抽检的样本量不是拍脑袋定的。设单件出错概率为 p，抽取 m 件独立样本，
至少发现一处错误的概率是 1-(1-p)^m。要求这个概率不低于 1-δ，解出

    m ≥ ln(δ) / ln(1-p)

这个式子决定了"抽多少才够"，抽检因此是个有统计依据的动作。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .risk import expected_loss, region_confidence, region_is_contested
from .types import FieldValue, Region, Severity


@dataclass
class SamplingPlan:
    """随机抽检计划。"""

    population: int
    sample_size: int
    estimated_error_rate: float
    detection_probability: float
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "population": self.population,
            "sample_size": self.sample_size,
            "estimated_error_rate": round(self.estimated_error_rate, 4),
            "detection_probability": round(self.detection_probability, 4),
            "rationale": self.rationale,
        }


@dataclass
class RoutingResult:
    """路由结论。"""

    verdict: str
    must_review: list[str] = field(default_factory=list)
    sampling: SamplingPlan | None = None
    reasons: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "must_review": self.must_review,
            "sampling": self.sampling.to_dict() if self.sampling else None,
            "reasons": self.reasons,
            "stats": self.stats,
        }


def required_sample_size(error_rate: float, detection_probability: float = 0.95) -> int:
    """达到目标检出概率所需的样本量。

    m ≥ ln(δ) / ln(1-p)，其中 δ = 1 - 目标检出概率。
    p 接近 0 时公式退化，直接返回 1（至少要看一件）。
    """

    if error_rate <= 0:
        return 0
    delta = 1.0 - detection_probability
    if delta <= 0:
        return 1
    denominator = math.log(1.0 - error_rate)
    if denominator == 0:
        return 1
    size = math.ceil(math.log(delta) / denominator)
    return max(1, size)


def detection_probability(error_rate: float, sample_size: int) -> float:
    """抽 n 件至少发现一处错误的概率。"""

    if sample_size <= 0:
        return 0.0
    return 1.0 - (1.0 - error_rate) ** sample_size


def route_document(
    fields: list[FieldValue],
    regions: dict[str, Region],
    policy,
    target_detection: float = 0.95,
) -> RoutingResult:
    """给整份文档定路由结论。"""

    reasons: list[str] = []
    must_review: list[tuple[str, float]] = []

    # 1) 定点复核：所有期望损失超过容忍度的字段所依赖的区域。
    tolerance = policy.residual_tolerance
    for fld in fields:
        loss = expected_loss(fld.confidence, fld.severity)
        if loss > tolerance:
            must_review.extend((rid, loss) for rid in fld.region_ids)

    # 2) 候选打架的区域，无论字段等级，一律要看。
    contested = [rid for rid, region in regions.items() if region_is_contested(region)]
    for rid in contested:
        must_review.append((rid, 1.0))

    # 3) 没有证据支撑的字段也要看。
    orphan = [fld for fld in fields if not fld.region_ids]
    for fld in orphan:
        must_review.append((f"field:{fld.name}", expected_loss(fld.confidence, fld.severity)))

    # 按期望损失排序，同分按区域号排，保证结果稳定可复现。
    must_review.sort(key=lambda item: (-item[1], item[0]))
    queue = [rid for rid, _ in must_review]
    seen: set[str] = set()
    ordered_queue = []
    for rid in queue:
        if rid not in seen:
            seen.add(rid)
            ordered_queue.append(rid)

    # 4) 估算整体出错率：按区域置信度加权平均。
    total_weight = 0.0
    total_error = 0.0
    for region in regions.values():
        weight = max(1, sum(1 for f in fields if region.id in f.region_ids))
        total_weight += weight
        total_error += weight * (1.0 - region_confidence(region))
    error_rate = (total_error / total_weight) if total_weight else 0.0

    population = len(regions)
    sampling: SamplingPlan | None = None
    if error_rate > 0 and population > 0:
        size = min(population, required_sample_size(error_rate, target_detection))
        sampling = SamplingPlan(
            population=population,
            sample_size=size,
            estimated_error_rate=error_rate,
            detection_probability=detection_probability(error_rate, size),
            rationale=(
                f"按单件出错率 {error_rate:.1%} 估算，抽取 {size}/{population} 件"
                f"时至少发现一处错误的概率约 {detection_probability(error_rate, size):.1%}"
            ),
        )

    # 5) 定档。
    critical_open = [
        f
        for f in fields
        if f.severity is Severity.CRITICAL and expected_loss(f.confidence, f.severity) > tolerance
    ]
    orphan_critical = [f for f in orphan if f.severity is Severity.CRITICAL]
    if orphan_critical:
        # 高危字段连原文都没找到，说明抽取这一环就出了问题，整份台账不可信。
        verdict = "full_review"
        reasons.append(
            f"{len(orphan_critical)} 个高危字段没有找到对应的原文区域"
            f"（{'、'.join(f.name for f in orphan_critical[:3])}），台账不完整，必须全检"
        )
    elif not ordered_queue and population:
        verdict = "auto_pass"
        reasons.append("所有字段的期望损失都在容忍度以内，且没有候选冲突，免检通过")
    elif len(ordered_queue) > max(3, population * 0.2) or len(critical_open) > 2:
        verdict = "full_review"
        reasons.append(
            f"需要定点复核 {len(ordered_queue)} 处（占全文 {len(ordered_queue) / max(population, 1):.0%}），"
            f"其中高危字段 {len(critical_open)} 个，建议全检"
        )
    else:
        verdict = "spot_check"
        reasons.append(
            f"仅 {len(ordered_queue)} 处需要定点复核，其余按抽样验收处理"
        )

    stats = {
        "population": population,
        "field_count": len(fields),
        "contested_regions": len(contested),
        "critical_open": len(critical_open),
        "estimated_error_rate": round(error_rate, 4),
    }
    return RoutingResult(
        verdict=verdict,
        must_review=ordered_queue,
        sampling=sampling,
        reasons=reasons,
        stats=stats,
    )


def render_review_checklist(
    regions: dict[str, Region],
    fields: list[FieldValue],
    queue: list[str],
) -> list[dict[str, Any]]:
    """生成给复核人员看的清单。

    每一条都写清楚"看什么""为什么要看""怀疑哪几个值"，让人不用回看
    整份文档就能开始干活。
    """

    checklist: list[dict[str, Any]] = []
    for rid in queue:
        region = regions.get(rid)
        if region is None:
            related = [f for f in fields if f.name == rid.split(":", 1)[-1]]
            for fld in related:
                checklist.append(
                    {
                        "target": fld.name,
                        "why": f"字段 {fld.name} 没有任何原文区域支撑，无法核对",
                        "candidates": [],
                    }
                )
            continue

        related_fields = [f for f in fields if region.id in f.region_ids]
        candidates = [
            {
                "text": cand.text,
                "score": cand.score,
                "origin": cand.origin,
                "verdict": cand.verdict.value,
            }
            for cand in region.candidates
        ]
        contested = region_is_contested(region)
        checklist.append(
            {
                "target": region.id,
                "page": region.page,
                "bbox": {
                    "x0": region.bbox.x0,
                    "y0": region.bbox.y0,
                    "x1": region.bbox.x1,
                    "y1": region.bbox.y1,
                },
                "why": (
                    "多个候选结果不一致，需要人工确认"
                    if contested
                    else "支撑字段的期望损失超过容忍度："
                    + "、".join(f"{f.name}={f.value}" for f in related_fields)
                ),
                "fields": [
                    {"name": f.name, "value": f.value, "confidence": round(f.confidence, 4)}
                    for f in related_fields
                ],
                "candidates": candidates,
            }
        )
    return checklist
