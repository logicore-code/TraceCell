"""置信度与风险。

这一层回答两个问题：

1. 某一格（区域）现在有多可信？
2. 由它支撑的业务字段有多可信，整份文档的残余风险有多大？

字段置信度采用**悲观继承**：字段的置信度等于支撑它的所有区域里最低的
那一个。理由很直接——合同金额如果是从两个区域拼出来的，其中一处认错
了，这个字段就是错的，取平均等于把风险藏起来。

候选之间打架时（重解析给出了不同结果）置信度会被打折，因为分歧本身
就是不确定性的证据。
"""

from __future__ import annotations

from .textnorm import normalize
from .types import SEVERITY_WEIGHT, FieldValue, Region, Severity

# 没有评分时给的保守置信度。既不是 0（那会让所有区域都进人工队列），
# 也不是 1（那是自欺欺人）。
UNSCORED_FALLBACK = 0.30

# 多个候选互相矛盾时的折扣。
DISAGREEMENT_FACTOR = 0.65


def scored_candidates(region: Region) -> list:
    return [c for c in region.candidates if c.scored]


def candidates_agree(region: Region) -> bool | None:
    """已评分候选之间是否一致。

    返回 None 表示候选不足两个，无从判断——这和"确认一致"是两回事，
    调用方必须区别对待。
    """

    scored = scored_candidates(region)
    if len(scored) < 2:
        return None
    texts = {normalize(c.text) for c in scored}
    return len(texts) == 1


def region_confidence(region: Region, fallback: float = UNSCORED_FALLBACK) -> float:
    """一个区域的置信度。"""

    scored = scored_candidates(region)
    if not scored:
        return fallback

    best = max(c.score or 0.0 for c in scored)
    if candidates_agree(region) is False:
        best *= DISAGREEMENT_FACTOR
    return max(0.0, min(1.0, best))


def region_is_contested(region: Region) -> bool:
    """是否存在互相矛盾的候选。"""

    return candidates_agree(region) is False


def field_confidence(field: FieldValue, regions: dict[str, Region]) -> float:
    """字段置信度：从支撑它的区域悲观继承。

    没有关联区域时（例如直接从整页文本里正则抓出来的字段），返回 0，
    交给上层标记为"无证据支撑"，而不是假装自己很可信。
    """

    supporting = [regions[rid] for rid in field.region_ids if rid in regions]
    if not supporting:
        return 0.0
    return min(region_confidence(region) for region in supporting)


def expected_loss(confidence: float, severity: Severity) -> float:
    """单字段的期望损失。

    损失上限是 1（完全错且是致命字段），随置信度线性下降。这个值不
    是概率，只用来在同一份文档里横向比较字段之间"谁更值得多查一次"。
    """

    weight = SEVERITY_WEIGHT[severity.value]
    return weight * max(0.0, 1.0 - confidence)


def aggregate_expected_loss(fields: list[FieldValue]) -> float:
    """整份文档所有字段的期望损失之和。"""

    return sum(expected_loss(f.confidence, f.severity) for f in fields)


def worst_field(fields: list[FieldValue]) -> FieldValue | None:
    """期望损失最大的字段。"""

    if not fields:
        return None
    return max(fields, key=lambda f: expected_loss(f.confidence, f.severity))


def risk_band(expected: float) -> str:
    """把期望损失映射成人类能读的三档。"""

    if expected >= 0.35:
        return "high"
    if expected >= 0.12:
        return "medium"
    return "low"


def severity_for(field: FieldValue, policy) -> Severity:
    """按策略给字段定级（委托给 RiskPolicy，参数化方便测试）。"""

    return policy.severity_for(field.name, field.value)


def summarize_risk(fields: list[FieldValue], regions: dict[str, Region]) -> dict:
    """给上层用的一份风险摘要。"""

    total = aggregate_expected_loss(fields)
    critical = [
        f for f in fields if f.severity is Severity.CRITICAL and expected_loss(f.confidence, f.severity) > 0.05
    ]
    contested = [rid for rid, region in regions.items() if region_is_contested(region)]
    return {
        "expected_loss": round(total, 6),
        "band": risk_band(total),
        "critical_open": [f.name for f in critical],
        "contested_regions": contested,
        "field_count": len(fields),
    }
