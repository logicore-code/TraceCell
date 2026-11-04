"""TraceCell 的数据契约。

这个模块只定义结构，不做任何判断。判断分散在 risk / budget / routing /
calibration 四个模块里，这样每一层都能单独替换、单独测试。

坐标系统一用归一化的 [0, 1]，原点在左上角。原因很简单：版面检测、
浏览器渲染、打印输出用的像素尺寸都不一样，只有归一化坐标能在不同后端
之间无损搬运。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Category(str, Enum):
    """版面块的类别。与 MiniCPM-OCR Jev 支持的三类保持一致。"""

    TEXT = "text"
    TABLE = "table"
    FORMULA = "formula"
    FIGURE = "figure"
    SEAL = "seal"


class Verdict(str, Enum):
    """Jev 评分的执行状态。

    官方文档反复强调的一点：`verdict != scored` 时返回的分数是失败态的
    固定值，不能当成"模型认为这段识别错了"。TraceCell 在数据层就把这两种
    情况分开，避免下游误判。
    """

    SCORED = "scored"
    NOT_SCORED = "not_scored"
    FAILED = "failed"


class Status(str, Enum):
    """区域或字段在核验闭环中的最终状态。"""

    ACCEPTED = "accepted"      # 证据充分，免检通过
    REPAIRED = "repaired"      # 初次解析有问题，重解析后修正
    PENDING = "pending"        # 证据不足，需要人工看
    ESCALATED = "escalated"    # 已升级到多模态模型复核，仍存疑
    DISPUTED = "disputed"      # 多份文档之间互相矛盾


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# 严重级别到风险权重的映射。用于期望损失计算，见 risk.py。
SEVERITY_WEIGHT: dict[str, float] = {
    Severity.CRITICAL.value: 1.0,
    Severity.HIGH.value: 0.55,
    Severity.MEDIUM.value: 0.25,
    Severity.LOW.value: 0.08,
}


@dataclass(frozen=True)
class BBox:
    """归一化的轴对齐矩形。"""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.x0 > self.x1 or self.y0 > self.y1:
            raise ValueError(f"非法的 bbox: {self}")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_pixels(self, width: int, height: int) -> tuple[int, int, int, int]:
        """转成 (left, top, right, bottom) 像素坐标，并夹在图像范围内。"""

        left = max(0, min(width, round(self.x0 * width)))
        top = max(0, min(height, round(self.y0 * height)))
        right = max(left, min(width, round(self.x1 * width)))
        bottom = max(top, min(height, round(self.y1 * height)))
        return left, top, right, bottom

    def iou(self, other: BBox) -> float:
        """交并比。用于判断两个字段是否来自同一块原文。"""

        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, other: BBox) -> bool:
        return (
            self.x0 <= other.x0
            and self.y0 <= other.y0
            and self.x1 >= other.x1
            and self.y1 >= other.y1
        )

    @classmethod
    def from_pixels(cls, left: int, top: int, right: int, bottom: int, width: int, height: int) -> BBox:
        return cls(
            x0=round(left / width, 6),
            y0=round(top / height, 6),
            x1=round(right / width, 6),
            y1=round(bottom / height, 6),
        )


@dataclass
class Candidate:
    """一个区域的一种识别结果。

    同一个区域可能有多条候选：首次解析、按不同视觉档位重解析、或者
    多模态模型复核后的修正。Jev 的作用就是在这堆候选里挑一条。
    """

    text: str
    origin: str                       # "primary" / "reparse-4x" / "reparse-16x" / "vlm-review"
    score: float | None = None        # Jev 分数，None 表示没评过
    verdict: Verdict = Verdict.NOT_SCORED
    detail: str = ""                  # 评分失败时的原因，便于排障
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def scored(self) -> bool:
        return self.verdict is Verdict.SCORED and self.score is not None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["verdict"] = self.verdict.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Candidate:
        data = dict(data)
        data["verdict"] = Verdict(data.get("verdict", "not_scored"))
        return cls(**data)


@dataclass
class Region:
    """页面上的一个版面块，是最小的证据单元。"""

    id: str
    page: int
    bbox: BBox
    category: Category
    text: str
    candidates: list[Candidate] = field(default_factory=list)
    status: Status = Status.PENDING
    note: str = ""

    @property
    def primary(self) -> Candidate | None:
        for cand in self.candidates:
            if cand.origin == "primary":
                return cand
        return self.candidates[0] if self.candidates else None

    @property
    def best(self) -> Candidate | None:
        """分数最高的已评分候选。没有分数时退回首次解析结果。"""

        scored = [c for c in self.candidates if c.scored]
        if scored:
            return max(scored, key=lambda c: c.score or 0.0)
        return self.primary

    @property
    def selected_text(self) -> str:
        best = self.best
        return best.text if best else self.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "page": self.page,
            "bbox": asdict(self.bbox),
            "category": self.category.value,
            "text": self.text,
            "candidates": [c.to_dict() for c in self.candidates],
            "status": self.status.value,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Region:
        return cls(
            id=data["id"],
            page=data["page"],
            bbox=BBox(**data["bbox"]),
            category=Category(data["category"]),
            text=data["text"],
            candidates=[Candidate.from_dict(c) for c in data.get("candidates", [])],
            status=Status(data.get("status", "pending")),
            note=data.get("note", ""),
        )


@dataclass
class FieldValue:
    """账本里的一个业务字段。

    字段置信度从支撑它的区域继承而来，详见 risk.py 的 `field_confidence`。
    """

    name: str
    value: str
    region_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    severity: Severity = Severity.MEDIUM
    status: Status = Status.PENDING
    source: str = ""                 # 抽取来源，例如 "kie" 或 "regex"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "region_ids": self.region_ids,
            "confidence": round(self.confidence, 4),
            "severity": self.severity.value,
            "status": self.status.value,
            "source": self.source,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldValue:
        return cls(
            name=data["name"],
            value=data["value"],
            region_ids=data.get("region_ids", []),
            confidence=data.get("confidence", 0.0),
            severity=Severity(data.get("severity", "medium")),
            status=Status(data.get("status", "pending")),
            source=data.get("source", ""),
            note=data.get("note", ""),
        )


@dataclass
class Finding:
    """核验过程中发现的问题。每条问题都挂在具体区域上。"""

    kind: str                        # low_score / contradiction / unresolved / budget_exhausted
    severity: Severity
    message: str
    region_ids: list[str] = field(default_factory=list)
    field_names: list[str] = field(default_factory=list)
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": self.severity.value,
            "message": self.message,
            "region_ids": self.region_ids,
            "field_names": self.field_names,
            "suggestion": self.suggestion,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Finding:
        return cls(
            kind=data["kind"],
            severity=Severity(data.get("severity", "medium")),
            message=data["message"],
            region_ids=data.get("region_ids", []),
            field_names=data.get("field_names", []),
            suggestion=data.get("suggestion", ""),
        )


@dataclass
class Decision:
    """Agent 针对某个区域做出的下一步动作及其理由。"""

    region_id: str
    action: str
    reason: str
    cost: float = 0.0
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    risk_before: float = 0.0
    risk_after: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetReport:
    """一次核验的成本账单。

    这个报表能说清楚哪些格子额外花了推理、换来了多少风险下降。
    """

    total_cost: float = 0.0
    base_cost: float = 0.0
    """初次解析与首轮评分的开销。这部分省不掉，也不占额外核验预算。"""

    extra_cost: float = 0.0
    """额外核验的开销，受 total_budget 约束。"""

    cost_by_action: dict[str, float] = field(default_factory=dict)
    extra_calls: int = 0
    accepted_regions: int = 0
    repaired_regions: int = 0
    escalated_regions: int = 0
    pending_regions: int = 0
    human_cells: int = 0
    residual_risk: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["cost_by_action"] = dict(self.cost_by_action)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BudgetReport:
        return cls(**data)


@dataclass
class DocumentLedger:
    """一份文档的完整证据账本。

    这是 TraceCell 的核心产物。区别于"输出一段 Markdown"，账本里每一
    个值都能回答三个问题：从哪来（bbox）、有多可信（confidence）、
    还需不需要人看（status）。
    """

    doc_id: str
    source: str
    pages: int = 1
    regions: list[Region] = field(default_factory=list)
    fields: list[FieldValue] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    budget: BudgetReport = field(default_factory=BudgetReport)
    verdict: str = "undecided"        # auto_pass / spot_check / full_review / undecided
    review_queue: list[str] = field(default_factory=list)
    markdown: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # -- 便捷访问 ---------------------------------------------------------
    def region(self, region_id: str) -> Region | None:
        for reg in self.regions:
            if reg.id == region_id:
                return reg
        return None

    def field(self, name: str) -> FieldValue | None:
        for fld in self.fields:
            if fld.name == name:
                return fld
        return None

    def regions_for(self, region_ids: Iterable[str]) -> list[Region]:
        wanted = set(region_ids)
        return [r for r in self.regions if r.id in wanted]

    # -- 序列化 -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "pages": self.pages,
            "regions": [r.to_dict() for r in self.regions],
            "fields": [f.to_dict() for f in self.fields],
            "findings": [f.to_dict() for f in self.findings],
            "decisions": [d.to_dict() for d in self.decisions],
            "budget": self.budget.to_dict(),
            "verdict": self.verdict,
            "review_queue": self.review_queue,
            "markdown": self.markdown,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentLedger:
        return cls(
            doc_id=data["doc_id"],
            source=data.get("source", ""),
            pages=data.get("pages", 1),
            regions=[Region.from_dict(r) for r in data.get("regions", [])],
            fields=[FieldValue.from_dict(f) for f in data.get("fields", [])],
            findings=[Finding.from_dict(f) for f in data.get("findings", [])],
            decisions=[Decision(**d) for d in data.get("decisions", [])],
            budget=BudgetReport.from_dict(data.get("budget", {})),
            verdict=data.get("verdict", "undecided"),
            review_queue=data.get("review_queue", []),
            markdown=data.get("markdown", ""),
            meta=data.get("meta", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, payload: str) -> DocumentLedger:
        return cls.from_dict(json.loads(payload))
