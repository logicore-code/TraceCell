"""三条路线的基线对比。

任何"文档核验"方案都要回答同一个问题：**人工要花多少力气，能拦下多少错误。**

对比三条路线：

* **全量人工**——每块都给人看。人工成本最高，漏检为零。
* **只信 OCR**——OCR 读什么就是什么。零人工成本，但识别错误和单据自身的
  矛盾全都漏掉。
* **TraceCell**——评分 + 勾稽 + 跨文档对账，按风险分流。

对比需要知道每块的**真值**。样例数据里每个区域带 ``truth`` 字段：默认就是
印在纸上的内容，需要修正的地方单独标出来（比如那张购销合同上算术就错的一行，
真值是 23,600.00 而不是印上去的 26,600.00）。

错误分两类，区别很重要：

* **识别错误**——OCR 把纸上写的读错了。换更强的模型或者再读一次能解决。
* **单据矛盾**——两边都读对了，但单据本身自相矛盾（明细加起来不等于合计、
  合同总价和发票对不上）。**换模型解决不了**，只能靠跨字段校验。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .chinese_amount import parse_chinese_amount
from .textnorm import normalize
from .types import DocumentLedger

ALL_HUMAN = "全量人工"
OCR_ONLY = "只信 OCR"
TRACECELL = "TraceCell"


@dataclass
class StrategyResult:
    """一条路线的结果。"""

    name: str
    human_cells: int = 0
    total_cells: int = 0
    missed_reading_errors: int = 0
    missed_document_errors: int = 0
    cost: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "human_cells": self.human_cells,
            "total_cells": self.total_cells,
            "human_ratio": round(self.human_cells / self.total_cells, 4) if self.total_cells else 0.0,
            "missed_reading_errors": self.missed_reading_errors,
            "missed_document_errors": self.missed_document_errors,
            "cost": round(self.cost, 4),
            "notes": self.notes,
        }


@dataclass
class BaselineReport:
    """三条路线的对比结果。"""

    results: list[StrategyResult] = field(default_factory=list)
    reading_errors: int = 0
    document_errors: int = 0
    total_cells: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cells": self.total_cells,
            "reading_errors": self.reading_errors,
            "document_errors": self.document_errors,
            "strategies": [r.to_dict() for r in self.results],
        }

    def render(self) -> str:
        """渲染成一张能直接贴进报告的表。"""

        header = (
            f"{'路线':<12}{'人工复核':>10}{'人工占比':>10}"
            f"{'漏检·识别错':>12}{'漏检·单据矛盾':>14}{'核验成本':>12}"
        )
        lines = [header, "-" * len(header)]
        for result in self.results:
            ratio = result.human_cells / result.total_cells if result.total_cells else 0.0
            lines.append(
                f"{result.name:<12}{result.human_cells:>10}{ratio:>9.1%}"
                f"{result.missed_reading_errors:>12}{result.missed_document_errors:>14}"
                f"{result.cost:>11.2f} rcu"
            )
        lines.append("")
        lines.append(
            f"共 {self.total_cells} 个区域，其中 {self.reading_errors} 处是识别错误"
            f"（OCR 没读对），{self.document_errors} 处是单据自身矛盾（读对了但单据错）。"
        )
        return "\n".join(lines)


def _same_reading(left: str, right: str) -> bool:
    """两次读法是不是同一个意思。

    字符串归一化之后还不一样，不一定就是错——中文大写金额的"圆整"和
    "元整"是同一个数，只是写法不同。这种情况不能算成识别错误。
    """

    if normalize(left) == normalize(right):
        return True
    left_amount = parse_chinese_amount(left)
    right_amount = parse_chinese_amount(right)
    if left_amount is not None and right_amount is not None:
        return left_amount == right_amount
    return False


def _accepted_by_ocr(region) -> str:
    """只信 OCR 时采用的读法：首个 primary 候选，没有就用区域原文。"""

    for candidate in region.candidates:
        if candidate.origin == "primary":
            return candidate.text
    return region.text


def _accepted_by_tracecell(region) -> str:
    """TraceCell 采用的读法：分数最高的候选。"""

    best = region.best
    return best.text if best is not None else region.text


def compare(
    ledgers: list[DocumentLedger],
    truths: dict[tuple[str, str], str],
    document_errors: int = 0,
) -> BaselineReport:
    """对比三条路线。

    truths 是 (文档标识, 区域标识) → 真值 的映射。区域真值缺席时按"没有错"
    处理。
    """

    report = BaselineReport()

    for ledger in ledgers:
        report.total_cells += len(ledger.regions)

    reading_errors = 0
    ocr_missed = 0
    tracecell_missed = 0
    ocr_missed_regions: list[str] = []
    tracecell_missed_regions: list[str] = []
    queue: set[tuple[str, str]] = set()

    for ledger in ledgers:
        doc_id = ledger.doc_id
        queue |= {(doc_id, rid) for rid in ledger.review_queue}
        for region in ledger.regions:
            truth = truths.get((doc_id, region.id))
            if truth is None:
                continue
            if not _same_reading(_accepted_by_ocr(region), truth):
                reading_errors += 1
                ocr_missed += 1
                ocr_missed_regions.append(f"{doc_id}:{region.id}")
            elif (
                not _same_reading(_accepted_by_tracecell(region), truth)
                and (doc_id, region.id) not in queue
            ):
                # TraceCell 也读错了，但这一格进了人工队列，所以不算漏检
                tracecell_missed += 1
                tracecell_missed_regions.append(f"{doc_id}:{region.id}")

    report.reading_errors = reading_errors
    report.document_errors = document_errors

    tracecell_cost = sum(ledger.budget.total_cost for ledger in ledgers)

    report.results = [
        StrategyResult(
            name=ALL_HUMAN,
            human_cells=report.total_cells,
            total_cells=report.total_cells,
            missed_reading_errors=0,
            missed_document_errors=0,
            cost=0.0,
            notes=["每块都给人看，漏检为零，但人力全压上去"],
        ),
        StrategyResult(
            name=OCR_ONLY,
            human_cells=0,
            total_cells=report.total_cells,
            missed_reading_errors=ocr_missed,
            missed_document_errors=document_errors,
            cost=0.0,
            notes=[
                "零人工成本，但识别错误和单据矛盾一起漏掉",
                f"漏掉的识别错误在 {', '.join(ocr_missed_regions)}" if ocr_missed_regions else "",
            ],
        ),
        StrategyResult(
            name=TRACECELL,
            human_cells=len(queue),
            total_cells=report.total_cells,
            missed_reading_errors=tracecell_missed,
            missed_document_errors=0,
            cost=tracecell_cost,
            notes=[
                f"人工只审 {len(queue)} 格（{len(queue) / report.total_cells:.0%}），"
                "识别错误交给评分拦，单据矛盾交给勾稽和跨文档对账"
                if report.total_cells
                else "",
                f"仍有漏检：{', '.join(tracecell_missed_regions)}"
                if tracecell_missed_regions
                else "",
            ],
        ),
    ]
    return report
