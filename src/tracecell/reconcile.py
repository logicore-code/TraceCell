"""跨文档对账。

单份文档识别得再准，也可能出现两份都识别正确、但互相矛盾的情况：合同
里的金额是 186,800，发票上是 168,800；合同甲方写的是"杭州明远科技"，
营业执照上是"杭州明远科技有限公司"。

OCR 质量模型解决不了这类问题——它只知道自己那一份文档看得对不对。
TraceCell 把对账放在证据层：矛盾项同样带着两侧的坐标和候选，人看一眼
就能判断谁对谁错。

字段类型靠字段名推断（金额 / 日期 / 主体），不同类型用不同的归一化规则，
避免把"¥186,800.00"和"186800"当成两个数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .textnorm import (
    normalize,
    normalize_date,
    normalize_entity,
    normalize_money,
    similarity,
)
from .types import DocumentLedger, Finding, Severity, Status

MONEY_KEYWORDS = ("金额", "合计", "税额", "单价", "价款", "总额", "应付", "实付")
DATE_KEYWORDS = ("日期", "签署", "开票", "到期", "生效", "期限", "年", "月", "日")
ENTITY_KEYWORDS = ("名称", "甲方", "乙方", "卖方", "买方", "供应商", "客户", "收款人", "付款人")


def infer_kind(field_name: str) -> str:
    """推断字段的归一化类型。"""

    if any(keyword in field_name for keyword in MONEY_KEYWORDS):
        return "money"
    if any(keyword in field_name for keyword in DATE_KEYWORDS):
        return "date"
    if any(keyword in field_name for keyword in ENTITY_KEYWORDS):
        return "entity"
    return "text"


def normalize_by_kind(field_name: str, value: str) -> str:
    """按字段类型归一化，用于比较。"""

    kind = infer_kind(field_name)
    if kind == "money":
        return normalize_money(value) or normalize(value)
    if kind == "date":
        return normalize_date(value) or normalize(value)
    if kind == "entity":
        return normalize_entity(value)
    return normalize(value)


@dataclass
class Mismatch:
    """一处跨文档矛盾。"""

    field_name: str
    values: list[dict[str, Any]] = field(default_factory=list)
    kind: str = "text"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "kind": self.kind,
            "values": self.values,
            "note": self.note,
        }


@dataclass
class ReconciliationResult:
    """对账结果。"""

    compared_fields: int = 0
    matched: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    single_source: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return not self.mismatches

    def to_dict(self) -> dict[str, Any]:
        return {
            "compared_fields": self.compared_fields,
            "matched": self.matched,
            "mismatch_count": len(self.mismatches),
            "mismatches": [m.to_dict() for m in self.mismatches],
            "single_source": self.single_source,
        }


def _collect(ledgers: list[DocumentLedger]) -> dict[str, list[tuple[DocumentLedger, Any]]]:
    grouped: dict[str, list[tuple[DocumentLedger, Any]]] = {}
    for ledger in ledgers:
        for fld in ledger.fields:
            if not fld.value:
                continue
            grouped.setdefault(fld.name, []).append((ledger, fld))
    return grouped


def reconcile(
    ledgers: list[DocumentLedger],
    entity_similarity: float = 0.85,
) -> ReconciliationResult:
    """多份文档之间做字段级对账。

    entity_similarity 是主体名称的相似度门槛。低于它的视为不一致——
    因为"杭州明远科技"和"杭州明远科技公司"在法律上是两个主体，不该
    因为字面接近就放过。
    """

    result = ReconciliationResult()
    grouped = _collect(ledgers)

    for name, entries in grouped.items():
        if len(entries) < 2:
            result.single_source.append(name)
            continue

        result.compared_fields += 1
        kind = infer_kind(name)
        normalized = [(ledger, fld, normalize_by_kind(name, fld.value)) for ledger, fld in entries]

        distinct = {value for _, _, value in normalized if value}
        conflict = False
        if kind == "entity" and len(distinct) > 1:
            base = normalized[0][2]
            conflict = all(similarity(base, other) < entity_similarity for _, _, other in normalized[1:])
        elif len(distinct) > 1:
            conflict = True

        if conflict:
            mismatch = Mismatch(
                field_name=name,
                kind=kind,
                note=_conflict_note(kind, normalized),
                values=[
                    {
                        "doc_id": ledger.doc_id,
                        "value": fld.value,
                        "normalized": value,
                        "confidence": round(fld.confidence, 4),
                        "region_ids": list(fld.region_ids),
                    }
                    for ledger, fld, value in normalized
                ],
            )
            result.mismatches.append(mismatch)
            for ledger, fld, _ in normalized:
                fld.status = Status.DISPUTED
                fld.note = f"与《{mismatch.note}》存在跨文档矛盾"
        else:
            result.matched += 1

    return result


def _conflict_note(kind: str, normalized: list[tuple[Any, Any, str]]) -> str:
    doc_ids = ", ".join(sorted({ledger.doc_id for ledger, _, _ in normalized}))
    if kind == "money":
        return f"金额不一致（{doc_ids}）"
    if kind == "date":
        return f"日期不一致（{doc_ids}）"
    if kind == "entity":
        return f"主体名称不一致（{doc_ids}）"
    return f"取值不一致（{doc_ids}）"


def findings_from_reconciliation(result: ReconciliationResult) -> list[Finding]:
    """把对账结果转成账本里的 Finding。"""

    findings: list[Finding] = []
    for mismatch in result.mismatches:
        region_ids: list[str] = []
        field_names: list[str] = []
        for entry in mismatch.values:
            region_ids.extend(entry.get("region_ids", []))
            field_names.append(entry["doc_id"])
        findings.append(
            Finding(
                kind="contradiction",
                severity=Severity.CRITICAL if mismatch.kind in ("money", "entity") else Severity.HIGH,
                message=f"字段「{mismatch.field_name}」在多份文档之间不一致：{mismatch.note}",
                region_ids=region_ids,
                field_names=field_names,
                suggestion="核对各文档原件，确认以哪一份为准",
            )
        )
    return findings


def merge_ledgers(ledgers: list[DocumentLedger]) -> DocumentLedger:
    """把多份文档合并成一份带对账结论的总账。"""

    if not ledgers:
        raise ValueError("没有可合并的账本")

    merged = DocumentLedger(
        doc_id=f"bundle:{len(ledgers)}",
        source="; ".join(ledger.source for ledger in ledgers),
        pages=sum(ledger.pages for ledger in ledgers),
    )
    for ledger in ledgers:
        for region in ledger.regions:
            # 区域 id 在不同文档间可能撞车，加文档前缀。
            region.id = f"{ledger.doc_id}:{region.id}"
            merged.regions.append(region)
        for fld in ledger.fields:
            fld.region_ids = [f"{ledger.doc_id}:{rid}" for rid in fld.region_ids]
            merged.fields.append(fld)

    result = reconcile(ledgers)
    merged.findings.extend(findings_from_reconciliation(result))
    merged.meta["reconciliation"] = result.to_dict()
    merged.verdict = "full_review" if result.mismatches else merged.verdict
    return merged
