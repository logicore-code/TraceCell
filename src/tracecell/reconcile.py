"""跨文档对账与单据内部勾稽校验。

两类问题都归到这里：

* **单据内部**：明细加起来对不对等于合计，数量乘单价对不对等于金额。
* **跨文档**：同一笔业务在不同单据上的落点对不对得上。

跨文档这一类以前是靠字段名撞车比对的，靠不住——合同叫"乙方"、发票叫
"销售方名称"，根本撞不上。现在改成按场景显式配对（见 scenarios.py 的
CROSS_RULES），顺带修掉一个隐患：以前发票的"金额"和报销单的"金额"会被
拿去互比，128,600 和 2,864.50 一比就是个假矛盾。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from .chinese_amount import parse_chinese_amount
from .scenarios import CROSS_RULES, ConsistencyRule, cross_rules_for
from .textnorm import (
    normalize,
    normalize_date,
    normalize_entity,
    normalize_money,
    similarity,
)
from .types import DocumentLedger, FieldValue, Finding, Severity, Status

MONEY_KEYWORDS = ("金额", "合计", "税额", "单价", "价款", "总额", "应付", "实付", "总计")
DATE_KEYWORDS = ("日期", "签署", "开票", "到期", "生效", "期限", "年", "月", "日")
ENTITY_KEYWORDS = ("名称", "甲方", "乙方", "需方", "供方", "卖方", "买方", "供应商", "客户",
                    "收款人", "付款人", "单位")


def infer_kind(field_name: str) -> str:
    """推断字段的归一化类型。"""

    if any(keyword in field_name for keyword in MONEY_KEYWORDS):
        return "money"
    if any(keyword in field_name for keyword in DATE_KEYWORDS):
        return "date"
    if any(keyword in field_name for keyword in ENTITY_KEYWORDS):
        return "entity"
    return "text"


def normalize_by_kind(field_name: str, value: str, kind: str | None = None) -> str:
    """按字段类型归一化，用于比较。"""

    kind = kind or infer_kind(field_name)
    if kind == "money":
        return normalize_money(value) or normalize(value)
    if kind == "date":
        return normalize_date(value) or normalize(value)
    if kind == "entity":
        return normalize_entity(value)
    return normalize(value)


def to_number(value: str) -> float | None:
    """把数值字段转成浮点。

    乘法勾稽要用：数量是"50"，单价是"2,572.00"，税率是"13%"。
    """

    if not value:
        return None
    cleaned = normalize(value).replace("¥", "").replace("￥", "").replace("%", "")
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


@dataclass
class Mismatch:
    """一处矛盾。"""

    field_name: str
    values: list[dict[str, Any]] = field(default_factory=list)
    kind: str = "text"
    note: str = ""
    left_field: str = ""
    right_field: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "kind": self.kind,
            "values": self.values,
            "note": self.note,
            "left_field": self.left_field,
            "right_field": self.right_field,
        }


@dataclass
class ReconciliationResult:
    """对账结果。"""

    compared_fields: int = 0
    matched: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    single_source: list[str] = field(default_factory=list)
    pairs: list[dict[str, Any]] = field(default_factory=list)

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
            "pairs": self.pairs,
        }


def _scenario_of(ledger: DocumentLedger) -> str:
    return str(ledger.meta.get("scenario", ""))


def _collect_pairs(ledgers: list[DocumentLedger]) -> list[dict[str, Any]]:
    """找出所有该比对的字段对。

    两条来源：场景显式配对（跨文档），以及同场景同名字段（同一类单据之间）。
    """

    scenarios = [_scenario_of(ledger) for ledger in ledgers]
    pairs: list[dict[str, Any]] = []
    covered: set[tuple[str, str, str, str]] = set()

    for rule in cross_rules_for(scenarios):
        for left in ledgers:
            if _scenario_of(left) != rule.left_scenario:
                continue
            for right in ledgers:
                if right.doc_id == left.doc_id or _scenario_of(right) != rule.right_scenario:
                    continue
                left_field = left.field(rule.left_field)
                right_field = right.field(rule.right_field)
                if left_field is None or right_field is None:
                    continue
                key = (left.doc_id, rule.left_field, right.doc_id, rule.right_field)
                covered.add(key)
                covered.add((right.doc_id, rule.right_field, left.doc_id, rule.left_field))
                pairs.append(
                    {
                        "left": left,
                        "left_field": left_field,
                        "right": right,
                        "right_field": right_field,
                        "kind": rule.kind,
                        "label": f"{rule.left_field} ↔ {rule.right_field}",
                        "source": "cross-rule",
                    }
                )

    grouped: dict[tuple[str, str], list[tuple[DocumentLedger, FieldValue]]] = defaultdict(list)
    for ledger in ledgers:
        for fld in ledger.fields:
            grouped[(_scenario_of(ledger), fld.name)].append((ledger, fld))

    for (scenario, name), items in grouped.items():
        for index in range(len(items)):
            for other in range(index + 1, len(items)):
                left_ledger, left_field = items[index]
                right_ledger, right_field = items[other]
                if left_ledger.doc_id == right_ledger.doc_id:
                    continue
                key = (left_ledger.doc_id, name, right_ledger.doc_id, name)
                if key in covered:
                    continue
                covered.add(key)
                covered.add((right_ledger.doc_id, name, left_ledger.doc_id, name))
                pairs.append(
                    {
                        "left": left_ledger,
                        "left_field": left_field,
                        "right": right_ledger,
                        "right_field": right_field,
                        "kind": infer_kind(name),
                        "label": name,
                        "source": "same-scenario",
                    }
                )

    return pairs


def reconcile(ledgers: list[DocumentLedger], entity_similarity: float = 0.85) -> ReconciliationResult:
    """多份文档之间做字段级对账。"""

    result = ReconciliationResult()
    pairs = _collect_pairs(ledgers)
    result.pairs = [
        {
            "left": pair["left"].doc_id,
            "left_field": pair["left_field"].name,
            "right": pair["right"].doc_id,
            "right_field": pair["right_field"].name,
            "kind": pair["kind"],
            "source": pair["source"],
        }
        for pair in pairs
    ]

    for pair in pairs:
        result.compared_fields += 1
        left_field = pair["left_field"]
        right_field = pair["right_field"]
        kind = pair["kind"]

        left_value = normalize_by_kind(left_field.name, left_field.value, kind)
        right_value = normalize_by_kind(right_field.name, right_field.value, kind)

        conflict = False
        if kind == "entity" and left_value != right_value:
            # 主体名称用相似度而不是精确匹配："杭州明远科技"和"杭州明远科技
            # 公司"在法律上是两个主体，不该因为字面接近就放过。
            conflict = similarity(left_field.value, right_field.value) < entity_similarity
        else:
            conflict = left_value != right_value

        if not conflict:
            result.matched += 1
            continue

        label = pair["label"]
        result.mismatches.append(
            Mismatch(
                field_name=label,
                kind=kind,
                left_field=left_field.name,
                right_field=right_field.name,
                note=_conflict_note(kind, pair),
                values=[
                    {
                        "doc_id": pair["left"].doc_id,
                        "value": left_field.value,
                        "normalized": left_value,
                        "confidence": round(left_field.confidence, 4),
                        "region_ids": list(left_field.region_ids),
                        "field": left_field.name,
                    },
                    {
                        "doc_id": pair["right"].doc_id,
                        "value": right_field.value,
                        "normalized": right_value,
                        "confidence": round(right_field.confidence, 4),
                        "region_ids": list(right_field.region_ids),
                        "field": right_field.name,
                    },
                ],
            )
        )
        for side in (left_field, right_field):
            side.status = Status.DISPUTED

    # 只在一份文档里出现、又没被任何配对覆盖的字段
    paired_names = {pair["left_field"].name for pair in pairs} | {
        pair["right_field"].name for pair in pairs
    }
    counts: dict[str, int] = defaultdict(int)
    for ledger in ledgers:
        for fld in ledger.fields:
            counts[fld.name] += 1
    result.single_source = sorted(
        name
        for name, count in counts.items()
        if count == 1 and name not in paired_names
    )

    return result


def _conflict_note(kind: str, pair: dict[str, Any]) -> str:
    left = pair["left"].doc_id
    right = pair["right"].doc_id
    if kind == "money":
        return f"金额不一致（{left} ↔ {right}）"
    if kind == "date":
        return f"日期不一致（{left} ↔ {right}）"
    if kind == "entity":
        return f"主体名称不一致（{left} ↔ {right}）"
    return f"取值不一致（{left} ↔ {right}）"


def group_line_items(
    fields: list[FieldValue], prefix: str = "明细"
) -> dict[str, dict[str, FieldValue]]:
    """把明细字段按品名分组。

    "明细:智能网关设备:数量" 这类字段名拆成 {"智能网关设备": {"数量": ...}}。
    逐行的加总与乘法校验都要靠它。
    """

    groups: dict[str, dict[str, FieldValue]] = {}
    marker = f"{prefix}:"
    for fld in fields:
        if not fld.name.startswith(marker):
            continue
        rest = fld.name[len(marker) :]
        if ":" not in rest:
            continue
        label, _, column = rest.partition(":")
        groups.setdefault(label, {})[column] = fld
    return groups


def check_sum_consistency(fields: list[FieldValue], rules: list[ConsistencyRule]) -> list[Finding]:
    """单据内部的勾稽校验。

    支持五种关系：

    * sum          总数 = 各分项之和
    * product      总数 = 各因子之积
    * amount_forms 大写金额与小写金额是同一笔数
    * rows_product 每一行的结果列 = 该行各因子列之积
    * rows_sum     整单合计 = 各明细行之和

    这类矛盾 OCR 质量模型解决不了——分项和总数都识别正确，错的是单据本身。

    两种逐行规则里的 total / components 指的是**列名**（"金额"、"数量"），
    不是字段名；其余三种指的是字段名。
    """

    findings: list[Finding] = []
    for rule in rules:
        if rule.kind in ("rows_product", "rows_sum"):
            findings.extend(_check_rows(fields, rule))
            continue

        total_field = next((f for f in fields if f.name == rule.total), None)
        if total_field is None:
            continue

        # 分项按前缀组匹配：规则写"明细"对应"明细:差旅费"这一组，
        # 规则写"金额"且存在同名字段时就是那一个。
        chosen: list[FieldValue] = []
        for name in rule.components:
            exact = next((f for f in fields if f.name == name), None)
            if exact is not None:
                chosen.append(exact)
                continue
            group = [f for f in fields if f.name.startswith(f"{name}:")]
            if not group:
                break
            chosen.extend(group)
        if len(chosen) < len(rule.components):
            continue

        if rule.kind == "amount_forms":
            # 大写与小写是同一笔金额的两种写法，比的是数值不是字符串
            chinese = parse_chinese_amount(chosen[0].value)
            arabic = normalize_money(total_field.value)
            if chinese is None or arabic is None:
                continue
            difference = float(chinese - Decimal(arabic))
            if abs(difference) <= rule.tolerance:
                continue
            findings.append(
                Finding(
                    kind="sum_mismatch",
                    severity=Severity.CRITICAL,
                    message=(
                        f"大写与小写金额不一致：大写「{chosen[0].value}」为 {chinese:,.2f}，"
                        f"小写「{total_field.value}」为 {float(Decimal(arabic)):,.2f}，"
                        f"差额 {difference:+,.2f}"
                    ),
                    region_ids=[rid for c in chosen for rid in c.region_ids]
                    + list(total_field.region_ids),
                    field_names=[c.name for c in chosen] + [total_field.name],
                    suggestion="合同以大写金额为准，核对原件上大写那一遍",
                )
            )
            continue

        if rule.kind == "product":
            numbers = [to_number(f.value) for f in chosen]
            if any(n is None for n in numbers):
                continue
            expected = 1.0
            for number in numbers:
                expected *= number
            actual = to_number(total_field.value)
            if actual is None:
                continue
            difference = expected - actual
            unit = " × ".join(f"{f.name.split(':')[-1]} {f.value}" for f in chosen)
            detail = f"{total_field.name} {total_field.value}，按 {unit} 应为 {expected:,.2f}"
        else:
            amounts = []
            broken = False
            for component in chosen:
                amount = normalize_money(component.value)
                if amount is None:
                    broken = True
                    break
                amounts.append(Decimal(amount))
            if broken:
                continue
            expected_dec = sum(amounts, Decimal("0"))
            total_amount = normalize_money(total_field.value)
            if total_amount is None:
                continue
            difference = float(expected_dec - Decimal(total_amount))
            unit = "、".join(f"{f.name.split(':')[-1]} {f.value}" for f in chosen)
            detail = f"{total_field.name} {total_field.value}，明细 {unit} 合计 {expected_dec:,.2f}"

        if abs(difference) <= rule.tolerance:
            continue

        findings.append(
            Finding(
                kind="sum_mismatch",
                severity=Severity.CRITICAL,
                message=f"勾稽不符：{detail}，差额 {difference:+,.2f}",
                region_ids=[rid for c in chosen for rid in c.region_ids]
                + list(total_field.region_ids),
                field_names=[c.name for c in chosen] + [total_field.name],
                suggestion="核对这一项，确认是单据笔误还是识别错误",
            )
        )
    return findings


def _check_rows(fields: list[FieldValue], rule: ConsistencyRule) -> list[Finding]:
    """逐行规则：行内乘法与整单加总。"""

    findings: list[Finding] = []
    groups = group_line_items(fields)

    if rule.kind == "rows_product":
        # 每一行的 结果列 = 各因子列之积
        for label, columns in groups.items():
            factors = [columns.get(name) for name in rule.components]
            result = columns.get(rule.total)
            if result is None or any(f is None for f in factors):
                continue
            numbers = [to_number(f.value) for f in factors]
            if any(n is None for n in numbers):
                continue
            expected = 1.0
            for number in numbers:
                expected *= number
            actual = to_number(result.value)
            if actual is None:
                continue
            difference = expected - actual
            if abs(difference) <= rule.tolerance:
                continue
            detail = " × ".join(f"{f.name.split(':')[-1]} {f.value}" for f in factors)
            findings.append(
                Finding(
                    kind="sum_mismatch",
                    severity=Severity.CRITICAL,
                    message=(
                        f"明细行勾稽不符：{label} 的 {rule.total} {result.value}，"
                        f"按 {detail} 应为 {expected:,.2f}，差额 {difference:+,.2f}"
                    ),
                    region_ids=[rid for f in factors if f.region_ids for rid in f.region_ids]
                    + list(result.region_ids),
                    field_names=[f.name for f in factors] + [result.name],
                    suggestion="核对这一行的数量、单价与金额",
                )
            )
        return findings

    # rows_sum：整单合计 = 各明细行之和
    total_field = next((f for f in fields if f.name == rule.total), None)
    if total_field is None:
        return findings
    column = rule.components[0] if rule.components else rule.total
    amounts: list[Decimal] = []
    labels: list[str] = []
    for label, columns in groups.items():
        item = columns.get(column)
        if item is None:
            continue
        amount = normalize_money(item.value)
        if amount is None:
            break
        amounts.append(Decimal(amount))
        labels.append(f"{label} {item.value}")
    if len(amounts) != len(groups) or not amounts:
        return findings
    expected = sum(amounts, Decimal("0"))
    actual = normalize_money(total_field.value)
    if actual is None:
        return findings
    difference = float(expected - Decimal(actual))
    if abs(difference) <= rule.tolerance:
        return findings
    findings.append(
        Finding(
            kind="sum_mismatch",
            severity=Severity.CRITICAL,
            message=(
                f"明细合计不符：{rule.total} {total_field.value}，"
                f"各明细相加为 {expected:,.2f}，差额 {difference:+,.2f}"
                f"（{'、'.join(labels)}）"
            ),
            region_ids=[
                rid for item in groups.values() for f in item.values() for rid in f.region_ids
            ]
            + list(total_field.region_ids),
            field_names=[total_field.name],
            suggestion="核对明细行与合计",
        )
    )
    return findings


def findings_from_reconciliation(result: ReconciliationResult) -> list[Finding]:
    """把对账结果转成账本里的 Finding。"""

    findings: list[Finding] = []
    for mismatch in result.mismatches:
        region_ids: list[str] = []
        field_names: list[str] = []
        for entry in mismatch.values:
            region_ids.extend(entry.get("region_ids", []))
            field_names.append(entry.get("field") or entry["doc_id"])
        findings.append(
            Finding(
                kind="contradiction",
                severity=Severity.CRITICAL if mismatch.kind in ("money", "entity") else Severity.HIGH,
                message=f"跨文档矛盾：{mismatch.note}",
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
