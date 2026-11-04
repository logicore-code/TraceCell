"""证据账本的组装与导出。

账本是 TraceCell 的核心产物。它跟"输出一段 Markdown"的区别在于：

* 每一个字段都能指回它在原件上的位置（区域 + 坐标）；
* 每一个字段都带着自己的置信度和判定状态；
* 每一次判定都留下了理由和账单。

导出成 CSV 就是一份可以直接交给财务或档案的台账；导出成 Markdown 就是
一份可以贴进工单的复核说明。
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from typing import Any

from .risk import field_confidence, region_confidence, summarize_risk
from .textnorm import normalize, similarity
from .types import DocumentLedger, FieldValue, Finding, Region, Severity, Status

# 值与区域文本的匹配阈值。低于这个相似度就不绑定，宁可让字段"无证据"
# 也不要指错地方。
MATCH_THRESHOLD = 0.55


def match_region_for_value(value: str, regions: Iterable[Region]) -> Region | None:
    """把一个字段值锚定到最可能的那块原文。

    先做归一化子串匹配（最可靠），匹配不上再退到字符 bigram 相似度。
    返回 None 表示找不到可对应的原文。
    """

    normalized_value = normalize(value)
    if not normalized_value:
        return None

    best: tuple[float, Region] | None = None
    for region in regions:
        text = normalize(region.selected_text or region.text)
        if not text:
            continue
        if normalized_value in text:
            # 命中子串：长度占比越高越可能是这块。
            ratio = len(normalized_value) / len(text)
            score = 0.9 + 0.1 * min(ratio, 1.0)
        else:
            score = similarity(value, region.selected_text or region.text)
        if score >= MATCH_THRESHOLD and (best is None or score > best[0]):
            best = (score, region)

    return best[1] if best else None


def bind_fields(
    fields: list[FieldValue], regions: dict[str, Region], policy, scenario=None
) -> list[FieldValue]:
    """给字段补上证据区域，并按悲观继承算置信度。"""

    ordered = list(regions.values())
    for fld in fields:
        if not fld.region_ids:
            region = match_region_for_value(fld.value, ordered)
            if region is not None:
                fld.region_ids = [region.id]
        fld.severity = policy.severity_for(fld.name, fld.value, scenario)
        fld.confidence = field_confidence(fld, regions)
        if fld.status is Status.PENDING and fld.confidence >= 0.9 and fld.region_ids:
            fld.status = Status.ACCEPTED
        elif not fld.region_ids:
            fld.status = Status.PENDING
    return fields


def collect_findings(
    fields: list[FieldValue],
    regions: dict[str, Region],
    budget_policy,
) -> list[Finding]:
    """把核验过程中的问题整理成清单。

    阈值来自 BudgetPolicy：容忍度决定"多小的期望损失可以放过"，
    升级阈值决定"低到多少就值得升级到多模态模型"。
    """

    findings: list[Finding] = []

    for region in regions.values():
        scored = [c for c in region.candidates if c.scored]
        unscored = [c for c in region.candidates if not c.scored]
        if unscored and scored:
            findings.append(
                Finding(
                    kind="unscored_candidate",
                    severity=Severity.LOW,
                    message=f"区域 {region.id} 有 {len(unscored)} 个候选未参与评分",
                    region_ids=[region.id],
                    suggestion="补一次评分可以让择优更充分",
                )
            )
        texts = {normalize(c.text) for c in scored}
        if len(scored) >= 2 and len(texts) > 1:
            findings.append(
                Finding(
                    kind="contested",
                    severity=Severity.HIGH,
                    message=f"区域 {region.id} 的候选结果不一致：{' / '.join(sorted(texts)[:3])}",
                    region_ids=[region.id],
                    suggestion="需要人工确认，或升级到多模态模型复核",
                )
            )
        if region.status is Status.PENDING and scored:
            best = max(c.score or 0.0 for c in scored)
            if best < budget_policy.escalate_threshold:
                findings.append(
                    Finding(
                        kind="low_score",
                        severity=Severity.HIGH,
                        message=f"区域 {region.id} 的最佳候选得分仅 {best:.3f}",
                        region_ids=[region.id],
                        suggestion="升级复核或转人工",
                    )
                )

    for fld in fields:
        if not fld.region_ids:
            findings.append(
                Finding(
                    kind="no_evidence",
                    severity=Severity.MEDIUM,
                    message=f"字段「{fld.name}」没有找到对应的原文区域",
                    field_names=[fld.name],
                    suggestion="检查抽取逻辑，或人工补录来源",
                )
            )
    return findings


def build_ledger(
    doc_id: str,
    source: str,
    regions: list[Region],
    fields: list[FieldValue],
    risk_policy,
    budget_policy,
    markdown: str = "",
    pages: int = 1,
    meta: dict[str, Any] | None = None,
    scenario=None,
) -> DocumentLedger:
    """组装一份账本。"""

    region_map = {r.id: r for r in regions}
    bound = bind_fields(fields, region_map, risk_policy, scenario)
    findings = collect_findings(bound, region_map, budget_policy)
    return DocumentLedger(
        doc_id=doc_id,
        source=source,
        pages=pages,
        regions=regions,
        fields=bound,
        findings=findings,
        markdown=markdown,
        meta=meta or {},
    )


def export_csv(ledger: DocumentLedger) -> str:
    """导出台账 CSV。

    一行一个字段，附带证据坐标和置信度。这张表可以直接进 Excel 做
    二次核对，也可以作为下游系统的输入。
    """

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "字段名",
            "字段值",
            "置信度",
            "风险等级",
            "状态",
            "证据区域",
            "页码",
            "坐标(x0,y0,x1,y1)",
            "原文",
            "候选数",
            "备注",
        ]
    )
    for fld in ledger.fields:
        region = ledger.region(fld.region_ids[0]) if fld.region_ids else None
        writer.writerow(
            [
                fld.name,
                fld.value,
                f"{fld.confidence:.4f}",
                fld.severity.value,
                fld.status.value,
                ",".join(fld.region_ids),
                region.page if region else "",
                (
                    f"{region.bbox.x0:.4f},{region.bbox.y0:.4f},"
                    f"{region.bbox.x1:.4f},{region.bbox.y1:.4f}"
                )
                if region
                else "",
                (region.selected_text or "").replace("\n", " ") if region else "",
                len(region.candidates) if region else 0,
                fld.note,
            ]
        )
    return buffer.getvalue()


def export_markdown(ledger: DocumentLedger) -> str:
    """导出一份可读的复核报告。"""

    lines: list[str] = []
    lines.append(f"# 证据账本 · {ledger.doc_id}")
    lines.append("")
    lines.append(f"- 来源：`{ledger.source}`，共 {ledger.pages} 页")
    lines.append(f"- 路由结论：**{ledger.verdict}**")
    budget = ledger.budget
    lines.append(
        f"- 核验成本：{budget.total_cost:.2f} rcu，额外调用 {budget.extra_calls} 次，"
        f"转人工 {budget.human_cells} 格"
    )
    lines.append(
        f"- 区域状态：放行 {budget.accepted_regions}、修正 {budget.repaired_regions}、"
        f"升级 {budget.escalated_regions}、待人工 {budget.pending_regions}"
    )
    lines.append("")

    lines.append("## 字段台账")
    lines.append("")
    lines.append("| 字段 | 值 | 置信度 | 风险 | 状态 | 证据位置 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for fld in ledger.fields:
        region = ledger.region(fld.region_ids[0]) if fld.region_ids else None
        where = f"第{region.page}页 {region.id}" if region else "—"
        lines.append(
            f"| {fld.name} | {fld.value} | {fld.confidence:.3f} | "
            f"{fld.severity.value} | {fld.status.value} | {where} |"
        )
    lines.append("")

    if ledger.findings:
        lines.append("## 发现的问题")
        lines.append("")
        for finding in ledger.findings:
            lines.append(f"- **[{finding.severity.value}]** {finding.message}")
            if finding.suggestion:
                lines.append(f"  - 建议：{finding.suggestion}")
        lines.append("")

    if ledger.decisions:
        lines.append("## 核验决策")
        lines.append("")
        lines.append("| 区域 | 动作 | 置信度变化 | 风险下降 | 成本 | 理由 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for decision in ledger.decisions:
            lines.append(
                f"| {decision.region_id} | {decision.action} | "
                f"{decision.confidence_before:.3f} → {decision.confidence_after:.3f} | "
                f"{max(0.0, decision.risk_before - decision.risk_after):.4f} | "
                f"{decision.cost:.2f} | {decision.reason} |"
            )
        lines.append("")

    return "\n".join(lines)


def ledger_summary(ledger: DocumentLedger) -> dict[str, Any]:
    """给界面顶部用的摘要。"""

    region_map = {r.id: r for r in ledger.regions}
    summary = summarize_risk(ledger.fields, region_map)
    summary.update(
        {
            "doc_id": ledger.doc_id,
            "verdict": ledger.verdict,
            "pages": ledger.pages,
            "region_count": len(ledger.regions),
            "field_count": len(ledger.fields),
            "finding_count": len(ledger.findings),
            "review_queue_size": len(ledger.review_queue),
            "budget": ledger.budget.to_dict(),
            "mean_region_confidence": (
                sum(region_confidence(r) for r in ledger.regions) / len(ledger.regions)
                if ledger.regions
                else 0.0
            ),
        }
    )
    return summary
