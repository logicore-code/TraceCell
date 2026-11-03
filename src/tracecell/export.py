"""导出。

一份账本能派生出好几种东西，给不同的人看：

* CSV 台账 —— 财务最常用，一行一个字段
* Excel 工作簿 —— 台账、问题清单、复核清单、候选明细各一个 sheet
* Markdown 复核报告 —— 贴进工单
* JSON —— 给下游系统

Excel 那份是特意做的。单张 CSV 看着方便，但实际用的时候总有人问
"哪些地方要复核""候选读数长什么样"，多 sheet 一次给全。
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from .ledger import export_csv, export_markdown
from .routing import render_review_checklist
from .types import DocumentLedger

__all__ = [
    "export_csv",
    "export_markdown",
    "export_json",
    "export_excel",
    "export_bundle",
]


def export_json(ledger: DocumentLedger, indent: int = 2) -> str:
    """完整账本 JSON。"""

    return ledger.to_json(indent=indent)


def _field_rows(ledger: DocumentLedger) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for fld in ledger.fields:
        region = ledger.region(fld.region_ids[0]) if fld.region_ids else None
        rows.append(
            [
                fld.name,
                fld.value,
                round(fld.confidence, 4),
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
                fld.note,
            ]
        )
    return rows


def _finding_rows(ledger: DocumentLedger) -> list[list[Any]]:
    return [
        [
            finding.kind,
            finding.severity.value,
            finding.message,
            ",".join(finding.region_ids),
            ",".join(finding.field_names),
            finding.suggestion,
        ]
        for finding in ledger.findings
    ]


def _checklist_rows(ledger: DocumentLedger) -> list[list[Any]]:
    region_map = {r.id: r for r in ledger.regions}
    rows: list[list[Any]] = []
    for item in render_review_checklist(region_map, ledger.fields, ledger.review_queue):
        bbox = item.get("bbox") or {}
        rows.append(
            [
                item.get("target", ""),
                item.get("page", ""),
                f"{bbox.get('x0', 0):.4f},{bbox.get('y0', 0):.4f},"
                f"{bbox.get('x1', 0):.4f},{bbox.get('y1', 0):.4f}",
                item.get("why", ""),
                " / ".join(f"{f['name']}={f['value']}" for f in item.get("fields", [])),
                " | ".join(
                    f"{c['text']}（{c['origin']} {c.get('score')}）"
                    for c in item.get("candidates", [])
                ),
            ]
        )
    return rows


def _candidate_rows(ledger: DocumentLedger) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for region in ledger.regions:
        for candidate in region.candidates:
            rows.append(
                [
                    region.id,
                    region.page,
                    region.category.value,
                    candidate.origin,
                    candidate.text,
                    candidate.score,
                    candidate.verdict.value,
                    candidate.detail,
                ]
            )
    return rows


def export_excel(ledger: DocumentLedger) -> bytes:
    """导出一个 Excel 工作簿。

    四个 sheet：字段台账、核验问题、复核清单、候选明细。没有 openpyxl
    的时候退回 CSV，保证功能不因为缺依赖就整个不可用。
    """

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        return export_csv(ledger).encode("utf-8")

    workbook = Workbook()
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2F5597")

    def add_sheet(title: str, header: list[str], rows: list[list[Any]]) -> None:
        sheet = workbook.create_sheet(title=title)
        sheet.append(header)
        for cell in sheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(vertical="center")
        for row in rows:
            sheet.append(row)
        widths = [16, 28, 10, 12, 12, 18, 8, 26, 40, 30]
        for index, width in enumerate(widths[: len(header)], start=1):
            sheet.column_dimensions[chr(64 + index)].width = width
        sheet.freeze_panes = "A2"

    workbook.remove(workbook.active)
    add_sheet(
        "字段台账",
        ["字段名", "字段值", "置信度", "风险等级", "状态", "证据区域", "页码", "坐标", "原文", "备注"],
        _field_rows(ledger),
    )
    add_sheet(
        "核验问题",
        ["类型", "严重级别", "说明", "涉及区域", "涉及字段", "处理建议"],
        _finding_rows(ledger),
    )
    add_sheet(
        "复核清单",
        ["目标", "页码", "坐标", "为什么看", "涉及字段", "候选读数"],
        _checklist_rows(ledger),
    )
    add_sheet(
        "候选明细",
        ["区域", "页码", "类别", "来源", "读数", "Jev 分数", "评分状态", "备注"],
        _candidate_rows(ledger),
    )

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def export_bundle(ledger: DocumentLedger) -> dict[str, bytes]:
    """一次性导出全部格式。"""

    return {
        "ledger.json": export_json(ledger).encode("utf-8"),
        "ledger.csv": export_csv(ledger).encode("utf-8"),
        "ledger.xlsx": export_excel(ledger),
        "report.md": export_markdown(ledger).encode("utf-8"),
    }
