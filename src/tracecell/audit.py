"""审计轨迹。

一份账本说明了"结论是什么"，但没说明"这个结论是怎么来的"。出了问题
要复盘时，需要知道：哪一格花了多少推理、走了哪条决策、为什么从 0.53
变成 0.88。

TraceCell 的账本里其实已经有这些散落的信息（决策流水、成本账单、候选
明细），但它们是分开的。审计轨迹把它们按时间串成一条线，可以整份导出。
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AuditEvent:
    """一条轨迹记录。"""

    kind: str
    doc_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["at"] = round(self.at, 3)
        return data


class AuditLog:
    """轨迹收集。"""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def record(self, kind: str, doc_id: str = "", **detail: Any) -> None:
        self.events.append(AuditEvent(kind=kind, doc_id=doc_id, detail=detail))

    def extend(self, events: list[AuditEvent]) -> None:
        self.events.extend(events)

    def by_kind(self, kind: str) -> list[AuditEvent]:
        return [event for event in self.events if event.kind == kind]

    def cost_of(self, doc_id: str) -> float:
        return sum(
            float(event.detail.get("cost", 0.0))
            for event in self.events
            if event.doc_id == doc_id and "cost" in event.detail
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            {"events": [event.to_dict() for event in self.events]},
            ensure_ascii=False,
            indent=indent,
        )

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["时间", "类型", "文档", "详情"])
        for event in self.events:
            writer.writerow(
                [
                    f"{event.at:.3f}",
                    event.kind,
                    event.doc_id,
                    json.dumps(event.detail, ensure_ascii=False),
                ]
            )
        return buffer.getvalue()

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        return {"events": len(self.events), "by_kind": counts}
