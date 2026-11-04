"""人工复核回流。

现在这条链是断的：系统把 8 格标成"待人工"，人看完之后呢？结果没有回到
系统里，下一轮还是同样的判断，同样的预算，同样的待人工清单。

回流之后会变成什么样：

    系统标出待人工 → 人在复核清单上标注对/错 → 标注回灌
      ├─ 确认对的区域，直接放行，置信度提到 1.0，不再重复复核
      ├─ 确认错的区域，纠正后的读数作为新候选进入账本
      └─ 人工标注自动变成标定样本，可以直接喂给 tracecell calibrate

最后一条是关键：人工复核花掉的力气，本来就该沉淀成阈值标定的数据，
不然每次都要重新标一遍。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .calibration import CalibrationSample
from .types import Candidate, DocumentLedger, Finding, Severity, Status, Verdict

CORRECT = "correct"
WRONG = "wrong"
UNREADABLE = "unreadable"


@dataclass
class ReviewFeedback:
    """一条人工复核结论。"""

    doc_id: str
    region_id: str
    verdict: str = CORRECT
    corrected_text: str = ""
    reviewer: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewFeedback:
        return cls(
            doc_id=data["doc_id"],
            region_id=data["region_id"],
            verdict=data.get("verdict", CORRECT),
            corrected_text=data.get("corrected_text", ""),
            reviewer=data.get("reviewer", ""),
            note=data.get("note", ""),
        )


def load_feedback(path: str | Path) -> list[ReviewFeedback]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("feedback", data) if isinstance(data, dict) else data
    return [ReviewFeedback.from_dict(item) for item in items]


def save_feedback(path: str | Path, items: list[ReviewFeedback]) -> None:
    Path(path).write_text(
        json.dumps([item.to_dict() for item in items], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def apply_feedback(ledger: DocumentLedger, feedback: ReviewFeedback) -> DocumentLedger:
    """把一条复核结论应用到账本上。"""

    region = ledger.region(feedback.region_id)
    if region is None:
        raise KeyError(f"账本 {ledger.doc_id} 里没有区域 {feedback.region_id}")

    if feedback.verdict == CORRECT:
        best = region.best
        if best is not None:
            best.meta["human_verified"] = True
            best.meta["reviewer"] = feedback.reviewer
            best.score = 1.0
        region.status = Status.ACCEPTED
        region.note = f"人工确认无误（{feedback.reviewer or '复核人'}）"
        for fld in ledger.fields:
            if region.id in fld.region_ids:
                fld.status = Status.ACCEPTED
                fld.confidence = 1.0
                fld.note = "人工确认"

    elif feedback.verdict == WRONG:
        if feedback.corrected_text:
            region.candidates.append(
                Candidate(
                    text=feedback.corrected_text,
                    origin="human",
                    score=1.0,
                    verdict=Verdict.SCORED,
                    detail=f"人工纠正（{feedback.reviewer or '复核人'}）",
                )
            )
            region.status = Status.REPAIRED
            region.note = f"人工纠正为「{feedback.corrected_text}」"
            for fld in ledger.fields:
                if region.id in fld.region_ids and fld.value in region.text:
                    fld.value = feedback.corrected_text
                    fld.status = Status.ACCEPTED
                    fld.confidence = 1.0
                    fld.note = "人工纠正"
        else:
            region.status = Status.PENDING
            region.note = "人工判定有误但未给出正确读数，仍需处理"
        ledger.findings.append(
            Finding(
                kind="human_review",
                severity=Severity.MEDIUM,
                message=f"人工复核判定区域 {region.id} 有误：{feedback.note or '未说明'}",
                region_ids=[region.id],
                suggestion="按纠正后的读数更新账本",
            )
        )

    else:  # UNREADABLE
        region.status = Status.PENDING
        region.note = "人工判定无法辨认，需重新采集"
        ledger.findings.append(
            Finding(
                kind="human_review",
                severity=Severity.HIGH,
                message=f"人工复核判定区域 {region.id} 无法辨认：{feedback.note or '未说明'}",
                region_ids=[region.id],
                suggestion="重新采集这一块的高质量图像",
            )
        )

    ledger.meta.setdefault("human_review", []).append(feedback.to_dict())
    return ledger


def apply_all(ledger: DocumentLedger, items: list[ReviewFeedback]) -> DocumentLedger:
    for item in items:
        if item.doc_id == ledger.doc_id:
            apply_feedback(ledger, item)
    return ledger


def to_labeled_samples(
    items: list[ReviewFeedback], ledgers: list[DocumentLedger]
) -> list[CalibrationSample]:
    """把人工标注变成标定样本。

    这是这条闭环最值钱的地方：人工复核的每一格都带一个可信的标签，
    直接就是阈值标定要的数据，不用另外标一遍。
    """

    samples: list[CalibrationSample] = []
    by_doc = {ledger.doc_id: ledger for ledger in ledgers}
    for item in items:
        ledger = by_doc.get(item.doc_id)
        if ledger is None:
            continue
        region = ledger.region(item.region_id)
        if region is None:
            continue
        for candidate in region.candidates:
            if candidate.origin == "human":
                continue
            if not candidate.scored:
                continue
            # 人工说这一整块都错了，那这个候选就是错的；说对了则是对的。
            correct = item.verdict == CORRECT
            samples.append(
                CalibrationSample(
                    score=float(candidate.score),
                    correct=correct,
                    category=region.category.value,
                    note=f"{item.doc_id}:{item.region_id}:{candidate.origin}",
                )
            )
    return samples


def feedback_stats(items: list[ReviewFeedback]) -> dict[str, Any]:
    total = len(items)
    correct = sum(1 for item in items if item.verdict == CORRECT)
    wrong = sum(1 for item in items if item.verdict == WRONG)
    unreadable = sum(1 for item in items if item.verdict == UNREADABLE)
    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "unreadable": unreadable,
        "accuracy": round(correct / total, 4) if total else None,
        "reviewers": sorted({item.reviewer for item in items if item.reviewer}),
    }
