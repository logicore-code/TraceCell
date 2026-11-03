"""批量处理。

真实场景里核验的不是一张单据，是一个文件夹——一整月的进项发票、一批
供应商合同、一个季度的报表。

批量的价值在于汇总：每份单据的问题都是局部事实，几十份单据放在一起
才是"这个供应商的发票一直对不上"这种结论。所以批量跑完除了逐份的账本，
还要给一份汇总。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .types import DocumentLedger

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
DOCUMENT_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}


@dataclass
class BatchItem:
    """批量里的一份文档。"""

    source: str
    doc_id: str
    scenario: str
    ledger: DocumentLedger | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "source": self.source,
            "doc_id": self.doc_id,
            "scenario": self.scenario,
            "error": self.error,
        }
        if self.ledger is not None:
            data.update(
                {
                    "verdict": self.ledger.verdict,
                    "fields": len(self.ledger.fields),
                    "regions": len(self.ledger.regions),
                    "findings": len(self.ledger.findings),
                    "review_queue": len(self.ledger.review_queue),
                    "cost": round(self.ledger.budget.total_cost, 4),
                }
            )
        return data


@dataclass
class BatchSummary:
    """批量汇总。"""

    items: list[BatchItem] = field(default_factory=list)

    @property
    def ok(self) -> list[BatchItem]:
        return [item for item in self.items if item.ledger is not None]

    @property
    def failed(self) -> list[BatchItem]:
        return [item for item in self.items if item.ledger is None]

    def to_dict(self) -> dict[str, Any]:
        by_verdict: dict[str, int] = {}
        for item in self.ok:
            by_verdict[item.ledger.verdict] = by_verdict.get(item.ledger.verdict, 0) + 1
        by_scenario: dict[str, int] = {}
        for item in self.items:
            by_scenario[item.scenario] = by_scenario.get(item.scenario, 0) + 1
        return {
            "total": len(self.items),
            "ok": len(self.ok),
            "failed": len(self.failed),
            "by_verdict": by_verdict,
            "by_scenario": by_scenario,
            "total_cost": round(sum(item.ledger.budget.total_cost for item in self.ok), 4),
            "total_findings": sum(len(item.ledger.findings) for item in self.ok),
            "total_review_queue": sum(len(item.ledger.review_queue) for item in self.ok),
            "items": [item.to_dict() for item in self.items],
        }


def discover(paths: list[str | Path]) -> list[Path]:
    """把路径列表展开成待处理的文件清单。"""

    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.suffix.lower() in DOCUMENT_SUFFIXES:
                    found.append(child)
        elif path.exists():
            found.append(path)
    # 去重，保持顺序
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def guess_scenario(path: Path, default: str = "invoice") -> str:
    """按文件名猜场景类型。

    猜错了也不要紧——文件名只是提示，场景配置决定抽哪些字段，猜错最多
    抽不到字段，不会算错。
    """

    name = path.stem.lower()
    if "purchase_contract" in name or "purchase" in name or "采购" in name:
        return "purchase_contract"
    if "financial_statement" in name or "statement" in name or "balance" in name or "报表" in name:
        return "financial_statement"
    if "invoice" in name or "发票" in name:
        return "invoice"
    if "expense" in name or "报销" in name:
        return "expense"
    if "contract" in name or "合同" in name:
        return "contract"
    return default


def run_batch(
    files: list[Path],
    process: Callable[[Path, str], DocumentLedger],
    default_scenario: str = "invoice",
    jobs: int = 1,
) -> BatchSummary:
    """跑一批文档。

    process 负责真正核验一份文件，批量层只管调度和汇总。一份失败不影响
    其余的——财务的活不能因为一张烂图就停下来。

    jobs > 1 时并发处理。核验的耗时几乎全在模型调用上（网络等待），线程
    就够，不必上多进程。结果顺序跟输入顺序一致。
    """

    summary = BatchSummary()
    scenarios = [guess_scenario(path, default_scenario) for path in files]

    def run_one(index: int) -> BatchItem:
        path = files[index]
        scenario = scenarios[index]
        try:
            ledger = process(path, scenario)
            return BatchItem(
                source=str(path), doc_id=ledger.doc_id, scenario=scenario, ledger=ledger
            )
        except Exception as exc:  # noqa: BLE001 - 单份失败不拖垮整批
            return BatchItem(
                source=str(path), doc_id=path.stem, scenario=scenario, error=str(exc)
            )

    if jobs > 1 and len(files) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            summary.items = list(pool.map(run_one, range(len(files))))
    else:
        summary.items = [run_one(index) for index in range(len(files))]
    return summary


def write_summary(summary: BatchSummary, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
