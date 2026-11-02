"""测试用的构造工具。"""

from __future__ import annotations

from tracecell.types import BBox, Candidate, Category, Region, Status, Verdict


def make_region(
    region_id: str = "r000",
    text: str = "价税合计（小写）¥145,318.00",
    bbox: BBox | None = None,
    category: Category = Category.TEXT,
    candidates: list[Candidate] | None = None,
    status: Status = Status.PENDING,
) -> Region:
    """造一个区域。默认带一个高分的 primary 候选。"""

    if candidates is None:
        candidates = [
            Candidate(text=text, origin="primary", score=0.97, verdict=Verdict.SCORED)
        ]
    return Region(
        id=region_id,
        page=1,
        bbox=bbox or BBox(0.1, 0.2, 0.6, 0.25),
        category=category,
        text=text,
        candidates=candidates,
        status=status,
    )
