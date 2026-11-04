"""基线对比的测试。

这条对比是作品价值的量化证明：人工花多少力气，漏多少错误。所以测试重点
是"漏检"的判定不能出错——尤其是中文大写金额的写法差异。
"""

from __future__ import annotations

from tracecell.baseline import ALL_HUMAN, OCR_ONLY, TRACECELL, _same_reading, compare
from tracecell.types import (
    BBox,
    Candidate,
    Category,
    DocumentLedger,
    Region,
    Verdict,
)


def make_ledger(
    doc_id: str,
    regions: list[tuple[str, str, str, float]],
    queue: list[str] | None = None,
) -> DocumentLedger:
    """regions 是 (区域id, primary读数, 其它候选读数, primary分数)。"""

    ledger = DocumentLedger(doc_id=doc_id, source=f"{doc_id}.png")
    for region_id, primary, other, score in regions:
        candidates = [
            Candidate(text=primary, origin="primary", score=score, verdict=Verdict.SCORED)
        ]
        if other:
            candidates.append(
                Candidate(text=other, origin="reparse-16x", score=0.4, verdict=Verdict.SCORED)
            )
        ledger.regions.append(
            Region(
                id=region_id,
                page=1,
                bbox=BBox(0.1, 0.1, 0.6, 0.2),
                category=Category.TEXT,
                text=primary,
                candidates=candidates,
            )
        )
    ledger.review_queue = queue or []
    return ledger


class TestSameReading:
    def test_identical_readings(self):
        assert _same_reading("金额 100.00", "金额 100.00")

    def test_chinese_amount_variants_are_the_same_number(self):
        """「圆整」和「元整」是同一个数，不能算成识别错误。"""

        assert _same_reading("合同金额（大写）：伍拾捌万圆整", "合同金额（大写）：伍拾捌万元整")

    def test_different_numbers_are_different(self):
        assert not _same_reading("伍拾捌万元整", "伍拾柒万捌仟元整")

    def test_different_text_is_different(self):
        assert not _same_reading("南京海川软件股份有限公司", "南京海川软件股分有限公司")


class TestCompare:
    def test_ocr_only_misses_a_confident_wrong_reading(self):
        ledger = make_ledger("d1", [("r000", "南京海川软件股分有限公司", "南京海川软件股份有限公司", 0.91)])
        truths = {("d1", "r000"): "南京海川软件股份有限公司"}
        report = compare([ledger], truths, document_errors=0)
        ocr = next(r for r in report.results if r.name == OCR_ONLY)
        assert ocr.missed_reading_errors == 1
        assert ocr.human_cells == 0

    def test_tracecell_catches_it_by_sending_it_to_review(self):
        ledger = make_ledger(
            "d1",
            [("r000", "南京海川软件股分有限公司", "南京海川软件股份有限公司", 0.91)],
            queue=["r000"],
        )
        truths = {("d1", "r000"): "南京海川软件股份有限公司"}
        report = compare([ledger], truths, document_errors=0)
        tracecell = next(r for r in report.results if r.name == TRACECELL)
        assert tracecell.human_cells == 1
        assert tracecell.missed_reading_errors == 0

    def test_a_wrong_reading_queued_for_review_is_not_a_miss(self):
        """进了人工队列的就不算漏检——人看了就会改。"""

        ledger = make_ledger(
            "d1",
            [("r000", "错的", "对的", 0.9)],
            queue=["r000"],
        )
        truths = {("d1", "r000"): "对的"}
        report = compare([ledger], truths, document_errors=0)
        tracecell = next(r for r in report.results if r.name == TRACECELL)
        assert tracecell.missed_reading_errors == 0

    def test_all_human_never_misses(self):
        ledger = make_ledger("d1", [("r000", "错的", "对的", 0.9)])
        report = compare([ledger], {("d1", "r000"): "对的"}, document_errors=5)
        human = next(r for r in report.results if r.name == ALL_HUMAN)
        assert human.human_cells == 1
        assert human.missed_reading_errors == 0
        assert human.missed_document_errors == 0

    def test_document_errors_only_hit_the_ocr_baseline(self):
        ledger = make_ledger("d1", [("r000", "对的", "", 0.95)])
        report = compare([ledger], {("d1", "r000"): "对的"}, document_errors=3)
        ocr = next(r for r in report.results if r.name == OCR_ONLY)
        tracecell = next(r for r in report.results if r.name == TRACECELL)
        assert ocr.missed_document_errors == 3
        assert tracecell.missed_document_errors == 0

    def test_render_is_readable(self):
        ledger = make_ledger("d1", [("r000", "对的", "", 0.95)])
        text = compare([ledger], {("d1", "r000"): "对的"}, document_errors=1).render()
        assert "人工复核" in text
        assert "全量人工" in text
        assert "只信 OCR" in text
        assert "TraceCell" in text
