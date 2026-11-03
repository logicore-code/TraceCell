"""核验闭环与账本导出的端到端测试。

这里用一个"假模型"驱动整条链路：解析器返回固定区域，评分器按候选内容
返回固定分数。目的是验证策略：什么时候重解析、什么时候升级、什么时候
转人工、最后账本长什么样。模型本身不测。
"""

from __future__ import annotations

import pytest

from tracecell.agent.loop import VerificationLoop
from tracecell.extractors import RegexFieldExtractor, schema_by_name
from tracecell.ledger import build_ledger, export_csv, export_markdown, match_region_for_value
from tracecell.policy import BudgetPolicy, RiskPolicy
from tracecell.providers.base import PageParse, ScoreResult
from tracecell.types import BBox, Candidate, Category, Region, Verdict


class FakeParser:
    """按区域序号返回不同文本，制造候选冲突。"""

    def __init__(self) -> None:
        self.calls = 0

    def parse_page(self, image: bytes, profile=None) -> PageParse:
        regions = [
            Region(
                id="r000",
                page=1,
                bbox=BBox(0.1, 0.1, 0.5, 0.15),
                category=Category.TEXT,
                text="价税合计（小写）¥145,318.00",
            ),
            Region(
                id="r001",
                page=1,
                bbox=BBox(0.1, 0.2, 0.5, 0.25),
                category=Category.TEXT,
                text="开票日期 2026年10月18日",
            ),
        ]
        return PageParse(regions=regions, markdown="价税合计（小写）¥145,318.00\n\n开票日期 2026年10月18日")

    def parse_region(self, image: bytes, bbox=None, profile=None) -> str:
        self.calls += 1
        return "价税合计（小写）¥145,318.00"


class FakeScorer:
    """给"错"的候选打低分，其余给高分。"""

    def __init__(self, wrong_text: str = "") -> None:
        self.wrong_text = wrong_text
        self.calls = 0

    def score(self, image: bytes, text: str, category: str = "text") -> ScoreResult:
        self.calls += 1
        if self.wrong_text and self.wrong_text in text:
            return ScoreResult(0.18, Verdict.SCORED)
        return ScoreResult(0.96, Verdict.SCORED)

    def score_best(self, image: bytes, candidates, category: str = "text"):
        return [self.score(image, text, category) for text in candidates]


class FakeEscalator:
    def __init__(self) -> None:
        self.calls = 0

    def review(self, image: bytes, question: str, hint: str = "") -> str:
        self.calls += 1
        return "价税合计（小写）¥145,318.00"


def build_loop(scorer=None, escalator=None, policy=None) -> VerificationLoop:
    return VerificationLoop(
        parser=FakeParser(),
        scorer=scorer or FakeScorer(),
        escalator=escalator,
        policy=policy or BudgetPolicy(),
        risk_policy=RiskPolicy(),
        extractor=RegexFieldExtractor(schema_by_name("invoice")),
    )


class TestVerificationLoop:
    def test_clean_document_needs_no_repair(self):
        loop = build_loop()
        report = loop.run(b"", "doc-1", "doc-1.png")
        ledger = report.ledger
        assert ledger.verdict in ("auto_pass", "spot_check")
        assert report.extra_calls == 0

    def test_fields_are_extracted_and_bound_to_regions(self):
        ledger = build_loop().run(b"", "doc-1", "doc-1.png").ledger
        total = ledger.field("价税合计")
        assert total is not None
        assert total.value == "145,318.00"
        assert total.region_ids == ["r000"]

    def test_low_score_triggers_additional_work(self):
        """评分低的区域必须被追查，不能直接放行。"""

        class LowScorer(FakeScorer):
            def score(self, image, text, category="text"):
                self.calls += 1
                if "价税合计" in text:
                    return ScoreResult(0.12, Verdict.SCORED)
                return ScoreResult(0.96, Verdict.SCORED)

        loop = build_loop(scorer=LowScorer())
        report = loop.run(b"", "doc-1", "doc-1.png")
        assert report.extra_calls > 0
        assert report.ledger.budget.total_cost > 0

    def test_budget_is_respected(self):
        policy = BudgetPolicy(total_budget=1.0)
        loop = build_loop(policy=policy)
        report = loop.run(b"", "doc-1", "doc-1.png")
        assert report.ledger.budget.extra_cost <= policy.total_budget + 0.001
        assert report.ledger.budget.total_cost >= report.ledger.budget.base_cost

    def test_ledger_is_json_serialisable(self):
        ledger = build_loop().run(b"", "doc-1", "doc-1.png").ledger
        restored = type(ledger).from_json(ledger.to_json())
        assert restored.doc_id == ledger.doc_id
        assert len(restored.fields) == len(ledger.fields)


class TestLedgerExport:
    def test_csv_contains_evidence_coordinates(self):
        ledger = build_loop().run(b"", "doc-1", "doc-1.png").ledger
        csv_text = export_csv(ledger)
        assert "价税合计" in csv_text
        assert "置信度" in csv_text

    def test_markdown_report_mentions_the_verdict(self):
        ledger = build_loop().run(b"", "doc-1", "doc-1.png").ledger
        report = export_markdown(ledger)
        assert "证据账本" in report
        assert "字段台账" in report

    def test_value_is_anchored_to_the_right_region(self):
        regions = [
            Region(id="r000", page=1, bbox=BBox(0, 0, 1, 1), category=Category.TEXT, text="无关内容"),
            Region(id="r001", page=1, bbox=BBox(0, 0, 1, 1), category=Category.TEXT, text="金额 145,318.00"),
        ]
        matched = match_region_for_value("145,318.00", regions)
        assert matched is not None
        assert matched.id == "r001"

    def test_unmatched_value_is_not_anchored(self):
        regions = [
            Region(id="r000", page=1, bbox=BBox(0, 0, 1, 1), category=Category.TEXT, text="完全不同的内容")
        ]
        assert match_region_for_value("145,318.00", regions) is None

    def test_findings_flag_contested_regions(self):
        regions = [
            Region(
                id="r000",
                page=1,
                bbox=BBox(0, 0, 1, 1),
                category=Category.TEXT,
                text="金额 100.00",
                candidates=[
                    Candidate(text="金额 100.00", origin="primary", score=0.9, verdict=Verdict.SCORED),
                    Candidate(text="金额 700.00", origin="reparse-16x", score=0.2, verdict=Verdict.SCORED),
                ],
            )
        ]
        ledger = build_ledger("d", "d.png", regions, [], RiskPolicy(), BudgetPolicy())
        kinds = {finding.kind for finding in ledger.findings}
        assert "contested" in kinds
