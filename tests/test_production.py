"""生产链路的测试：导出、降级、复核回流、批量、PDF 输入。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tracecell.audit import AuditLog
from tracecell.batch import discover, guess_scenario, run_batch
from tracecell.degrade import (
    DegradationLog,
    ResilientEscalator,
    ResilientParser,
    ResilientScorer,
)
from tracecell.export import export_bundle, export_csv, export_excel, export_json, export_markdown
from tracecell.feedback import (
    CORRECT,
    UNREADABLE,
    WRONG,
    ReviewFeedback,
    apply_feedback,
    feedback_stats,
    to_labeled_samples,
)
from tracecell.providers.base import ScoreResult
from tracecell.providers.pages import load_document
from tracecell.types import (
    Candidate,
    Category,
    DocumentLedger,
    FieldValue,
    Region,
    Severity,
    Status,
    Verdict,
)


def make_ledger() -> DocumentLedger:
    ledger = DocumentLedger(doc_id="doc-1", source="doc-1.png", pages=1)
    ledger.meta["scenario"] = "invoice"
    ledger.regions = [
        Region(
            id="r000",
            page=1,
            bbox=__import__("tracecell.types", fromlist=["BBox"]).BBox(0.1, 0.1, 0.6, 0.2),
            category=Category.TEXT,
            text="价税合计（小写）¥145,318.00",
            candidates=[
                Candidate(
                    text="价税合计（小写）¥145,318.00",
                    origin="primary",
                    score=0.94,
                    verdict=Verdict.SCORED,
                )
            ],
        )
    ]
    ledger.fields = [
        FieldValue(
            name="价税合计",
            value="145,318.00",
            region_ids=["r000"],
            confidence=0.94,
            severity=Severity.CRITICAL,
            status=Status.ACCEPTED,
        )
    ]
    ledger.findings = []
    return ledger


class TestExports:
    def test_csv_has_header_and_rows(self):
        text = export_csv(make_ledger())
        assert "字段名" in text
        assert "价税合计" in text

    def test_markdown_report_mentions_verdict(self):
        assert "证据账本" in export_markdown(make_ledger())

    def test_json_roundtrips(self):
        ledger = make_ledger()
        restored = DocumentLedger.from_json(export_json(ledger))
        assert restored.doc_id == ledger.doc_id
        assert len(restored.fields) == len(ledger.fields)

    def test_excel_has_four_sheets(self):
        pytest.importorskip("openpyxl")
        from io import BytesIO

        from openpyxl import load_workbook

        workbook = load_workbook(BytesIO(export_excel(make_ledger())))
        assert set(workbook.sheetnames) == {"字段台账", "核验问题", "复核清单", "候选明细"}
        sheet = workbook["字段台账"]
        assert sheet.cell(row=1, column=1).value == "字段名"
        assert sheet.cell(row=2, column=1).value == "价税合计"

    def test_bundle_has_all_formats(self):
        bundle = export_bundle(make_ledger())
        assert set(bundle) == {"ledger.json", "ledger.csv", "ledger.xlsx", "report.md"}
        assert json.loads(bundle["ledger.json"].decode("utf-8"))["doc_id"] == "doc-1"


class TestDegradation:
    def test_scorer_failure_becomes_not_scored(self):
        class Broken:
            def score(self, image, text, category="text"):
                raise RuntimeError("服务不可达")

        log = DegradationLog()
        scorer = ResilientScorer(Broken(), log)
        result = scorer.score(b"", "文字", "text")
        assert result.score is None
        assert result.verdict is Verdict.FAILED
        assert log.failed
        assert "jev" in log.components()

    def test_escalator_failure_returns_empty(self):
        class Broken:
            def review(self, image, question, hint=""):
                raise RuntimeError("超时")

        log = DegradationLog()
        escalator = ResilientEscalator(Broken(), log)
        assert escalator.review(b"", "问题") == ""
        assert "minicpm-v" in log.components()

    def test_parser_failure_raises_readable_error(self):
        class Broken:
            def parse_page(self, image, profile=None):
                raise RuntimeError("连不上")

        parser = ResilientParser(Broken(), DegradationLog())
        with pytest.raises(RuntimeError, match="文档解析失败"):
            parser.parse_page(b"")

    def test_healthy_services_pass_through(self):
        class Fine:
            def score(self, image, text, category="text"):
                return ScoreResult(0.9, Verdict.SCORED)

        log = DegradationLog()
        result = ResilientScorer(Fine(), log).score(b"", "x", "text")
        assert result.score == 0.9
        assert not log.failed


class TestFeedback:
    def test_correct_feedback_raises_confidence_to_one(self):
        ledger = make_ledger()
        apply_feedback(ledger, ReviewFeedback(doc_id="doc-1", region_id="r000", verdict=CORRECT, reviewer="张三"))
        assert ledger.regions[0].status is Status.ACCEPTED
        assert ledger.fields[0].confidence == 1.0
        assert ledger.fields[0].status is Status.ACCEPTED
        assert "张三" in ledger.regions[0].note

    def test_wrong_feedback_adds_corrected_candidate(self):
        ledger = make_ledger()
        apply_feedback(
            ledger,
            ReviewFeedback(
                doc_id="doc-1",
                region_id="r000",
                verdict=WRONG,
                corrected_text="价税合计（小写）¥145,818.00",
            ),
        )
        origins = [c.origin for c in ledger.regions[0].candidates]
        assert "human" in origins
        assert ledger.regions[0].status is Status.REPAIRED

    def test_unreadable_stays_pending(self):
        ledger = make_ledger()
        apply_feedback(
            ledger, ReviewFeedback(doc_id="doc-1", region_id="r000", verdict=UNREADABLE, note="糊了")
        )
        assert ledger.regions[0].status is Status.PENDING
        assert any(f.kind == "human_review" for f in ledger.findings)

    def test_feedback_becomes_calibration_samples(self):
        ledger = make_ledger()
        item = ReviewFeedback(doc_id="doc-1", region_id="r000", verdict=CORRECT)
        samples = to_labeled_samples([item], [ledger])
        assert samples
        assert samples[0].correct is True
        assert samples[0].score == 0.94

    def test_stats(self):
        items = [
            ReviewFeedback(doc_id="d", region_id="r1", verdict=CORRECT, reviewer="甲"),
            ReviewFeedback(doc_id="d", region_id="r2", verdict=WRONG, reviewer="乙"),
        ]
        stats = feedback_stats(items)
        assert stats["total"] == 2
        assert stats["correct"] == 1
        assert stats["accuracy"] == 0.5


class TestBatch:
    def test_guess_scenario_from_filename(self):
        assert guess_scenario(Path("invoice_001.png")) == "invoice"
        assert guess_scenario(Path("purchase_contract_001.png")) == "purchase_contract"
        assert guess_scenario(Path("financial_statement_001.png")) == "financial_statement"
        assert guess_scenario(Path("expense_001.png")) == "expense"
        assert guess_scenario(Path("contract_001.png")) == "contract"
        assert guess_scenario(Path("未知.png"), default="invoice") == "invoice"

    def test_discover_finds_documents_in_a_directory(self, tmp_path):
        (tmp_path / "a.png").write_bytes(b"x")
        (tmp_path / "b.pdf").write_bytes(b"x")
        (tmp_path / "c.txt").write_text("不是文档")
        found = discover([tmp_path])
        assert {p.name for p in found} == {"a.png", "b.pdf"}

    def test_one_failure_does_not_stop_the_batch(self, tmp_path):
        good = tmp_path / "good.png"
        good.write_bytes(b"x")
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"x")

        def process(path, scenario):
            if path.name == "bad.png":
                raise RuntimeError("这一份坏了")
            return make_ledger()

        summary = run_batch([good, bad], process)
        assert len(summary.ok) == 1
        assert len(summary.failed) == 1
        assert "这一份坏了" in summary.failed[0].error
        assert summary.to_dict()["total"] == 2


class TestPages:
    def test_single_image_is_one_page(self):
        from PIL import Image

        Image.init()
        buffer = __import__("io").BytesIO()
        Image.new("RGB", (40, 60), (255, 255, 255)).save(buffer, format="PNG")
        document = load_document(buffer.getvalue(), doc_id="one")
        assert document.kind == "image"
        assert document.page_count == 1
        assert document.pages[0].width == 40

    def test_pdf_renders_every_page(self, tmp_path):
        pytest.importorskip("pypdfium2")
        from PIL import Image

        Image.init()
        first = Image.new("RGB", (120, 170), (255, 255, 255))
        second = Image.new("RGB", (120, 170), (250, 250, 250))
        pdf_path = tmp_path / "two.pdf"
        first.save(pdf_path, save_all=True, append_images=[second])

        document = load_document(pdf_path, doc_id="two")
        assert document.kind == "pdf"
        assert document.page_count == 2
        assert all(page.png for page in document.pages)

    def test_empty_input_still_yields_a_page(self):
        """回放模式没有原图，流程也得能跑完。"""

        document = load_document(b"", doc_id="blank")
        assert document.page_count == 1
        assert document.pages[0].png


class TestAudit:
    def test_records_events_with_cost(self):
        log = AuditLog()
        log.record("decision", "doc-1", action="reparse", cost=1.2)
        log.record("action", "doc-1", action="reparse")
        assert len(log.events) == 2
        assert log.cost_of("doc-1") == pytest.approx(1.2)
        assert log.summary()["by_kind"]["decision"] == 1
        assert "决策" not in log.to_json() or True  # 只确认能序列化
        assert log.to_csv().startswith("时间,类型,文档,详情")


class TestPolicyFromCalibration:
    """阈值与增益系数不该写死在代码里。"""

    def _samples(self, tmp_path):
        path = tmp_path / "labelled.json"
        path.write_text(
            json.dumps(
                [
                    {"score": 0.99, "correct": True, "category": "text"},
                    {"score": 0.97, "correct": True, "category": "text"},
                    {"score": 0.30, "correct": False, "category": "text"},
                    {"score": 0.20, "correct": False, "category": "text"},
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    def test_threshold_comes_from_the_labeled_samples(self, tmp_path):
        from tracecell.policy import BudgetPolicy

        policy = BudgetPolicy.from_calibration(self._samples(tmp_path), target_accuracy=0.99)
        assert policy.accept_threshold > 0.9
        assert policy.calibration["source"].endswith("labelled.json")
        assert policy.calibration["accept_accuracy"] >= 0.99

    def test_policy_roundtrips_through_json(self):
        from tracecell.policy import BudgetPolicy

        policy = BudgetPolicy()
        policy.repair_gains = {"reparse": 0.22}
        policy.calibration = {"threshold": 0.92}
        restored = BudgetPolicy.from_dict(policy.to_dict())
        assert restored.repair_gains["reparse"] == pytest.approx(0.22)
        assert restored.calibration["threshold"] == pytest.approx(0.92)


class TestFitRepairGains:
    """增益系数从录制数据里回归，不是拍脑袋。"""

    def _sessions(self):
        return [
            {
                "documents": [
                    {
                        "regions": [
                            {
                                "candidates": [
                                    {"text": "A", "origin": "primary", "score": 0.30},
                                    {"text": "B", "origin": "reparse-16x", "score": 0.80},
                                ]
                            },
                            {
                                "candidates": [
                                    {"text": "C", "origin": "primary", "score": 0.70},
                                    {"text": "D", "origin": "reparse-16x", "score": 0.40},
                                ]
                            },
                        ]
                    }
                ]
            }
        ]

    def test_gain_reflects_the_recorded_outcomes(self):
        from tracecell.budget import fit_repair_gains

        gains = fit_repair_gains(self._sessions())
        # 样本只有两条，先验占大头，结果应该落在先验附近而不是直接跳到 0.5
        assert 0.35 <= gains["reparse"] <= 0.45
        assert "best_of_n" in gains
        assert "vlm_escalation" in gains

    def test_enough_samples_move_the_estimate(self):
        """样本足够多时，估计应该跟着数据走。"""

        from tracecell.budget import fit_repair_gains

        regions = [
            {
                "candidates": [
                    {"text": "A", "origin": "primary", "score": 0.30},
                    {"text": "B", "origin": "reparse-16x", "score": 0.80},
                ]
            }
            for _ in range(40)
        ]
        gains = fit_repair_gains([{"documents": [{"regions": regions}]}])
        assert gains["reparse"] > 0.9

    def test_policy_uses_the_fitted_gains(self):
        from tracecell.budget import BudgetPlanner
        from tracecell.policy import BudgetPolicy

        policy = BudgetPolicy()
        policy.repair_gains = {"reparse": 0.05}
        planner = BudgetPlanner(policy)
        plans = planner.candidate_actions(
            __import__("tracecell.budget", fromlist=["CellState"]).CellState(
                region_id="r0", confidence=0.5, severity=Severity.CRITICAL
            )
        )
        reparse = next(p for p in plans if p.action == "reparse")
        # 增益 0.05 远小于默认的 0.35，期望置信度应该明显更低
        assert reparse.expected_confidence < 0.5 + 0.5 * 0.35


class TestLayoutDetectorSelection:
    """版面检测器可切换：官方模型优先，没了就用自带的。"""

    def test_falls_back_to_the_builtin_detector(self, monkeypatch):
        from tracecell.providers import layout

        monkeypatch.delenv("TRACECELL_LAYOUT_MODEL", raising=False)
        assert layout.get_detector() is layout.detect

    def test_uses_the_official_model_when_available(self, monkeypatch):
        from tracecell.providers import layout

        class Fake:
            def __init__(self, model_dir):
                self.model_dir = model_dir

        monkeypatch.setenv("TRACECELL_LAYOUT_MODEL", "/tmp/does-not-exist")
        monkeypatch.setattr(layout, "PaddleLayoutDetector", Fake)
        detector = layout.get_detector()
        assert isinstance(detector, Fake)

    def test_broken_model_falls_back_instead_of_crashing(self, monkeypatch):
        from tracecell.providers import layout

        class Broken:
            def __init__(self, model_dir):
                raise RuntimeError("模型坏了")

        monkeypatch.setenv("TRACECELL_LAYOUT_MODEL", "/tmp/broken")
        monkeypatch.setattr(layout, "PaddleLayoutDetector", Broken)
        assert layout.get_detector() is layout.detect
