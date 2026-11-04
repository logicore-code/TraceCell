"""三个独立应用案例的测试。

三个模型各自都能单独跑成一个完整案例——每个案例都得
能单独跑通，不能只在流水线上才有意义。
"""

from __future__ import annotations

import json

import pytest

from tracecell.cases import run_jev_case, run_ocr_case, run_vlm_case
from tracecell.providers.replay import (
    ReplayEscalator,
    ReplayParser,
    ReplayScorer,
    ReplaySession,
)

DOC_ID = "invoice-001"


@pytest.fixture()
def session(tmp_path):
    data = {
        "documents": [
            {
                "doc_id": DOC_ID,
                "source": "samples/invoice_001.png",
                "schema": "invoice",
                "regions": [
                    {
                        "id": "r000",
                        "page": 1,
                        "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.6, "y1": 0.2},
                        "category": "text",
                        "text": "价税合计（小写）¥145,318.00",
                        "candidates": [
                            {
                                "text": "价税合计（小写）¥145,318.00",
                                "origin": "primary",
                                "score": 0.97,
                                "verdict": "scored",
                            },
                            {
                                "text": "价税合计（小写）¥145,818.00",
                                "origin": "reparse-16x",
                                "score": 0.19,
                                "verdict": "scored",
                            },
                        ],
                    }
                ],
                "fields": [{"name": "价税合计", "value": "145,318.00", "region_ids": ["r000"]}],
            }
        ]
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return ReplaySession.load(path)


class TestCases:
    def test_ocr_case_returns_regions_with_coordinates(self, session):
        result = run_ocr_case(ReplayParser(session, DOC_ID), b"")
        assert result.case == "ocr"
        assert result.models == ["MiniCPM-OCR 1.0"]
        assert result.regions
        assert set(result.regions[0]["bbox"]) == {"x0", "y0", "x1", "y1"}

    def test_jev_case_splits_regions_by_threshold(self, session):
        scorer = ReplayScorer(session, DOC_ID)
        parser = ReplayParser(session, DOC_ID)
        result = run_jev_case(scorer, parser, b"", threshold=0.5)
        assert result.case == "jev"
        assert result.summary["auto_accept"] >= 1
        assert result.summary["threshold"] == 0.5

    def test_vlm_case_works_without_locate_support(self, session):
        """回放后端没有 locate，实现不能假设它存在。"""

        result = run_vlm_case(ReplayEscalator(session, DOC_ID), b"")
        assert result.case == "vlm"
        assert result.models == ["MiniCPM-V 4.7"]
        assert "located_bbox" in result.summary

    def test_every_case_declares_its_models(self, session):
        cases = [
            run_ocr_case(ReplayParser(session, DOC_ID), b""),
            run_jev_case(ReplayScorer(session, DOC_ID), ReplayParser(session, DOC_ID), b""),
            run_vlm_case(ReplayEscalator(session, DOC_ID), b""),
        ]
        for case in cases:
            assert case.models
            assert all(name.startswith("MiniCPM") for name in case.models)
