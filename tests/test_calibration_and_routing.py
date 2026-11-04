"""阈值标定与文档级路由的测试。"""

from __future__ import annotations

import math

import pytest
from helpers import make_region

from tracecell.calibration import (
    CalibrationSample,
    apply_isotonic,
    auroc,
    calibrate,
    isotonic_regression,
    recommend_threshold,
    threshold_curve,
)
from tracecell.policy import BudgetPolicy
from tracecell.routing import (
    detection_probability,
    render_review_checklist,
    required_sample_size,
    route_document,
)
from tracecell.types import FieldValue, Severity, Status


class TestAuroc:
    def test_perfectly_separated_scores(self):
        samples = [
            CalibrationSample(score=0.9, correct=True),
            CalibrationSample(score=0.8, correct=True),
            CalibrationSample(score=0.3, correct=False),
            CalibrationSample(score=0.2, correct=False),
        ]
        assert auroc(samples) == pytest.approx(1.0)

    def test_random_scores(self):
        samples = [
            CalibrationSample(score=0.5, correct=True),
            CalibrationSample(score=0.5, correct=False),
        ]
        assert auroc(samples) == pytest.approx(0.5)

    def test_single_class_returns_nan(self):
        samples = [CalibrationSample(score=0.9, correct=True)]
        assert math.isnan(auroc(samples))


class TestIsotonic:
    def test_monotonic_output(self):
        samples = [
            CalibrationSample(score=0.1, correct=False),
            CalibrationSample(score=0.2, correct=False),
            CalibrationSample(score=0.8, correct=True),
            CalibrationSample(score=0.9, correct=True),
        ]
        xs, ys = isotonic_regression(samples)
        assert ys == sorted(ys)
        assert len(xs) == len(ys)

    def test_calibrated_probability_tracks_observation(self):
        samples = [CalibrationSample(score=0.9, correct=True) for _ in range(9)]
        samples += [CalibrationSample(score=0.1, correct=False) for _ in range(1)]
        xs, ys = isotonic_regression(samples)
        assert apply_isotonic(xs, ys, 0.9) == pytest.approx(1.0)
        assert apply_isotonic(xs, ys, 0.1) == pytest.approx(0.0)


class TestThresholdCalibration:
    def _samples(self):
        return [
            CalibrationSample(score=0.99, correct=True),
            CalibrationSample(score=0.97, correct=True),
            CalibrationSample(score=0.95, correct=True),
            CalibrationSample(score=0.92, correct=True),
            CalibrationSample(score=0.41, correct=False),
            CalibrationSample(score=0.33, correct=False),
            CalibrationSample(score=0.22, correct=False),
        ]

    def test_threshold_respects_target_accuracy(self):
        result = calibrate(self._samples(), target_accuracy=0.99)
        assert result.accept_accuracy >= 0.99
        assert result.threshold > 0.9

    def test_looser_target_accepts_more(self):
        loose = calibrate(self._samples(), target_accuracy=0.7)
        strict = calibrate(self._samples(), target_accuracy=0.99)
        assert loose.accept_rate >= strict.accept_rate

    def test_empty_sample_is_handled(self):
        result = calibrate([], target_accuracy=0.99)
        assert result.total == 0
        assert "无法" in result.note or result.threshold == 1.0

    def test_accept_rate_shrinks_as_threshold_grows(self):
        """阈值抬高，自动放行的比例只会变小——这是单调的。"""

        curve = threshold_curve(self._samples())
        rates = [point["accept_rate"] for point in curve]
        assert all(a >= b - 1e-9 for a, b in zip(rates, rates[1:], strict=False))

    def test_cost_aware_recommendation(self):
        """人工贵的时候阈值应该放宽，人工便宜的时候应该收紧。"""

        samples = self._samples()
        expensive_human = recommend_threshold(samples, human_review_cost=50.0, error_cost=1.0)
        cheap_human = recommend_threshold(samples, human_review_cost=0.01, error_cost=1.0)
        assert expensive_human["threshold"] <= cheap_human["threshold"]


class TestSampling:
    def test_sample_size_grows_as_error_rate_drops(self):
        assert required_sample_size(0.1) > required_sample_size(0.3)

    def test_zero_error_rate_needs_no_sample(self):
        assert required_sample_size(0.0) == 0

    def test_detection_probability_matches_formula(self):
        assert detection_probability(0.1, 10) == pytest.approx(1 - 0.9**10)

    def test_sample_size_achieves_target_probability(self):
        size = required_sample_size(0.05, detection_probability=0.95)
        assert detection_probability(0.05, size) >= 0.95


class TestRouting:
    def test_clean_document_is_auto_passed(self):
        region = make_region("r000", "备注 内容", candidates=[
            _candidate("备注 内容", 0.99)
        ])
        fields = [FieldValue(name="备注", value="备注 内容", region_ids=["r000"], confidence=0.99,
                             severity=Severity.LOW, status=Status.ACCEPTED)]
        result = route_document(fields, {"r000": region}, BudgetPolicy())
        assert result.verdict == "auto_pass"

    def test_contested_region_forces_review(self):
        region = make_region("r000", "金额 100.00", candidates=[
            _candidate("金额 100.00", 0.95),
            _candidate("金额 700.00", 0.20),
        ])
        fields = [FieldValue(name="金额", value="100.00", region_ids=["r000"], confidence=0.6,
                             severity=Severity.CRITICAL)]
        result = route_document(fields, {"r000": region}, BudgetPolicy())
        assert result.verdict in ("spot_check", "full_review")
        assert "r000" in result.must_review

    def test_orphan_field_is_flagged(self):
        fields = [FieldValue(name="金额", value="100.00", region_ids=[], confidence=0.0,
                             severity=Severity.CRITICAL)]
        result = route_document(fields, {}, BudgetPolicy())
        assert result.verdict == "full_review"
        assert any(item.startswith("field:") for item in result.must_review)

    def test_checklist_explains_why(self):
        region = make_region("r000", "金额 100.00", candidates=[
            _candidate("金额 100.00", 0.95),
            _candidate("金额 700.00", 0.20),
        ])
        fields = [FieldValue(name="金额", value="100.00", region_ids=["r000"], confidence=0.4,
                             severity=Severity.CRITICAL)]
        checklist = render_review_checklist({"r000": region}, fields, ["r000"])
        assert checklist
        assert "不一致" in checklist[0]["why"]
        assert len(checklist[0]["candidates"]) == 2


def _candidate(text: str, score: float):
    from tracecell.types import Candidate, Verdict

    return Candidate(text=text, origin="primary", score=score, verdict=Verdict.SCORED)
