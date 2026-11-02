"""置信度继承与预算决策的测试。

这两块是 TraceCell 最核心的判断逻辑，边界条件值得钉死。
"""

from __future__ import annotations

import pytest

from tracecell.budget import BudgetPlanner, CellState
from tracecell.policy import BudgetPolicy, RiskPolicy
from tracecell.risk import (
    UNSCORED_FALLBACK,
    candidates_agree,
    expected_loss,
    field_confidence,
    region_confidence,
)
from tracecell.types import BBox, Candidate, Category, FieldValue, Region, Severity, Status, Verdict

from helpers import make_region


class TestRegionConfidence:
    def test_single_scored_candidate(self):
        region = make_region()
        assert region_confidence(region) == pytest.approx(0.97)

    def test_unscored_region_gets_conservative_fallback(self):
        region = make_region(
            candidates=[Candidate(text="x", origin="primary", verdict=Verdict.NOT_SCORED)]
        )
        assert region_confidence(region) == UNSCORED_FALLBACK

    def test_failed_verdict_is_not_treated_as_low_score(self):
        """官方服务失败时返回的固定值不能当成"模型认为错了"。"""

        region = make_region(
            candidates=[
                Candidate(text="x", origin="primary", score=0.0, verdict=Verdict.FAILED),
            ]
        )
        assert region_confidence(region) == UNSCORED_FALLBACK

    def test_disagreeing_candidates_get_discounted(self):
        region = make_region(
            candidates=[
                Candidate(text="145,318.00", origin="primary", score=0.94, verdict=Verdict.SCORED),
                Candidate(text="145,818.00", origin="reparse-16x", score=0.21, verdict=Verdict.SCORED),
            ]
        )
        assert candidates_agree(region) is False
        assert region_confidence(region) == pytest.approx(0.94 * 0.65)

    def test_agreeing_candidates_keep_top_score(self):
        region = make_region(
            candidates=[
                Candidate(text="145,318.00", origin="primary", score=0.88, verdict=Verdict.SCORED),
                Candidate(text="145,318.00", origin="reparse-16x", score=0.95, verdict=Verdict.SCORED),
            ]
        )
        assert candidates_agree(region) is True
        assert region_confidence(region) == pytest.approx(0.95)

    def test_single_candidate_has_no_agreement_information(self):
        assert candidates_agree(make_region()) is None


class TestFieldConfidence:
    def test_field_inherits_the_worst_supporting_region(self):
        """字段置信度取支撑区域里的最小值，不取平均。"""

        good = make_region("r000", "金额 100.00", candidates=[
            Candidate(text="金额 100.00", origin="primary", score=0.98, verdict=Verdict.SCORED)
        ])
        bad = make_region("r001", "金额 100.00", candidates=[
            Candidate(text="金额 100.00", origin="primary", score=0.30, verdict=Verdict.SCORED)
        ])
        regions = {"r000": good, "r001": bad}
        field = FieldValue(name="金额", value="100.00", region_ids=["r000", "r001"])
        assert field_confidence(field, regions) == pytest.approx(0.30)

    def test_field_without_evidence_has_zero_confidence(self):
        field = FieldValue(name="金额", value="100.00", region_ids=[])
        assert field_confidence(field, {}) == 0.0


class TestExpectedLoss:
    def test_critical_field_with_low_confidence_costs_more(self):
        assert expected_loss(0.2, Severity.CRITICAL) > expected_loss(0.2, Severity.LOW)

    def test_loss_is_zero_at_full_confidence(self):
        assert expected_loss(1.0, Severity.CRITICAL) == 0.0


class TestBudgetDecisions:
    def test_high_confidence_region_is_accepted_without_spending(self):
        planner = BudgetPlanner(BudgetPolicy())
        state = CellState(region_id="r000", confidence=0.97, severity=Severity.HIGH)
        decision = planner.decide(state)
        assert decision.action == "accept"
        assert decision.cost == 0.0

    def test_low_confidence_region_triggers_repair(self):
        planner = BudgetPlanner(BudgetPolicy())
        state = CellState(region_id="r000", confidence=0.31, severity=Severity.CRITICAL)
        decision = planner.decide(state)
        assert decision.action in ("reparse", "best_of_n", "vlm_escalation")
        assert decision.cost > 0

    def test_contested_region_is_never_auto_accepted(self):
        planner = BudgetPlanner(BudgetPolicy())
        state = CellState(region_id="r000", confidence=0.97, severity=Severity.HIGH, contested=True)
        decision = planner.decide(state)
        assert decision.action != "accept"

    def test_exhausted_budget_falls_back_to_human(self):
        policy = BudgetPolicy(total_budget=0.0)
        planner = BudgetPlanner(policy)
        state = CellState(region_id="r000", confidence=0.10, severity=Severity.CRITICAL)
        decision = planner.decide(state)
        assert decision.action == "human_review"

    def test_round_limit_stops_automatic_verification(self):
        policy = BudgetPolicy()
        planner = BudgetPlanner(policy)
        state = CellState(region_id="r000", confidence=0.10, severity=Severity.CRITICAL, rounds=3)
        decision = planner.decide(state)
        assert decision.action == "human_review"
        assert "轮次" in decision.reason

    def test_critical_regions_get_a_larger_local_budget(self):
        policy = BudgetPolicy(total_budget=10.0)
        assert policy.budget_for(Severity.CRITICAL) > policy.budget_for(Severity.LOW)


class TestRiskPolicy:
    def test_amount_field_is_critical(self):
        policy = RiskPolicy()
        assert policy.severity_for("价税合计", "145,318.00") is Severity.CRITICAL

    def test_large_amount_escalates_further(self):
        policy = RiskPolicy()
        small = policy.severity_for("金额", "100.00")
        large = policy.severity_for("金额", "8,000,000.00")
        assert large.value in ("high", "critical")
        assert small.value != "critical" or large.value == "critical"

    def test_note_field_stays_low(self):
        policy = RiskPolicy()
        assert policy.severity_for("备注", "备注内容") is Severity.LOW
