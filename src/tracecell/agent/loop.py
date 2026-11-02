"""核验闭环。

这是 TraceCell 的 Agent 主循环。一份文档进来，出去的是一份带证据的账本，
中间经过这样一条链子：

    解析 ──► 评分 ──► 决策 ──► 修复/升级/转人工 ──► 复评 ──► 收敛

每一步都由 budget.BudgetPlanner 决定要不要做、花多少做。这个循环本身
不"思考"，它只负责执行策略，并把每一步的理由和花费记进账本。

三个模型在这里各司其职：

* MiniCPM-OCR 1.0 负责读；
* MiniCPM-OCR Jev 负责判断读得对不对；
* MiniCPM-V 4.7 负责在前两者都搞不定时做最后裁决。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from ..budget import BudgetPlanner, CellState
from ..budget import Decision as PlanDecision
from ..extractors import RegexFieldExtractor, build_extractor, schema_by_name
from ..ledger import build_ledger, collect_findings
from ..policy import BudgetPolicy, RiskPolicy, sum_rule_for
from ..providers import layout as layout_mod
from ..providers.base import (
    PROFILE_ALT,
    PROFILE_PRIMARY,
    CATEGORY_TO_JEV,
    Escalator,
    ParseProfile,
    Parser,
    Scorer,
)
from ..providers.vlm import build_review_question
from ..risk import region_confidence
from ..reconcile import check_sum_consistency
from ..routing import route_document
from ..types import (
    BudgetReport,
    Candidate,
    Category,
    Decision,
    DocumentLedger,
    FieldValue,
    Region,
    Severity,
    Status,
    Verdict,
)

SEVERITY_ORDER = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]


@dataclass
class LoopReport:
    """一次闭环的运行结果。"""

    ledger: DocumentLedger
    elapsed: float = 0.0
    extra_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.ledger.doc_id,
            "verdict": self.ledger.verdict,
            "elapsed": round(self.elapsed, 3),
            "extra_calls": self.extra_calls,
            "budget": self.ledger.budget.to_dict(),
        }


class VerificationLoop:
    """把一张图变成一份可信的证据账本。"""

    def __init__(
        self,
        parser: Parser,
        scorer: Scorer,
        escalator: Escalator | None = None,
        policy: BudgetPolicy | None = None,
        risk_policy: RiskPolicy | None = None,
        extractor: Callable[[list[Region]], list[FieldValue]] | None = None,
        schema: str = "invoice",
        alt_profile: ParseProfile = PROFILE_ALT,
    ) -> None:
        self.parser = parser
        self.scorer = scorer
        self.escalator = escalator
        self.policy = policy or BudgetPolicy()
        self.risk_policy = risk_policy or RiskPolicy()
        self.schema = schema
        self.extractor = extractor or build_extractor(schema)
        self.alt_profile = alt_profile
        self.planner = BudgetPlanner(self.policy)
        self._rounds: dict[str, int] = {}
        self._extra_calls = 0

    # -- 主流程 -----------------------------------------------------------
    def run(self, image: bytes, doc_id: str, source: str = "") -> LoopReport:
        started = time.time()
        self.planner = BudgetPlanner(self.policy)
        self._rounds = {}
        self._extra_calls = 0

        # 1) 解析
        page = self.parser.parse_page(image, PROFILE_PRIMARY)
        regions = self._ensure_primary(page.regions)

        # 2) 给首轮候选打分
        self._score_with_images(regions, image)
        self.planner.charge_base(self.policy.cost_model.primary_parse)

        # 3) 抽字段、定风险
        fields = self._extract(regions)
        for fld in fields:
            fld.severity = self.risk_policy.severity_for(fld.name, fld.value)

        # 4) 闭环
        max_iterations = len(regions) * (self.policy.max_rounds + 1) + 8
        for _ in range(max_iterations):
            pending = [
                region
                for region in regions
                if region.status in (Status.PENDING, Status.ESCALATED)
            ]
            if not pending:
                break
            progressed = False
            for region in pending:
                state = self._state_for(region, fields)
                if state.confidence >= self.policy.accept_threshold and not state.contested:
                    region.status = Status.ACCEPTED
                    continue
                if state.rounds >= self.policy.max_rounds:
                    region.status = Status.PENDING
                    continue

                decision = self.planner.decide(state)
                action = decision.action

                if action == "accept":
                    region.status = Status.ACCEPTED
                    self._record(region, decision)
                    continue

                if action == "human_review":
                    region.status = Status.PENDING
                    self._record(region, decision)
                    continue

                changed = self._execute(region, action, image)
                self._extra_calls += 1
                self.planner.spend(region.id, decision.cost)
                self._rounds[region.id] = self._rounds.get(region.id, 0) + 1
                self._record(region, decision)

                # 重新算一次置信度，能过线就放行
                refreshed = region_confidence(region)
                if refreshed >= self.policy.accept_threshold and not _contested(region):
                    region.status = Status.ACCEPTED
                if changed:
                    progressed = True
            if not progressed:
                break

        # 5) 组装账本
        ledger = build_ledger(
            doc_id=doc_id,
            source=source,
            regions=regions,
            fields=fields,
            risk_policy=self.risk_policy,
            budget_policy=self.policy,
            markdown=page.markdown,
            pages=page.meta.get("pages", 1),
            meta={
                "profile": PROFILE_PRIMARY.to_dict(),
                "alt_profile": self.alt_profile.to_dict(),
                "models": page.meta.get("models", {}),
            },
        )

        # 6) 路由
        region_map = {r.id: r for r in regions}
        routing = route_document(ledger.fields, region_map, self.policy)
        ledger.verdict = routing.verdict
        ledger.review_queue = routing.must_review
        ledger.meta["routing"] = routing.to_dict()

        # 7) 账单与问题清单
        ledger.budget = self._budget_report(regions, routing)
        ledger.findings = collect_findings(ledger.fields, region_map, self.policy)

        # 8) 单据内部的勾稽校验：明细加起来等于不等于合计
        rule = sum_rule_for(self.schema)
        sum_findings = check_sum_consistency(ledger.fields, rule)
        if sum_findings:
            ledger.findings.extend(sum_findings)
            ledger.meta["sum_check"] = {
                "total": rule.total,
                "components": rule.components,
                "status": "mismatch",
            }

        return LoopReport(
            ledger=ledger,
            elapsed=time.time() - started,
            extra_calls=self._extra_calls,
        )

    # -- 内部动作 ---------------------------------------------------------
    def _extract(self, regions: list[Region]) -> list[FieldValue]:
        """调用抽取器。兼容"有 extract 方法"和"本身就是可调用对象"两种写法。"""

        extractor = self.extractor
        if hasattr(extractor, "extract"):
            return list(extractor.extract(regions))
        return list(extractor(regions))

    def _ensure_primary(self, regions: list[Region]) -> list[Region]:
        for region in regions:
            if not region.candidates:
                region.candidates = [
                    Candidate(text=region.text, origin="primary", verdict=Verdict.NOT_SCORED)
                ]
            elif region.candidates[0].origin != "primary":
                region.candidates.insert(
                    0, Candidate(text=region.text, origin="primary", verdict=Verdict.NOT_SCORED)
                )
        return regions

    def _score_with_images(self, regions: list[Region], image: bytes) -> None:
        """给所有还没有分数的候选打分。"""

        for region in regions:
            crop = layout_mod.crop(image, region.bbox)
            for candidate in region.candidates:
                if candidate.verdict is not Verdict.NOT_SCORED:
                    continue
                result = self.scorer.score(crop, candidate.text, _jev_category(region.category))
                candidate.score = result.score
                candidate.verdict = result.verdict
                candidate.detail = result.detail
                self.planner.charge_base(self.policy.cost_model.score)

    def _execute(self, region: Region, action: str, image: bytes) -> bool:
        """执行一个动作。返回区域状态是否发生变化。"""

        crop = layout_mod.crop(image, region.bbox)

        if action == "reparse":
            text = self.parser.parse_region(crop, None, self.alt_profile)
            if not text:
                return False
            candidate = Candidate(text=text, origin=self.alt_profile.name, verdict=Verdict.NOT_SCORED)
            region.candidates.append(candidate)
            result = self.scorer.score(crop, text, _jev_category(region.category))
            candidate.score = result.score
            candidate.verdict = result.verdict
            candidate.detail = result.detail
            self.planner.spend(region.id, self.policy.cost_model.score)
            if _is_better(region, candidate):
                region.status = Status.REPAIRED
            return True

        if action == "best_of_n":
            texts = [c.text for c in region.candidates]
            results = self.scorer.score_best(crop, texts, _jev_category(region.category))
            changed = False
            for candidate, result in zip(region.candidates, results):
                if candidate.verdict is Verdict.NOT_SCORED:
                    candidate.score = result.score
                    candidate.verdict = result.verdict
                    candidate.detail = result.detail
                    changed = True
            self.planner.spend(region.id, self.policy.cost_model.score * max(1, len(texts) - 1))
            return changed

        if action == "vlm_escalation":
            if self.escalator is None:
                return False
            question = build_review_question(region.selected_text, [c.text for c in region.candidates])
            answer = self.escalator.review(crop, question)
            if not answer:
                return False
            candidate = Candidate(text=answer, origin="vlm-review", verdict=Verdict.NOT_SCORED)
            region.candidates.append(candidate)
            result = self.scorer.score(crop, answer, _jev_category(region.category))
            candidate.score = result.score
            candidate.verdict = result.verdict
            candidate.detail = result.detail
            self.planner.spend(region.id, self.policy.cost_model.score)
            region.status = Status.REPAIRED if _is_better(region, candidate) else Status.ESCALATED
            return True

        return False

    def _state_for(self, region: Region, fields: list[FieldValue]) -> CellState:
        """构造决策所需的区域状态。"""

        related = [f for f in fields if region.id in f.region_ids]
        severity = Severity.LOW
        for fld in related:
            if SEVERITY_ORDER.index(fld.severity) > SEVERITY_ORDER.index(severity):
                severity = fld.severity

        scored = [c for c in region.candidates if c.scored]
        impact = 1.0
        if related:
            impact = 1.0 + 0.5 * sum(
                1 for f in related if f.severity in (Severity.CRITICAL, Severity.HIGH)
            )

        return CellState(
            region_id=region.id,
            confidence=region_confidence(region),
            severity=severity,
            contested=_contested(region),
            rounds=self._rounds.get(region.id, 0),
            candidate_count=len(region.candidates),
            scored_count=len(scored),
            impact=impact,
        )

    def _record(self, region: Region, decision: PlanDecision) -> None:
        region.note = decision.reason

    def _budget_report(self, regions: list[Region], routing) -> BudgetReport:
        accepted = sum(1 for r in regions if r.status is Status.ACCEPTED)
        repaired = sum(1 for r in regions if r.status is Status.REPAIRED)
        escalated = sum(1 for r in regions if r.status is Status.ESCALATED)
        pending = sum(1 for r in regions if r.status is Status.PENDING)
        report = BudgetReport(
            total_cost=round(self.planner.total_cost, 4),
            base_cost=round(self.planner.base_cost, 4),
            extra_cost=round(self.planner.spent, 4),
            extra_calls=self._extra_calls,
            accepted_regions=accepted,
            repaired_regions=repaired,
            escalated_regions=escalated,
            pending_regions=pending,
            human_cells=len(routing.must_review),
        )
        for region_id, cost in self.planner.spent_by_region().items():
            if region_id == "__initial__":
                continue
            report.cost_by_action[region_id] = round(cost, 4)
        return report


def _jev_category(category: Category) -> str:
    return CATEGORY_TO_JEV.get(category.value, "text")


def _contested(region: Region) -> bool:
    scored = [c for c in region.candidates if c.scored]
    if len(scored) < 2:
        return False
    return len({c.text for c in scored}) > 1


def _is_better(region: Region, candidate: Candidate) -> bool:
    """新候选是否优于当前选中的候选。"""

    best = region.best
    if best is None or not best.scored:
        return True
    if not candidate.scored:
        return False
    return (candidate.score or 0.0) > (best.score or 0.0)
