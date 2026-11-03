"""逐格预算决策。

这是 TraceCell 的"大脑"。每个区域在核验过程中都会被问同一个问题：

    现在停下的残余风险有多大？再花一次推理能换回多少？这次推理要花多少？
    让人来看一眼又要花多少、换回多少？

三个数字一比较，答案就出来了。这套决策不追求"把所有格子都查到 100%"
——那既昂贵也没有必要；它追求的是**在给定预算下，把残余风险压到最低**。

两条设计原则：

1. 人的复核成本被显式建模。人工复核贵，但确定性强（看一眼基本不会错）。
   在低危格子上让人看是浪费，在高危格子上让人看可能比再跑三次模型更划算。
2. 预算按风险分配。高危字段拿走大部分预算，低危字段大概率直接放行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import SEVERITY_WEIGHT, Severity

# 人工复核的正确率。看过原图的人几乎不会读错，但它不是 1.0——疲劳、
# 潦草字迹都会出错。
HUMAN_ACCURACY = 0.995

# 各类动作"修好这一格"的期望概率。
# 不是拍脑袋的常数：它们描述的是"换一种解析策略后，这一格被纠正的
# 可能性"，离线样例和线上日志都可以回归校准，见 calibration.py。
REPAIR_SUCCESS = {
    "reparse": 0.35,
    "best_of_n": 0.55,
    "vlm_escalation": 0.65,
}


def fit_repair_gains(sessions: list[dict[str, Any]]) -> dict[str, float]:
    """从录制数据里回归"换一种策略能修好这一格的概率"。

    REPAIR_SUCCESS 那几个常数一开始是拍脑袋的，后来觉得该有出处。判据是
    这样的：一个区域有两条读法不同的候选，说明"第一次读错了，换个参数
    重读可能读对"。有多少个区域确实被后来的候选纠正了，比率就是重解析
    的期望增益。

    样本少的时候这个估计不可靠，宁可保守一点——所以返回的增益会往小了
    取，并同时给出样本量。
    """

    attempts = {"reparse": 0, "best_of_n": 0, "vlm_escalation": 0}
    successes = dict(attempts)

    for session in sessions:
        for document in session.get("documents", []):
            for region in document.get("regions", []):
                candidates = [c for c in region.get("candidates", []) if c.get("score") is not None]
                if len(candidates) < 2:
                    continue
                primary = next((c for c in candidates if c.get("origin") == "primary"), None)
                if primary is None:
                    continue
                for candidate in candidates:
                    if candidate.get("origin") == "primary":
                        continue
                    if candidate.get("text") == primary.get("text"):
                        continue
                    # 有分歧，说明第一次读出了别的东西
                    origin = candidate.get("origin", "reparse-16x")
                    key = "vlm_escalation" if origin == "vlm-review" else "reparse"
                    attempts[key] = attempts.get(key, 0) + 1
                    if (candidate.get("score") or 0) > (primary.get("score") or 0):
                        successes[key] = successes.get(key, 0) + 1

    gains: dict[str, float] = {}
    for key, count in attempts.items():
        if count == 0:
            gains[key] = REPAIR_SUCCESS.get(key, 0.35)
            continue
        # 样本少时往保守了取：加一个先验，避免一两个样本就得出 100%
        prior = 0.35
        weight = min(count, 20) / 20
        gains[key] = round((successes[key] / count) * weight + prior * (1 - weight), 4)
    return gains


@dataclass
class CellState:
    """决策时需要知道的关于一格的全部信息。"""

    region_id: str
    confidence: float
    severity: Severity = Severity.MEDIUM
    contested: bool = False
    rounds: int = 0
    candidate_count: int = 1
    scored_count: int = 0
    """已评分的候选数。为 0 时连 best_of_n 都做不了。"""

    impact: float = 1.0
    """影响面：这一格支撑着几个高危字段。数值越大越值得查。"""

    @property
    def weight(self) -> float:
        return SEVERITY_WEIGHT[self.severity.value] * self.impact


@dataclass
class ActionPlan:
    """一个候选动作及其预期效果。"""

    action: str
    cost: float
    expected_confidence: float
    reason: str
    risk_reduction: float = 0.0
    efficiency: float = 0.0
    feasible: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "cost": self.cost,
            "expected_confidence": round(self.expected_confidence, 4),
            "reason": self.reason,
            "risk_reduction": round(self.risk_reduction, 6),
            "efficiency": round(self.efficiency, 6),
            "feasible": self.feasible,
            "note": self.note,
        }


@dataclass
class Decision:
    """决策结果。"""

    action: str
    reason: str
    cost: float
    confidence_before: float
    confidence_after: float
    risk_before: float
    risk_after: float
    alternatives: list[ActionPlan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "reason": self.reason,
            "cost": self.cost,
            "confidence_before": round(self.confidence_before, 4),
            "confidence_after": round(self.confidence_after, 4),
            "risk_before": round(self.risk_before, 6),
            "risk_after": round(self.risk_after, 6),
            "alternatives": [a.to_dict() for a in self.alternatives],
        }


class BudgetPlanner:
    """按风险分配预算，按效率选择动作。"""

    def __init__(self, policy) -> None:
        self.policy = policy
        self.cost_model = policy.cost_model
        self._spent = 0.0
        self._base_cost = 0.0
        self._spent_by_region: dict[str, float] = {}

    # -- 预算记账 ---------------------------------------------------------
    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return max(0.0, self.policy.total_budget - self._spent)

    def spend(self, region_id: str, cost: float) -> None:
        """记一笔额外核验开销。这部分受预算约束。"""

        self._spent += cost
        self._spent_by_region[region_id] = self._spent_by_region.get(region_id, 0.0) + cost

    def charge_base(self, cost: float) -> None:
        """记一笔基础开销（首次解析、首轮评分）。

        这部分无论策略怎么决策都会发生，因此不占用额外核验预算——预算
        约束的是"要不要再多查一次"，不是"已经查过的账"。
        """

        self._base_cost += cost

    @property
    def base_cost(self) -> float:
        return self._base_cost

    @property
    def total_cost(self) -> float:
        return self._base_cost + self._spent

    def region_spent(self, region_id: str) -> float:
        return self._spent_by_region.get(region_id, 0.0)

    def spent_by_region(self) -> dict[str, float]:
        return dict(self._spent_by_region)

    def local_budget_left(self, state: CellState) -> float:
        """这一格还能花多少。

        取全局剩余预算和按风险分配的局部上限里更小的那个。
        """

        local = self.policy.budget_for(state.severity) - self.region_spent(state.region_id)
        return max(0.0, min(self.remaining, local))

    # -- 动作评估 ---------------------------------------------------------
    def _risk(self, state: CellState, confidence: float) -> float:
        return state.weight * max(0.0, 1.0 - confidence)

    def _plan(
        self,
        state: CellState,
        action: str,
        expected_confidence: float,
        reason: str,
    ) -> ActionPlan:
        cost = self.cost_model.cost_of(action)
        risk_before = self._risk(state, state.confidence)
        risk_after = self._risk(state, expected_confidence)
        reduction = max(0.0, risk_before - risk_after)
        efficiency = reduction / cost if cost > 0 else float("inf") if reduction > 0 else 0.0
        return ActionPlan(
            action=action,
            cost=cost,
            expected_confidence=expected_confidence,
            reason=reason,
            risk_reduction=reduction,
            efficiency=efficiency,
        )

    def candidate_actions(self, state: CellState) -> list[ActionPlan]:
        """列出当前状态下所有可考虑的动作。"""

        plans: list[ActionPlan] = []

        # 重解析：换一套视觉参数重新读一遍。因为新旧候选都会留下并交给
        # Jev 择优，所以这一步只可能变好，不会变差。
        gain = self.policy.repair_gains.get("reparse", REPAIR_SUCCESS["reparse"]) * (
            1.0 - state.confidence
        )
        plans.append(
            self._plan(
                state,
                "reparse",
                state.confidence + gain,
                "换一套解析参数重读该区域，旧候选保留参与择优",
            )
        )

        # 择优：在已有候选里挑最好的。需要至少有一个未评分的候选。
        if state.candidate_count > state.scored_count and state.candidate_count >= 2:
            gain = self.policy.repair_gains.get("best_of_n", REPAIR_SUCCESS["best_of_n"]) * (
                1.0 - state.confidence
            )
            plans.append(
                self._plan(
                    state,
                    "best_of_n",
                    state.confidence + gain,
                    f"已有 {state.candidate_count} 个候选、{state.scored_count} 个已评分，补齐评分后择优",
                )
            )

        # 升级到多模态模型：贵，但对疑难区域最有效。
        gain = self.policy.repair_gains.get(
            "vlm_escalation", REPAIR_SUCCESS["vlm_escalation"]
        ) * (1.0 - state.confidence)
        plans.append(
            self._plan(
                state,
                "vlm_escalation",
                state.confidence + gain,
                "升级到 MiniCPM-V 4.7 定向复核该区域",
            )
        )

        # 人工复核：确定性强，但成本最高。
        plans.append(
            self._plan(
                state,
                "human_review",
                HUMAN_ACCURACY,
                "转人工复核（保留原图裁剪与全部候选）",
            )
        )
        return plans

    # -- 主决策 -----------------------------------------------------------
    def decide(self, state: CellState) -> Decision:
        """决定这一格下一步怎么办。"""

        risk_before = self._risk(state, state.confidence)

        # 1) 证据充分且没有分歧，直接放行。
        if (
            state.confidence >= self.policy.accept_threshold
            and not state.contested
            and state.rounds < self.policy.max_rounds
        ):
            return Decision(
                action="accept",
                reason=(
                    f"置信度 {state.confidence:.3f} ≥ 阈值 {self.policy.accept_threshold:.2f}"
                    f"且候选之间无分歧，免检通过"
                ),
                cost=0.0,
                confidence_before=state.confidence,
                confidence_after=state.confidence,
                risk_before=risk_before,
                risk_after=risk_before,
            )

        # 2) 轮次用尽，别再耗了。
        if state.rounds >= self.policy.max_rounds:
            plan = self._plan(state, "human_review", HUMAN_ACCURACY, "核验轮次已用尽")
            return Decision(
                action="human_review",
                reason=f"已达到最大核验轮次 {self.policy.max_rounds}，停止自动核验并转人工",
                cost=plan.cost,
                confidence_before=state.confidence,
                confidence_after=HUMAN_ACCURACY,
                risk_before=risk_before,
                risk_after=self._risk(state, HUMAN_ACCURACY),
                alternatives=[plan],
            )

        # 3) 比较所有候选动作。
        plans = self.candidate_actions(state)
        budget_left = self.local_budget_left(state)

        for plan in plans:
            if plan.action == "human_review":
                continue
            if plan.cost > budget_left:
                plan.feasible = False
                plan.note = f"超出可用预算（剩 {budget_left:.2f} rcu）"

        auto_plans = [p for p in plans if p.action != "human_review"]
        human_plan = next(p for p in plans if p.action == "human_review")

        feasible = [p for p in auto_plans if p.feasible and p.risk_reduction > 0]
        if not feasible:
            # 预算不够做自动核验，那就看人工值不值。
            if human_plan.risk_reduction > 0:
                return Decision(
                    action="human_review",
                    reason=(
                        f"自动核验预算不足（剩 {budget_left:.2f} rcu），"
                        f"转人工复核以消除 {human_plan.risk_reduction:.3f} 的期望损失"
                    ),
                    cost=human_plan.cost,
                    confidence_before=state.confidence,
                    confidence_after=HUMAN_ACCURACY,
                    risk_before=risk_before,
                    risk_after=self._risk(state, HUMAN_ACCURACY),
                    alternatives=plans,
                )
            return Decision(
                action="accept",
                reason="预算耗尽且该格期望损失已低于容忍度，按现状放行",
                cost=0.0,
                confidence_before=state.confidence,
                confidence_after=state.confidence,
                risk_before=risk_before,
                risk_after=risk_before,
                alternatives=plans,
            )

        # 4) 效率最高的自动动作。
        best = max(feasible, key=lambda p: p.efficiency)

        # 5) 人工复核在"每单位风险的花费"上更划算吗？
        #    人工成本高，但准确率接近 1 sometimes 反而是性价比最高的选项。
        if human_plan.efficiency > best.efficiency:
            return Decision(
                action="human_review",
                reason=(
                    f"人工复核每单位风险的花费更低"
                    f"（{human_plan.efficiency:.3f} > {best.efficiency:.3f}）"
                ),
                cost=human_plan.cost,
                confidence_before=state.confidence,
                confidence_after=HUMAN_ACCURACY,
                risk_before=risk_before,
                risk_after=self._risk(state, HUMAN_ACCURACY),
                alternatives=plans,
            )

        # 6) 自动动作能把残余风险压到容忍度以下吗？
        residual = self._risk(state, best.expected_confidence)
        if residual > self.policy.residual_tolerance:
            return Decision(
                action=best.action,
                reason=(
                    f"残余风险 {residual:.4f} 仍高于容忍度 {self.policy.residual_tolerance:.4f}，"
                    f"执行最优动作 {best.action}"
                ),
                cost=best.cost,
                confidence_before=state.confidence,
                confidence_after=best.expected_confidence,
                risk_before=risk_before,
                risk_after=residual,
                alternatives=plans,
            )

        return Decision(
            action=best.action,
            reason=best.reason,
            cost=best.cost,
            confidence_before=state.confidence,
            confidence_after=best.expected_confidence,
            risk_before=risk_before,
            risk_after=residual,
            alternatives=plans,
        )


def plan_summary(plans: list[ActionPlan]) -> list[dict[str, Any]]:
    """给日志和界面用的动作清单。"""

    return [p.to_dict() for p in plans]
