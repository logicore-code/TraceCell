"""场景层的测试。

场景是这个项目泛化能力的核心：加一种新单据只该改一份配置，不该动引擎。
"""

from __future__ import annotations

import pytest

from tracecell.extractors import build_extractor
from tracecell.policy import RiskPolicy
from tracecell.reconcile import check_sum_consistency, reconcile
from tracecell.scenarios import (
    CROSS_RULES,
    SCENARIOS,
    ConsistencyRule,
    cross_rules_for,
    get_scenario,
    list_scenarios,
)
from tracecell.types import (
    BBox,
    Category,
    DocumentLedger,
    FieldValue,
    Region,
    Severity,
    Status,
)


def make_region(region_id: str, text: str) -> Region:
    return Region(
        id=region_id,
        page=1,
        bbox=BBox(0.1, 0.1, 0.9, 0.15),
        category=Category.TEXT,
        text=text,
    )


class TestScenarioRegistry:
    def test_all_scenarios_are_listed(self):
        items = list_scenarios()
        assert len(items) == len(SCENARIOS)
        assert {item["name"] for item in items} == set(SCENARIOS)

    def test_every_scenario_has_fields_and_a_label(self):
        for name, scenario in SCENARIOS.items():
            assert scenario.label, f"{name} 缺 label"
            assert scenario.fields, f"{name} 没定义字段"

    def test_unknown_scenario_raises(self):
        with pytest.raises(KeyError):
            get_scenario("不存在的场景")

    def test_scenarios_cover_the_documented_ones(self):
        assert set(SCENARIOS) >= {
            "invoice",
            "contract",
            "expense",
            "financial_statement",
            "purchase_contract",
        }


class TestFieldExtraction:
    @pytest.mark.parametrize(
        "scenario,text,expected",
        [
            ("invoice", "发票号码：25117000000012345678", ("发票号码", "25117000000012345678")),
            ("invoice", "价税合计（小写）¥145,318.00", ("价税合计", "145,318.00")),
            ("financial_statement", "资产总计：¥21,000,000.00", ("资产总计", "21,000,000.00")),
            ("financial_statement", "负债和所有者权益总计：¥21,000,000.00", ("负债和所有者权益总计", "21,000,000.00")),
            ("purchase_contract", "合同总价：¥146,318.00", ("合同总价", "146,318.00")),
            ("purchase_contract", "数量：50 台", ("数量", "50")),
            ("purchase_contract", "税率：13%", ("税率", "13%")),
            ("purchase_contract", "需方（盖章）：杭州明远网络科技有限公司", ("需方", "杭州明远网络科技有限公司")),
        ],
    )
    def test_extracts_the_expected_value(self, scenario, text, expected):
        fields = build_extractor(scenario).extract([make_region("r000", text)])
        names = {f.name: f.value for f in fields}
        assert names.get(expected[0]) == expected[1]

    def test_invoice_total_is_not_confused_with_amount(self):
        """「价税合计」里的合计不能被「金额」规则抢走。"""

        fields = build_extractor("invoice").extract(
            [make_region("r000", "金额 ¥128,600.00"), make_region("r001", "价税合计（小写）¥145,318.00")]
        )
        names = {f.name: f.value for f in fields}
        assert names["金额"] == "128,600.00"
        assert names["价税合计"] == "145,318.00"


class TestScenarioRisk:
    def test_scenario_override_wins_over_default_rules(self):
        """财务报表的「资产总计」默认会被当成中等风险，场景覆盖把它提到致命。"""

        policy = RiskPolicy()
        scenario = get_scenario("financial_statement")
        assert policy.severity_for("资产总计", "21,000,000.00") is Severity.MEDIUM
        assert policy.severity_for("资产总计", "21,000,000.00", scenario) is Severity.CRITICAL

    def test_invoice_has_no_override_and_falls_back(self):
        policy = RiskPolicy()
        scenario = get_scenario("invoice")
        assert policy.severity_for("价税合计", "145,318.00", scenario) is Severity.CRITICAL


class TestConsistencyRules:
    def test_balance_sheet_imbalance_is_detected(self):
        scenario = get_scenario("financial_statement")
        fields = [
            FieldValue(name="资产总计", value="21,000,000.00"),
            FieldValue(name="流动资产合计", value="12,860,000.00"),
            FieldValue(name="非流动资产合计", value="8,140,000.00"),
            FieldValue(name="负债合计", value="6,800,000.00"),
            FieldValue(name="所有者权益合计", value="14,100,000.00"),
        ]
        findings = check_sum_consistency(fields, scenario.rules)
        # 资产 = 流动 + 非流动 这条通过，资产 = 负债 + 权益 这条不通过
        assert len(findings) == 1
        assert "100,000.00" in findings[0].message
        assert findings[0].severity is Severity.CRITICAL

    def test_balanced_balance_sheet_passes(self):
        scenario = get_scenario("financial_statement")
        fields = [
            FieldValue(name="资产总计", value="21,000,000.00"),
            FieldValue(name="流动资产合计", value="12,860,000.00"),
            FieldValue(name="非流动资产合计", value="8,140,000.00"),
            FieldValue(name="负债合计", value="6,800,000.00"),
            FieldValue(name="所有者权益合计", value="14,200,000.00"),
        ]
        assert check_sum_consistency(fields, scenario.rules) == []

    def test_purchase_contract_multiplication_and_sum(self):
        scenario = get_scenario("purchase_contract")
        fields = [
            FieldValue(name="数量", value="50"),
            FieldValue(name="单价", value="2,572.00"),
            FieldValue(name="金额", value="128,600.00"),
            FieldValue(name="税额", value="16,718.00"),
            FieldValue(name="合同总价", value="145,318.00"),
        ]
        assert check_sum_consistency(fields, scenario.rules) == []

        # 转置错：总价多了 1000
        fields[-1].value = "146,318.00"
        findings = check_sum_consistency(fields, scenario.rules)
        assert len(findings) == 1
        assert "1,000.00" in findings[0].message

    def test_expense_line_item_group_is_summed(self):
        scenario = get_scenario("expense")
        fields = [
            FieldValue(name="金额", value="2,864.50"),
            FieldValue(name="明细:差旅费", value="1,480.00"),
            FieldValue(name="明细:办公用品", value="386.50"),
            FieldValue(name="明细:业务招待费", value="1,098.00"),
        ]
        findings = check_sum_consistency(fields, scenario.rules)
        assert len(findings) == 1
        assert "2,964.50" in findings[0].message

    def test_invoice_total_equals_amount_plus_tax(self):
        scenario = get_scenario("invoice")
        fields = [
            FieldValue(name="金额", value="128,600.00"),
            FieldValue(name="税额", value="16,718.00"),
            FieldValue(name="价税合计", value="145,318.00"),
        ]
        assert check_sum_consistency(fields, scenario.rules) == []


def _ledger(doc_id: str, scenario: str, fields: list[tuple[str, str]]) -> DocumentLedger:
    ledger = DocumentLedger(doc_id=doc_id, source=f"{doc_id}.png")
    ledger.meta["scenario"] = scenario
    for name, value in fields:
        ledger.fields.append(
            FieldValue(name=name, value=value, region_ids=[f"{doc_id}:r000"], confidence=0.95)
        )
    return ledger


class TestCrossScenarioReconciliation:
    def test_contract_total_compared_with_invoice_total(self):
        """合同叫「合同总价」，发票叫「价税合计」，靠字段名撞不出来。"""

        ledgers = [
            _ledger("pc-001", "purchase_contract", [("合同总价", "146,318.00")]),
            _ledger("inv-001", "invoice", [("价税合计", "145,318.00")]),
        ]
        result = reconcile(ledgers)
        assert len(result.mismatches) == 1
        assert result.mismatches[0].left_field == "合同总价"
        assert result.mismatches[0].right_field == "价税合计"

    def test_matching_totals_pass(self):
        ledgers = [
            _ledger("pc-001", "purchase_contract", [("合同总价", "145,318.00")]),
            _ledger("inv-001", "invoice", [("价税合计", "145,318.00")]),
        ]
        assert reconcile(ledgers).consistent

    def test_different_scenarios_do_not_compare_same_named_fields(self):
        """发票的「金额」和报销单的「金额」不该互比。

        同名字段只在同场景之间比对，否则 128,600 和 2,864.50 会变成假矛盾。
        """

        ledgers = [
            _ledger("inv-001", "invoice", [("金额", "128,600.00")]),
            _ledger("exp-001", "expense", [("金额", "2,864.50")]),
        ]
        result = reconcile(ledgers)
        assert result.consistent
        assert result.compared_fields == 0

    def test_same_scenario_same_name_is_compared(self):
        ledgers = [
            _ledger("inv-001", "invoice", [("价税合计", "145,318.00")]),
            _ledger("inv-002", "invoice", [("价税合计", "145,818.00")]),
        ]
        result = reconcile(ledgers)
        assert len(result.mismatches) == 1

    def test_entity_names_use_similarity_threshold(self):
        ledgers = [
            _ledger("con-001", "contract", [("乙方", "南京海川信息技术有限公司")]),
            _ledger("inv-001", "invoice", [("销售方名称", "南京海川软件股份有限公司")]),
        ]
        result = reconcile(ledgers)
        assert len(result.mismatches) == 1
        assert result.mismatches[0].kind == "entity"

    def test_close_entity_names_are_tolerated(self):
        ledgers = [
            _ledger("con-001", "contract", [("乙方", "南京海川软件股份有限公司")]),
            _ledger("inv-001", "invoice", [("销售方名称", "南京海川软件股份有限公司")]),
        ]
        assert reconcile(ledgers).consistent

    def test_conflict_marks_both_sides_disputed(self):
        ledgers = [
            _ledger("pc-001", "purchase_contract", [("合同总价", "146,318.00")]),
            _ledger("inv-001", "invoice", [("价税合计", "145,318.00")]),
        ]
        reconcile(ledgers)
        assert all(f.status is Status.DISPUTED for l in ledgers for f in l.fields)


class TestCrossRuleSelection:
    def test_only_rules_between_present_scenarios_apply(self):
        rules = cross_rules_for(["invoice", "purchase_contract"])
        assert rules
        assert all(
            r.left_scenario in ("invoice", "purchase_contract")
            and r.right_scenario in ("invoice", "purchase_contract")
            for r in rules
        )

    def test_single_scenario_has_no_cross_rules(self):
        assert cross_rules_for(["invoice"]) == []

    def test_every_cross_rule_refers_to_known_scenarios(self):
        for rule in CROSS_RULES:
            assert rule.left_scenario in SCENARIOS
            assert rule.right_scenario in SCENARIOS
