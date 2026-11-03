"""文本归一化与跨文档对账的测试。"""

from __future__ import annotations

import pytest

from tracecell.reconcile import infer_kind, merge_ledgers, normalize_by_kind, reconcile
from tracecell.textnorm import (
    normalize,
    normalize_date,
    normalize_entity,
    normalize_money,
    similarity,
    to_halfwidth,
)
from tracecell.types import DocumentLedger, FieldValue, Severity, Status


class TestTextNormalisation:
    def test_fullwidth_to_halfwidth(self):
        assert to_halfwidth("ＡＢＣ１２３") == "ABC123"

    def test_normalize_strips_whitespace(self):
        assert normalize(" 价税 合计 ") == "价税合计"

    def test_money_with_currency_and_separators(self):
        assert normalize_money("¥145,318.00") == "145318.00"

    def test_money_plain(self):
        assert normalize_money("145318") == "145318.00"

    def test_money_picks_largest_number_in_verbose_string(self):
        assert normalize_money("145,318.00（大写）壹拾肆万伍仟叁佰壹拾捌元整") == "145318.00"

    def test_money_returns_none_for_chinese_numerals(self):
        assert normalize_money("壹拾肆万伍仟叁佰壹拾捌元整") is None

    def test_date_chinese_format(self):
        assert normalize_date("2026年10月18日") == "2026-10-18"

    def test_date_iso_format(self):
        assert normalize_date("2026-10-18") == "2026-10-18"

    def test_entity_keeps_distinct_company_names_apart(self):
        assert normalize_entity("南京海川软件股份有限公司") != normalize_entity("南京海川信息技术有限公司")

    def test_similarity_of_identical_text(self):
        assert similarity("价税合计", "价税合计") == 1.0

    def test_money_values_compare_equal_after_normalisation(self):
        assert normalize_money("145,318.00") == normalize_money("145318.00")


class TestKindInference:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("价税合计", "money"),
            ("合同金额", "money"),
            ("开票日期", "date"),
            ("购买方名称", "entity"),
            ("发票号码", "text"),
        ],
    )
    def test_infer(self, name, expected):
        assert infer_kind(name) == expected


def _ledger(doc_id: str, fields: list[tuple[str, str]]) -> DocumentLedger:
    ledger = DocumentLedger(doc_id=doc_id, source=f"{doc_id}.png")
    for name, value in fields:
        ledger.fields.append(
            FieldValue(name=name, value=value, region_ids=[f"{doc_id}:r000"], confidence=0.95)
        )
    return ledger


class TestReconcile:
    def test_matching_amounts_are_consistent(self):
        result = reconcile(
            [
                _ledger("contract", [("合同金额", "¥580,000.00")]),
                _ledger("invoice", [("合同金额", "580000.00")]),
            ]
        )
        assert result.consistent
        assert result.matched == 1

    def test_conflicting_amounts_are_detected(self):
        result = reconcile(
            [
                _ledger("contract", [("合同金额", "580,000.00")]),
                _ledger("invoice", [("合同金额", "518,000.00")]),
            ]
        )
        assert not result.consistent
        assert len(result.mismatches) == 1
        assert result.mismatches[0].kind == "money"

    def test_conflicting_entity_names_are_detected(self):
        result = reconcile(
            [
                _ledger("contract", [("乙方", "南京海川软件股份有限公司")]),
                _ledger("invoice", [("乙方", "南京海川信息技术有限公司")]),
            ]
        )
        assert not result.consistent

    def test_single_source_field_is_not_a_conflict(self):
        result = reconcile([_ledger("a", [("金额", "1.00")]), _ledger("b", [("其他", "2.00")])])
        assert result.consistent
        assert "金额" in result.single_source

    def test_conflicting_fields_are_marked_disputed(self):
        contract = _ledger("contract", [("合同金额", "580,000.00")])
        invoice = _ledger("invoice", [("合同金额", "518,000.00")])
        reconcile([contract, invoice])
        assert contract.fields[0].status is Status.DISPUTED
        assert invoice.fields[0].status is Status.DISPUTED

    def test_merge_prefixes_region_ids(self):
        merged = merge_ledgers([_ledger("a", [("金额", "1.00")]), _ledger("b", [("金额", "1.00")])])
        assert all(":" in rid for fld in merged.fields for rid in fld.region_ids)


class TestSumConsistency:
    """单据内部的勾稽校验。

    这类矛盾两边都识别正确，只有加总校验才能发现——OCR 质量模型管不了。
    """

    def _fields(self, total: str, items: list[tuple[str, str]]):
        fields = [FieldValue(name="金额", value=total, region_ids=["r_total"])]
        for index, (label, amount) in enumerate(items):
            fields.append(
                FieldValue(name=f"明细:{label}", value=amount, region_ids=[f"r{index}"])
            )
        return fields

    def test_mismatch_is_reported(self):
        from tracecell.reconcile import check_sum_consistency
        from tracecell.scenarios import ConsistencyRule

        rule = ConsistencyRule(kind="sum", total="金额", components=["明细"])
        fields = self._fields(
            "2,864.50", [("差旅费", "1,480.00"), ("办公用品", "386.50"), ("业务招待费", "1,098.00")]
        )
        findings = check_sum_consistency(fields, [rule])
        assert len(findings) == 1
        assert findings[0].kind == "sum_mismatch"
        assert findings[0].severity is Severity.CRITICAL
        assert "2,964.50" in findings[0].message
        assert "+100.00" in findings[0].message

    def test_consistent_totals_pass(self):
        from tracecell.reconcile import check_sum_consistency
        from tracecell.scenarios import ConsistencyRule

        rule = ConsistencyRule(kind="sum", total="金额", components=["明细"])
        fields = self._fields("1,866.50", [("差旅费", "1,480.00"), ("办公用品", "386.50")])
        assert check_sum_consistency(fields, [rule]) == []

    def test_single_line_item_is_still_checked(self):
        """只有一行明细时，合计仍然应该等于它。"""

        from tracecell.reconcile import check_sum_consistency
        from tracecell.scenarios import ConsistencyRule

        rule = ConsistencyRule(kind="sum", total="金额", components=["明细"])
        fields = self._fields("100.00", [("差旅费", "100.00")])
        assert check_sum_consistency(fields, [rule]) == []

        fields = self._fields("99.00", [("差旅费", "100.00")])
        assert len(check_sum_consistency(fields, [rule])) == 1

    def test_missing_component_is_skipped(self):
        """分项没抽到就不下结论——可能是抽取失败，不是单据有问题。"""

        from tracecell.reconcile import check_sum_consistency
        from tracecell.scenarios import ConsistencyRule

        rule = ConsistencyRule(kind="sum", total="价税合计", components=["金额", "税额"])
        fields = [FieldValue(name="价税合计", value="145,318.00", region_ids=["r0"])]
        assert check_sum_consistency(fields, [rule]) == []

    def test_product_rule_checks_multiplication(self):
        """乘法勾稽：金额 = 数量 × 单价。"""

        from tracecell.reconcile import check_sum_consistency
        from tracecell.scenarios import ConsistencyRule

        rule = ConsistencyRule(kind="product", total="金额", components=["数量", "单价"])
        good = [
            FieldValue(name="数量", value="50"),
            FieldValue(name="单价", value="2,572.00"),
            FieldValue(name="金额", value="128,600.00"),
        ]
        assert check_sum_consistency(good, [rule]) == []

        bad = [
            FieldValue(name="数量", value="50"),
            FieldValue(name="单价", value="2,572.00"),
            FieldValue(name="金额", value="146,318.00"),
        ]
        findings = check_sum_consistency(bad, [rule])
        assert len(findings) == 1
        assert "17,718.00" in findings[0].message

    def test_missing_total_is_skipped(self):
        from tracecell.reconcile import check_sum_consistency
        from tracecell.scenarios import ConsistencyRule

        rule = ConsistencyRule(kind="sum", total="金额", components=["明细"])
        fields = [FieldValue(name="明细:差旅费", value="100.00")]
        assert check_sum_consistency(fields, [rule]) == []


class TestLineItemExtraction:
    def test_line_items_are_extracted_with_label_and_amount(self):
        from tracecell.extractors import LineItemExtractor
        from tracecell.types import BBox, Category, Region

        regions = [
            Region(
                id="r008",
                page=1,
                bbox=BBox(0, 0, 1, 1),
                category=Category.TEXT,
                text="差旅费　杭州至南京项目现场实施差旅　1,480.00",
            ),
            Region(
                id="r011",
                page=1,
                bbox=BBox(0, 0, 1, 1),
                category=Category.TEXT,
                text="合计金额：¥2,864.50",
            ),
        ]
        items = LineItemExtractor().extract(regions)
        assert len(items) == 1
        assert items[0].name == "明细:差旅费"
        assert items[0].value == "1,480.00"

    def test_total_row_is_not_treated_as_a_line_item(self):
        from tracecell.extractors import LineItemExtractor
        from tracecell.types import BBox, Category, Region

        regions = [
            Region(
                id="r011",
                page=1,
                bbox=BBox(0, 0, 1, 1),
                category=Category.TEXT,
                text="合计金额：¥2,864.50",
            )
        ]
        assert LineItemExtractor().extract(regions) == []
