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
