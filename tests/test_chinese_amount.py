"""中文金额解析的测试。

合同金额写两遍，一遍阿拉伯数字一遍中文大写，法律上以大写为准。这两个
数必须对得上，而 OCR 只能告诉你"读到了这串字"，判断不了它值多少钱。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from tracecell.chinese_amount import (
    amounts_agree,
    find_chinese_amount,
    normalize_amount_text,
    parse_chinese_amount,
)


class TestParseChineseAmount:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("伍拾捌万元整", "580000.00"),
            ("伍拾捌万圆整", "580000.00"),
            ("壹拾肆万伍仟叁佰壹拾捌元整", "145318.00"),
            ("壹拾肆万伍仟叁佰壹拾捌圆整", "145318.00"),
            ("壹佰零伍元", "105.00"),
            ("壹仟零伍佰元", "1500.00"),
            ("贰万零伍佰元", "20500.00"),
            ("拾元", "10.00"),
            ("壹拾伍元", "15.00"),
            ("贰拾元整", "20.00"),
            ("壹万元整", "10000.00"),
            ("壹亿贰仟万叁仟元", "120003000.00"),
            ("零元", "0.00"),
        ],
    )
    def test_common_amounts(self, text, expected):
        assert parse_chinese_amount(text) == Decimal(expected)

    def test_jiao_and_fen(self):
        assert parse_chinese_amount("壹佰贰拾叁元肆角伍分") == Decimal("123.45")
        assert parse_chinese_amount("玖仟玖佰玖拾玖元玖角玖分") == Decimal("9999.99")

    def test_returns_none_for_garbage(self):
        assert parse_chinese_amount("") is None
        assert parse_chinese_amount("若干") is None
        assert parse_chinese_amount("hello") is None

    def test_ignores_surrounding_text(self):
        assert parse_chinese_amount("合同金额（大写）：伍拾捌万元整") == Decimal("580000.00")


class TestNormalize:
    def test_yuan_variants(self):
        assert normalize_amount_text("伍拾捌万圆整") == "伍拾捌万元"
        assert normalize_amount_text("壹拾伍元整") == "壹拾伍元"


class TestAmountsAgree:
    def test_matching(self):
        assert amounts_agree("伍拾捌万元整", "¥580,000.00") is True
        assert amounts_agree("伍拾捌万圆整", "580000") is True

    def test_mismatching(self):
        assert amounts_agree("伍拾捌万捌仟元整", "¥580,000.00") is False

    def test_unparsable_returns_none(self):
        assert amounts_agree("若干", "¥580,000.00") is None
        assert amounts_agree("伍拾捌万元整", "若干") is None


class TestFindInText:
    def test_finds_amount_in_a_sentence(self):
        assert find_chinese_amount("甲方应于合同生效后支付人民币伍拾捌万元整。") == Decimal("580000.00")

    def test_returns_none_when_absent(self):
        assert find_chinese_amount("本合同自双方签字盖章之日起生效。") is None
