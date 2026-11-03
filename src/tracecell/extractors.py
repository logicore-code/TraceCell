"""字段抽取。

账本里的业务字段是从原文里抽出来的。这里提供三条路：

* ``RegexFieldExtractor``：按场景的正则抽。确定性强、零依赖、离线可跑，
  适合版式稳定的单据（发票、合同、报销单、报表、购销合同）。
* ``LineItemExtractor``：把表格里的明细行抽成字段。字段名带 ``明细:``
  前缀，勾稽校验要用。
* ``KieFieldExtractor``：调 MiniCPM-OCR 1.0 的关键信息抽取能力，按字段名
  提问，返回 JSON。适合版式多变的文档。

三条路都遵守同一条规矩：**抽不到就是抽不到**，绝不填占位值。一份台账
里混着编造的字段，比空着更危险。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .scenarios import SCENARIOS, FieldPattern, Scenario, get_scenario
from .types import FieldValue, Region

__all__ = [
    "FieldPattern",
    "LineItemExtractor",
    "KieFieldExtractor",
    "RegexFieldExtractor",
    "SchemaExtractor",
    "build_extractor",
    "schema_by_name",
]


class RegexFieldExtractor:
    """按场景的正则从区域文本里抽字段。"""

    def __init__(self, patterns: list[FieldPattern] | Scenario) -> None:
        self.patterns = patterns.fields if isinstance(patterns, Scenario) else patterns
        self._compiled = [
            (pattern.name, re.compile(pattern.pattern), pattern.group)
            for pattern in self.patterns
        ]

    def extract(self, regions: list[Region]) -> list[FieldValue]:
        fields: list[FieldValue] = []
        seen: set[str] = set()
        for region in regions:
            text = region.selected_text or region.text
            if not text:
                continue
            for name, regex, group in self._compiled:
                if name in seen:
                    continue
                match = regex.search(text)
                if not match:
                    continue
                value = (
                    match.group(group).strip()
                    if group <= (match.lastindex or 0)
                    else match.group(0).strip()
                )
                if not value:
                    continue
                fields.append(
                    FieldValue(
                        name=name,
                        value=value,
                        region_ids=[region.id],
                        source="regex",
                    )
                )
                seen.add(name)
        return fields


class LineItemExtractor:
    """把表格里的明细行抽成字段。

    单据上的明细行长得都差不多：一段标签，加一个金额。例如

        差旅费　杭州至南京项目现场实施差旅　1,480.00

    抽出来是 {"name": "明细:差旅费", "value": "1,480.00"}。有了明细字段，
    才能校验"明细加起来等不等于合计"——这类矛盾两边都识别正确，只有加总
    校验才能发现。
    """

    # 这些词出现在金额前面时，说明这一行是会计科目或小计，不是明细行
    ACCOUNTING = ("合计", "总计", "总额", "小计", "价税", "金额", "税额", "单价", "税率", "数量")

    PATTERN = re.compile(r"([^\d\s，,；;]{2,20})[\s　:：]*[¥￥]?\s*([\d,]+\.\d{2})\s*$")

    def __init__(self, exclude: tuple[str, ...] = ("合计", "总计", "总额", "价税合计")) -> None:
        self.exclude = exclude

    def extract(self, regions: list[Region]) -> list[FieldValue]:
        fields: list[FieldValue] = []
        seen: set[str] = set()
        for region in regions:
            text = (region.selected_text or region.text or "").strip()
            if not text or any(keyword in text for keyword in self.exclude):
                continue
            match = self.PATTERN.search(text.replace("\t", " "))
            if not match:
                continue
            # 明细名取整行里第一个分隔符之前的那一段：
            # "差旅费　杭州至南京项目现场实施差旅　1,480.00" → "差旅费"
            head = text[: match.start()].strip(" :：\t")
            parts = [part for part in re.split(r"[\s　:：]+", head) if part]
            label = parts[0].strip() if parts else ""
            if not label or label in seen:
                continue
            if any(keyword in label for keyword in self.ACCOUNTING):
                continue
            if re.fullmatch(r"[\d.,%¥￥]+", label):
                continue
            seen.add(label)
            fields.append(
                FieldValue(
                    name=f"明细:{label}",
                    value=match.group(2),
                    region_ids=[region.id],
                    source="line-item",
                )
            )
        return fields


class KieFieldExtractor:
    """调用 MiniCPM-OCR 的关键信息抽取能力。"""

    def __init__(self, request: Callable[[str], str], fields: list[str]) -> None:
        self.request = request
        self.fields = fields

    def extract(self, regions: list[Region]) -> list[FieldValue]:
        if not self.fields:
            return []
        joined = "\n".join(region.selected_text or region.text for region in regions)
        prompt = (
            "请从以下文本中抽取指定字段，以 JSON 返回，"
            '格式为 {"字段名": "字段值"}。'
            "文本中不存在的字段请填 \"###\"。\n"
            f"需要抽取的字段：{'、'.join(self.fields)}\n"
            f"文本：\n{joined}"
        )
        answer = self.request(prompt)
        payload = _parse_json_object(answer)
        result: list[FieldValue] = []
        for name in self.fields:
            value = str(payload.get(name, "") or "").strip()
            if not value or value == "###":
                continue
            result.append(FieldValue(name=name, value=value, source="kie"))
        return result


@dataclass
class SchemaExtractor:
    """把常规字段抽取和明细行抽取合在一起。

    明细字段单独抽出来，是为了让"明细加起来等不等于合计"这类校验有东西
    可加。字段名前缀是"明细:"，勾稽规则按前缀匹配。
    """

    scenario: Scenario
    with_line_items: bool = True

    def __init__(self, scenario: Scenario, with_line_items: bool = True) -> None:
        self.scenario = scenario
        self.with_line_items = with_line_items
        self.base = RegexFieldExtractor(scenario.fields)
        self.line_items = LineItemExtractor() if with_line_items else None

    def extract(self, regions: list[Region]) -> list[FieldValue]:
        fields = list(self.base.extract(regions))
        if self.line_items is not None:
            fields.extend(self.line_items.extract(regions))
        return fields


def _parse_json_object(text: str) -> dict[str, Any]:
    """从模型回答里抠出 JSON 对象。

    模型经常在 JSON 外面裹一层 ```json 或者加一句说明，这里都能兜住。
    """

    if not text:
        return {}
    match = re.search(r"\{.*\}", text, re.S)
    candidate = match.group(0) if match else text
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass
    cleaned = (
        candidate.replace("“", '"')
        .replace("”", '"')
        .replace("，}", "}")
        .replace(",}", "}")
    )
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def build_extractor(scenario_name: str) -> SchemaExtractor:
    """按场景名造抽取器。"""

    return SchemaExtractor(get_scenario(scenario_name))


def schema_by_name(name: str) -> Scenario:
    """兼容旧调用方式：按名字取场景。"""

    return get_scenario(name)
