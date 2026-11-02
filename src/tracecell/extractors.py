"""字段抽取。

账本里的一行一行是从原文里抽出来的业务字段。这里提供两条路：

* ``RegexFieldExtractor``：正则抽取。确定性强、零依赖、离线可跑，适合
  版式稳定的单据（发票、报销单）。字段没抽到就留空，不猜。
* ``KieFieldExtractor``：调 MiniCPM-OCR 1.0 的关键信息抽取能力，按字段名
  提问，返回 JSON（模型对找不到的字段返回 "###"）。适合版式多变的文档。

两条路都遵守同一条规矩：**抽不到就是抽不到**，绝不填占位值。一份台账
里混着编造的字段，比空着更危险。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .types import FieldValue, Region

# 常见单据的字段抽取规则。name 是台账里的字段名，pattern 是要在原文里
# 找的东西，group 是取值用的捕获组编号。
@dataclass
class FieldPattern:
    name: str
    pattern: str
    group: int = 1
    value_group: int = 2
    """取值组与"整个命中"分开，便于把标签和值分别留在证据里。"""

    flags: int = 0


@dataclass
class FieldSchema:
    """一个单据类型的字段表。"""

    name: str
    patterns: list[FieldPattern] = field(default_factory=list)


INVOICE_SCHEMA = FieldSchema(
    name="invoice",
    patterns=[
        FieldPattern("发票号码", r"发票号码[:：]?\s*([0-9]{8,20})"),
        FieldPattern("发票代码", r"发票代码[:：]?\s*([0-9]{10,12})"),
        FieldPattern("开票日期", r"开票日期[:：]?\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)"),
        FieldPattern("购买方名称", r"购买方名称[:：]?\s*([^\n\r]+)"),
        FieldPattern("销售方名称", r"销售方名称[:：]?\s*([^\n\r]+)"),
        FieldPattern("纳税人识别号", r"纳税人识别号[:：]?\s*([0-9A-Z]{15,20})"),
        FieldPattern("金额", r"(?<!合计)(?<!小写)金额[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("税额", r"税额[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("价税合计", r"价税合计[^0-9]*[¥￥]?\s*([\d,]+\.\d{2})"),
    ],
)

CONTRACT_SCHEMA = FieldSchema(
    name="contract",
    patterns=[
        FieldPattern("合同编号", r"合同编号[:：]?\s*([A-Za-z0-9\-]+)"),
        FieldPattern("甲方", r"甲方[（(]?(?:盖章[)）]?)?[:：]?\s*([^\n\r]+)"),
        FieldPattern("乙方", r"乙方[（(]?(?:盖章[)）]?)?[:：]?\s*([^\n\r]+)"),
        FieldPattern("合同金额", r"(?:合同|总|价税)?(?:金额|价款|总价)[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("签署日期", r"签署日期[:：]?\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)"),
        FieldPattern("签署日期", r"签订日期[:：]?\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)"),
        FieldPattern("付款期限", r"付款期限[:：]?\s*([^\n\r]+)"),
    ],
)

EXPENSE_SCHEMA = FieldSchema(
    name="expense",
    patterns=[
        FieldPattern("单据号", r"单据(?:编号|号)[:：]?\s*([A-Za-z0-9\-]+)"),
        FieldPattern("报销人", r"报销人[:：]?\s*([^\n\r]+)"),
        FieldPattern("部门", r"所属部门[:：]?\s*([^\n\r]+)"),
        FieldPattern("费用类型", r"费用类型[:：]?\s*([^\n\r]+)"),
        FieldPattern("金额", r"(?:合计|金额|报销金额)[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("日期", r"日期[:：]?\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)"),
    ],
)

SCHEMAS: dict[str, FieldSchema] = {
    "invoice": INVOICE_SCHEMA,
    "contract": CONTRACT_SCHEMA,
    "expense": EXPENSE_SCHEMA,
}


class RegexFieldExtractor:
    """按正则从区域文本里抽字段。"""

    def __init__(self, schema: FieldSchema) -> None:
        self.schema = schema
        self._compiled = [
            (pattern.name, re.compile(pattern.pattern), pattern.group)
            for pattern in schema.patterns
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
                value = match.group(group).strip() if group <= (match.lastindex or 0) else match.group(0).strip()
                if not value:
                    continue
                fields.append(
                    FieldValue(
                        name=name,
                        value=value,
                        region_ids=[region.id],
                        source=f"regex:{self.schema.name}",
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

    # 行尾是金额，前面是标签
    PATTERN = re.compile(r"([^\d\s，,；;]{2,20})[\s　:：]*[¥￥]?\s*([\d,]+\.\d{2})\s*$")

    # 这些词出现在金额前面时，说明这一行是会计科目或小计，不是明细行
    ACCOUNTING = ("合计", "总计", "总额", "小计", "价税", "金额", "税额", "单价", "税率", "数量", "价税合计")

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
    # 再试一次：把中文引号和尾逗号清掉
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


def schema_by_name(name: str) -> FieldSchema:
    if name not in SCHEMAS:
        raise KeyError(f"未知的单据类型：{name}，可选：{', '.join(SCHEMAS)}")
    return SCHEMAS[name]


class SchemaExtractor:
    """把常规字段抽取和明细行抽取合在一起。

    明细字段单独抽出来，是为了让"明细加起来等不等于合计"这类校验有东西
    可加。字段名前缀是"明细:"，勾稽规则按前缀匹配。
    """

    def __init__(self, schema: FieldSchema, with_line_items: bool = True) -> None:
        self.base = RegexFieldExtractor(schema)
        self.line_items = LineItemExtractor() if with_line_items else None

    def extract(self, regions: list[Region]) -> list[FieldValue]:
        fields = list(self.base.extract(regions))
        if self.line_items is not None:
            fields.extend(self.line_items.extract(regions))
        return fields


def build_extractor(schema_name: str) -> SchemaExtractor:
    return SchemaExtractor(schema_by_name(schema_name))
