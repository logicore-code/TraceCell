"""场景定义。

一份单据的"场景"包含四样东西：

1. 怎么抽字段
2. 哪些字段危险（风险分级）
3. 单据内部有哪些勾稽关系（明细加总、乘法关系）
4. 跟别的场景怎么对齐（跨文档对账的字段配对）

以前这些散落在 extractors / policy / reconcile 三个模块里，而且写死在代码
中。加一种新单据要改三个地方，很容易漏。这里把它们收拢到一个注册表里，
加场景只需要写一份配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import Severity


@dataclass
class FieldPattern:
    """一个字段的抽取规则。"""

    name: str
    pattern: str
    group: int = 1


@dataclass
class ConsistencyRule:
    """单据内部的一条勾稽关系。

    kind="sum"      总数 = 各分项之和
    kind="product"  总数 = 各因子之积（数量 × 单价 = 金额）
    """

    kind: str
    total: str
    components: list[str]
    tolerance: float = 0.01

    def matches(self, field_name: str) -> bool:
        return field_name.startswith(self.total) or field_name == self.total


@dataclass
class CrossRule:
    """跨场景的一条字段配对。

    left/right 是两个场景里的字段名，kind 决定归一化方式。金额对金额，
    主体对主体，日期对日期。
    """

    left_scenario: str
    left_field: str
    right_scenario: str
    right_field: str
    kind: str = "money"


@dataclass
class Scenario:
    """一个场景的完整定义。"""

    name: str
    label: str
    fields: list[FieldPattern] = field(default_factory=list)
    rules: list[ConsistencyRule] = field(default_factory=list)
    risk_overrides: dict[str, Severity] = field(default_factory=dict)
    description: str = ""

    def severity_for(self, field_name: str) -> Severity | None:
        """场景级的风险覆盖。返回 None 表示交给默认分级表。"""

        return self.risk_overrides.get(field_name)


# ---------------------------------------------------------------------------
# 发票
# ---------------------------------------------------------------------------

INVOICE = Scenario(
    name="invoice",
    label="增值税发票",
    description="价税合计 = 金额 + 税额",
    fields=[
        FieldPattern("发票代码", r"发票代码[:：]?\s*([0-9]{8,20})"),
        FieldPattern("发票号码", r"发票号码[:：]?\s*([0-9]{8,20})"),
        FieldPattern("开票日期", r"开票日期[:：]?\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)"),
        FieldPattern("购买方名称", r"购买方名称[:：]?\s*([^\n\r]+)"),
        FieldPattern("销售方名称", r"销售方名称[:：]?\s*([^\n\r]+)"),
        FieldPattern("纳税人识别号", r"纳税人识别号[:：]?\s*([0-9A-Z]{15,20})"),
        FieldPattern("金额", r"(?<!合计)(?<!小写)金额[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("税额", r"税额[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("价税合计", r"价税合计[^0-9]*[¥￥]?\s*([\d,]+\.\d{2})"),
    ],
    rules=[
        ConsistencyRule("sum", "价税合计", ["金额", "税额"]),
    ],
)

# ---------------------------------------------------------------------------
# 服务合同
# ---------------------------------------------------------------------------

CONTRACT = Scenario(
    name="contract",
    label="技术服务合同",
    description="大小写金额互为印证，靠跨文档与发票比对",
    fields=[
        FieldPattern("合同编号", r"合同编号[:：]?\s*([A-Za-z0-9\-]+)"),
        FieldPattern("甲方", r"甲方[（(]?(?:盖章[)）]?)?[:：]?\s*([^\n\r]+)"),
        FieldPattern("乙方", r"乙方[（(]?(?:盖章[)）]?)?[:：]?\s*([^\n\r]+)"),
        FieldPattern("合同金额", r"(?:合同|总|价税)?(?:金额|价款|总价)[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("签署日期", r"签署日期[:：]?\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)"),
        FieldPattern("签署日期", r"签订日期[:：]?\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)"),
        FieldPattern("付款期限", r"付款期限[:：]?\s*([^\n\r]+)"),
    ],
    rules=[],
)

# ---------------------------------------------------------------------------
# 报销单
# ---------------------------------------------------------------------------

EXPENSE = Scenario(
    name="expense",
    label="员工报销单",
    description="明细加总 = 合计",
    fields=[
        FieldPattern("单据号", r"单据(?:编号|号)[:：]?\s*([A-Za-z0-9\-]+)"),
        FieldPattern("报销人", r"报销人[:：]?\s*([^\n\r]+)"),
        FieldPattern("部门", r"所属部门[:：]?\s*([^\n\r]+)"),
        FieldPattern("费用类型", r"费用类型[:：]?\s*([^\n\r]+)"),
        FieldPattern("金额", r"(?:合计|金额|报销金额)[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("日期", r"日期[:：]?\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)"),
    ],
    rules=[
        ConsistencyRule("sum", "金额", ["明细"]),
    ],
    risk_overrides={
        "金额": Severity.CRITICAL,
        "报销人": Severity.HIGH,
        "部门": Severity.MEDIUM,
        "费用类型": Severity.MEDIUM,
        "单据号": Severity.HIGH,
        "日期": Severity.HIGH,
    },
)

# ---------------------------------------------------------------------------
# 财务报表
# ---------------------------------------------------------------------------

FINANCIAL_STATEMENT = Scenario(
    name="financial_statement",
    label="资产负债表",
    description="资产总计 = 流动资产合计 + 非流动资产合计；资产 = 负债 + 所有者权益",
    fields=[
        FieldPattern("编制单位", r"编制单位[:：]?\s*([^\n\r]+)"),
        FieldPattern("报表期间", r"报表期间[:：]?\s*([^\n\r]+)"),
        FieldPattern("流动资产合计", r"流动资产合计[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("非流动资产合计", r"非流动资产合计[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("资产总计", r"资产总计[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("负债合计", r"负债合计[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("所有者权益合计", r"所有者权益合计[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("负债和所有者权益总计", r"负债和所有者权益总计[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("营业收入", r"营业收入[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("净利润", r"净利润[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
    ],
    rules=[
        ConsistencyRule("sum", "资产总计", ["流动资产合计", "非流动资产合计"]),
        ConsistencyRule("sum", "资产总计", ["负债合计", "所有者权益合计"]),
    ],
    risk_overrides={
        "资产总计": Severity.CRITICAL,
        "负债合计": Severity.CRITICAL,
        "所有者权益合计": Severity.CRITICAL,
        "负债和所有者权益总计": Severity.CRITICAL,
        "营业收入": Severity.CRITICAL,
        "净利润": Severity.CRITICAL,
        "流动资产合计": Severity.HIGH,
        "非流动资产合计": Severity.HIGH,
        "报表期间": Severity.HIGH,
        "编制单位": Severity.MEDIUM,
    },
)

# ---------------------------------------------------------------------------
# 购销合同
# ---------------------------------------------------------------------------

PURCHASE_CONTRACT = Scenario(
    name="purchase_contract",
    label="设备采购合同",
    description="金额 = 数量 × 单价；合同总价 = 金额 + 税额",
    fields=[
        FieldPattern("合同编号", r"合同编号[:：]?\s*([A-Za-z0-9\-]+)"),
        FieldPattern("需方", r"需方[（(]?(?:盖章[)）]?)?[:：]?\s*([^\n\r]+)"),
        FieldPattern("供方", r"供方[（(]?(?:盖章[)）]?)?[:：]?\s*([^\n\r]+)"),
        FieldPattern("品名", r"品名[:：]?\s*([^\n\r]+)"),
        FieldPattern("数量", r"数量[:：]?\s*([\d,]+(?:\.\d+)?)"),
        FieldPattern("单价", r"单价[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("金额", r"(?<!合同总价)(?<!税额)金额[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("税率", r"税率[:：]?\s*(\d+(?:\.\d+)?%)"),
        FieldPattern("税额", r"税额[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("合同总价", r"合同总价[:：]?\s*[¥￥]?\s*([\d,]+\.\d{2})"),
        FieldPattern("签署日期", r"签署日期[:：]?\s*(\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日)"),
    ],
    rules=[
        ConsistencyRule("product", "金额", ["数量", "单价"]),
        ConsistencyRule("sum", "合同总价", ["金额", "税额"]),
    ],
    risk_overrides={
        "合同总价": Severity.CRITICAL,
        "金额": Severity.CRITICAL,
        "税额": Severity.CRITICAL,
        "单价": Severity.HIGH,
        "税率": Severity.HIGH,
        "数量": Severity.HIGH,
        "需方": Severity.HIGH,
        "供方": Severity.HIGH,
        "签署日期": Severity.HIGH,
        "品名": Severity.MEDIUM,
    },
)


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in (INVOICE, CONTRACT, EXPENSE, FINANCIAL_STATEMENT, PURCHASE_CONTRACT)
}


# ---------------------------------------------------------------------------
# 跨场景配对
# ---------------------------------------------------------------------------

# 同一笔业务在不同单据上的落点。这张表是跨文档对账的依据——靠字段名撞车
# 是撞不出来的（合同叫"乙方"，发票叫"销售方名称"）。
CROSS_RULES: list[CrossRule] = [
    CrossRule("purchase_contract", "合同总价", "invoice", "价税合计", "money"),
    CrossRule("purchase_contract", "金额", "invoice", "金额", "money"),
    CrossRule("purchase_contract", "需方", "invoice", "购买方名称", "entity"),
    CrossRule("purchase_contract", "供方", "invoice", "销售方名称", "entity"),
    CrossRule("contract", "甲方", "invoice", "购买方名称", "entity"),
    CrossRule("contract", "乙方", "invoice", "销售方名称", "entity"),
    CrossRule("contract", "合同金额", "invoice", "价税合计", "money"),
    CrossRule("financial_statement", "营业收入", "invoice", "金额", "money"),
]


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"未知场景：{name}，可选：{', '.join(SCENARIOS)}")
    return SCENARIOS[name]


def list_scenarios() -> list[dict[str, Any]]:
    """给 CLI 和界面用的场景清单。"""

    return [
        {
            "name": s.name,
            "label": s.label,
            "description": s.description,
            "field_count": len(s.fields),
            "rule_count": len(s.rules),
        }
        for s in SCENARIOS.values()
    ]


def cross_rules_for(scenarios: list[str]) -> list[CrossRule]:
    """取出涉及给定场景的跨场景配对。

    一条规则只有两端场景都在场时才参与比对。
    """

    present = set(scenarios)
    return [
        rule
        for rule in CROSS_RULES
        if rule.left_scenario in present and rule.right_scenario in present
    ]
