"""策略配置：风险分级表与成本模型。

TraceCell 不把"哪些字段要紧""一次推理值多少钱"写死在代码里。不同的
行业差异极大——合同里的"合同编号"可能无所谓，报销单里的"金额"却要命。
所以这里定义了一组可替换的策略对象，业务方改配置就能适配。

配置可以从 JSON/YAML 读进来，也可以直接用代码构造。默认值针对的是
中文商务单据（发票、合同、报销单、结算单）这一类场景。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .types import Severity

# ---------------------------------------------------------------------------
# 风险分级
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldRule:
    """一条字段风险规则。

    pattern 匹配字段名（子串，忽略大小写）。第一个命中的规则生效，
    所以更精确的规则要排在前面。
    """

    pattern: str
    severity: Severity
    label: str = ""
    scale: float = 1.0
    """同一字段名在不同规模下风险不同时的放大系数，例如金额越大越要紧。"""

    def matches(self, field_name: str) -> bool:
        return self.pattern.lower() in field_name.lower()


# 默认分级表，按"精确 → 宽泛"排列。
DEFAULT_FIELD_RULES: tuple[FieldRule, ...] = (
    # 资金与身份标识：错一位就是真事故
    FieldRule("价税合计", Severity.CRITICAL, "价税合计", 1.6),
    FieldRule("含税金额", Severity.CRITICAL, "含税金额", 1.6),
    FieldRule("总金额", Severity.CRITICAL, "总金额", 1.6),
    FieldRule("金额", Severity.CRITICAL, "金额", 1.2),
    FieldRule("税额", Severity.CRITICAL, "税额", 1.4),
    FieldRule("税率", Severity.CRITICAL, "税率"),
    FieldRule("单价", Severity.HIGH, "单价"),
    FieldRule("账号", Severity.CRITICAL, "银行账号"),
    FieldRule("开户行", Severity.MEDIUM, "开户行"),
    FieldRule("身份证", Severity.CRITICAL, "身份证号"),
    FieldRule("统一社会信用代码", Severity.CRITICAL, "统一社会信用代码"),
    FieldRule("纳税人识别号", Severity.CRITICAL, "纳税人识别号"),
    FieldRule("发票号码", Severity.HIGH, "发票号码"),
    FieldRule("发票代码", Severity.HIGH, "发票代码"),
    FieldRule("合同编号", Severity.HIGH, "合同编号"),
    FieldRule("条款", Severity.HIGH, "条款"),
    FieldRule("结算", Severity.HIGH, "结算"),
    FieldRule("余额", Severity.HIGH, "余额"),
    FieldRule("日期", Severity.HIGH, "日期"),
    FieldRule("签署", Severity.HIGH, "签署日期"),
    FieldRule("甲方", Severity.HIGH, "甲方"),
    FieldRule("乙方", Severity.HIGH, "乙方"),
    FieldRule("名称", Severity.MEDIUM, "名称"),
    FieldRule("地址", Severity.MEDIUM, "地址"),
    FieldRule("联系人", Severity.MEDIUM, "联系人"),
    FieldRule("电话", Severity.MEDIUM, "电话"),
    FieldRule("备注", Severity.LOW, "备注"),
)

# 字段值本身含数字时，按数值大小再放大一档风险。
MAGNITUDE_RULES: tuple[tuple[str, float, float], ...] = (
    # (字段名关键字, 阈值, 放大系数)
    ("金额", 100_000, 1.25),
    ("金额", 1_000_000, 1.5),
    ("合计", 100_000, 1.25),
    ("合计", 1_000_000, 1.5),
)


@dataclass
class RiskPolicy:
    """字段风险分级策略。"""

    rules: tuple[FieldRule, ...] = DEFAULT_FIELD_RULES
    default_severity: Severity = Severity.MEDIUM
    magnitude_scaling: bool = True

    def severity_for(self, field_name: str, value: str = "") -> Severity:
        """给字段定级。

        先按名字匹配规则，再看数值大小是否要升档。只升不降——一个叫"备注"
        的字段就算值很长也不会变成高危。
        """

        severity = self.default_severity
        for rule in self.rules:
            if rule.matches(field_name):
                severity = rule.severity
                break

        if self.magnitude_scaling:
            number = _extract_number(value)
            if number is not None:
                for keyword, threshold, factor in MAGNITUDE_RULES:
                    if keyword in field_name and number >= threshold:
                        severity = _escalate(severity, factor)
                        break
        return severity

    def weight_for(self, severity: Severity) -> float:
        from .types import SEVERITY_WEIGHT

        return SEVERITY_WEIGHT[severity.value]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskPolicy":
        rules = tuple(
            FieldRule(
                pattern=r["pattern"],
                severity=Severity(r["severity"]),
                label=r.get("label", ""),
                scale=float(r.get("scale", 1.0)),
            )
            for r in data.get("rules", [])
        )
        return cls(
            rules=rules or DEFAULT_FIELD_RULES,
            default_severity=Severity(data.get("default_severity", "medium")),
            magnitude_scaling=bool(data.get("magnitude_scaling", True)),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RiskPolicy":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _escalate(severity: Severity, factor: float) -> Severity:
    order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
    index = order.index(severity)
    if factor >= 1.4 and index < 3:
        index += 1
    elif factor > 1.0 and index < 2:
        index += 1
    return order[index]


def _extract_number(value: str) -> float | None:
    """从字段值里取出最大的一串数字。

    金额字段经常写成 "¥18,680.00 (大写) 壹万捌仟陆佰捌拾元整"，取最大的
    连续数字串比逐个解析稳得多。
    """

    if not value:
        return None
    cleaned = value.replace(",", "")
    numbers = re.findall(r"\d+(?:\.\d+)?", cleaned)
    if not numbers:
        return None
    try:
        return max(float(n) for n in numbers)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 勾稽关系
# ---------------------------------------------------------------------------


@dataclass
class SumRule:
    """单据内部的加总关系。

    明细行加起来应该等于合计。识别得再准也发现不了这类矛盾——两边都读对了，
    错的是单据本身，只能靠跨字段校验。

    components 用前缀匹配明细字段名，"明细" 可以匹配 "明细:差旅费"、
    "明细:办公用品" 这些由 LineItemExtractor 抽出来的字段。
    """

    total: str
    components: list[str]
    tolerance: float = 0.01

    def matches(self, field_name: str) -> bool:
        return any(field_name.startswith(prefix) for prefix in self.components)


# 按单据类型声明的勾稽关系
DEFAULT_SUM_RULES: dict[str, SumRule] = {
    "expense": SumRule(total="金额", components=["明细"]),
    "invoice": SumRule(total="价税合计", components=["金额", "税额"]),
}


def sum_rule_for(schema_name: str) -> SumRule | None:
    """取某个单据类型的勾稽规则，没有就返回 None。"""

    return DEFAULT_SUM_RULES.get(schema_name)


# ---------------------------------------------------------------------------
# 成本模型
# ---------------------------------------------------------------------------


@dataclass
class CostModel:
    """一次核验动作的成本，单位是"相对成本单位"（rcu）。

    默认值按同一张消费级显卡上的实测耗时折算：一次整页解析约等于十几次
    区域评分，多模态升级复核最贵，人工复核最贵——但它买来的是确定性。
    这几个数字只决定"值不值得再查一次"的相对关系，业务方应当用自己的
    实测数据重标。
    """

    primary_parse: float = 1.0
    score: float = 0.3
    reparse: float = 1.2
    best_of_n: float = 0.4
    vlm_escalation: float = 3.0
    human_review: float = 8.0

    def cost_of(self, action: str) -> float:
        return {
            "accept": 0.0,
            "score": self.score,
            "reparse": self.reparse,
            "best_of_n": self.best_of_n,
            "vlm_escalation": self.vlm_escalation,
            "human_review": self.human_review,
        }.get(action, 0.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CostModel":
        return cls(**{k: float(v) for k, v in data.items() if k in cls.__annotations__})


# ---------------------------------------------------------------------------
# 预算策略
# ---------------------------------------------------------------------------


@dataclass
class BudgetPolicy:
    """核验预算与停止条件。"""

    total_budget: float = 12.0
    """整份文档允许消耗的额外核验预算（rcu）。"""

    max_rounds: int = 3
    """单个区域最多核验几轮，防止陷入反复重解析。"""

    accept_threshold: float = 0.90
    """Jev 分数高于此值且没有竞争候选时，直接免检。"""

    escalate_threshold: float = 0.55
    """低于此值判定为严重存疑，值得升级到多模态模型。"""

    residual_tolerance: float = 0.05
    """单字段可接受的残余风险。

    残余风险 = 风险权重 × (1 - 置信度)。风险权重按等级取 1.0 / 0.55 /
    0.25 / 0.08，所以默认容忍度下：致命字段要 0.95 的置信度才算免检，
    高危要 0.91，中危要 0.80，低危只要 0.38。

    调低它=更保守（更多转人工），调高它=更激进（更少人工）。财务、合同
    场景建议保持默认或更低；内部流转单据可以适当放宽。
    """

    human_review_ratio: float = 1.0
    """人工复核的成本相对自动核验的倍数，写在 CostModel 里，这里保留扩展位。"""

    cost_model: CostModel = field(default_factory=CostModel)

    def budget_for(self, severity: Severity) -> float:
        """按风险等级分配的局部预算上限。

        高危字段值得多花几次推理，低危字段则倾向于快速放行或者直接交给人。
        """

        share = {
            Severity.CRITICAL.value: 0.45,
            Severity.HIGH.value: 0.30,
            Severity.MEDIUM.value: 0.15,
            Severity.LOW.value: 0.05,
        }[severity.value]
        return self.total_budget * max(share, 0.1)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetPolicy":
        data = dict(data)
        cost = data.pop("cost_model", None)
        policy = cls(
            **{k: v for k, v in data.items() if k in cls.__annotations__ and k != "cost_model"},
        )
        if cost:
            policy.cost_model = CostModel.from_dict(cost)
        return policy
