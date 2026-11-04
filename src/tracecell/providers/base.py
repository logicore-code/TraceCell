"""模型接入层的公共契约。

TraceCell 把三个模型的职责切得很干净：

* **Parser** 负责把图像变成带坐标的文本块（MiniCPM-OCR 1.0）。
* **Scorer** 负责判断某个识别结果对不对（MiniCPM-OCR Jev）。
* **Escalator** 负责在前面两步都没搞定时，用更强的视觉模型去看一眼
  （MiniCPM-V 4.7）。

三个接口都是同步阻塞的，调用方（核验闭环）负责决定什么时候调、调几次。
把决策和执行分开，是为了让策略层可以脱离模型单独测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..types import BBox, Candidate, Category, Region, Verdict

# MiniCPM-OCR 1.0 的任务前缀。
#
# 模型训练时就固定下来的任务指令。换成自然语言问句，模型会答非所问
# （实测过，会输出"请提供需要的信息"）。按版面类别选对前缀是用好它的前提。
TASK_PROMPTS = {
    "text": "Text Recognition:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "seal": "Seal Recognition:",
    "figure": "Chart to Table:",
}


@dataclass
class ParseProfile:
    """一次解析的视觉参数。

    同一块内容在不同参数下会得到不同结果，这是对抗式核验的原材料：
    先用默认参数解析，再用另一组参数重解析一次，候选就凑齐了。
    """

    name: str = "primary"
    downsample_mode: str = "4x"
    """视觉 token 压缩档位。4x 保留更多细节，16x 更快。"""

    max_slice_nums: int = 36
    prompt: str = TASK_PROMPTS["text"]
    padding: float = 0.015
    """裁剪区域时向外扩展的比例，避免切掉笔画。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "downsample_mode": self.downsample_mode,
            "max_slice_nums": self.max_slice_nums,
            "padding": self.padding,
        }


# 常用的一组解析档位。前两个构成最常见的一对"对手"：一个细看，一个快看。
PROFILE_PRIMARY = ParseProfile(name="primary", downsample_mode="4x", max_slice_nums=36)
PROFILE_ALT = ParseProfile(name="reparse-16x", downsample_mode="16x", max_slice_nums=9)


@dataclass
class ScoreResult:
    """一次评分的结果。

    `verdict` 必须被认真对待：官方服务在图像解码失败、渲染失败等情况下
    会返回固定的分数值，含义是"这一步根本没跑成"，不是"模型认为识别错了"。
    TraceCell 在这里就把两种情况分开。
    """

    score: float | None
    verdict: Verdict
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict is Verdict.SCORED and self.score is not None

    def to_candidate(self, origin: str, text: str) -> Candidate:
        return Candidate(
            text=text,
            origin=origin,
            score=self.score,
            verdict=self.verdict,
            detail=self.detail,
        )


@dataclass
class PageParse:
    """一页的解析结果。"""

    regions: list[Region] = field(default_factory=list)
    markdown: str = ""
    raw: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Parser(Protocol):
    """把图像变成带坐标的文本块。"""

    def parse_page(self, image: bytes, profile: ParseProfile) -> PageParse:
        ...

    def parse_region(self, image: bytes, bbox: BBox, profile: ParseProfile) -> str:
        ...


@runtime_checkable
class Scorer(Protocol):
    """给识别结果打分。"""

    def score(self, image: bytes, text: str, category: str) -> ScoreResult:
        ...

    def score_best(
        self, image: bytes, candidates: list[str], category: str
    ) -> list[ScoreResult]:
        ...


@runtime_checkable
class Escalator(Protocol):
    """疑难区域升级复核。"""

    def review(self, image: bytes, question: str, hint: str = "") -> str:
        ...


CATEGORY_TO_JEV = {
    Category.TEXT.value: "text",
    Category.TABLE.value: "table",
    Category.FORMULA.value: "formula",
    Category.SEAL.value: "text",
    Category.FIGURE.value: "text",
}
"""TraceCell 的版面类别到 Jev 支持的三类的映射。

Jev 只认 text / formula / table 三类，印章和图表按文本处理。
"""
