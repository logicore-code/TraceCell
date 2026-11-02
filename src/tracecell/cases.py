"""独立应用案例。

进阶赛道的要求是"为所有可用 MiniCPM 模型配置完整应用案例"。完整流水线
（TraceCell 本体）把三个模型串在一起用，但每个模型也都能单独跑成一个
完整案例：

* ``run_ocr_case``  —— 只用 MiniCPM-OCR 1.0：把单据变成带坐标的结构化结果
* ``run_jev_case``  —— 只用 MiniCPM-OCR Jev：给别人的 OCR 结果做质检
* ``run_vlm_case``  —— 只用 MiniCPM-V 4.7：看图问答与视觉定位

三个案例都是"输入 → 处理 → 输出"，可以各自独立运行、独立交付。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .providers import layout as layout_mod
from .providers.base import CATEGORY_TO_JEV, ParseProfile, PROFILE_PRIMARY
from .providers.vlm import build_review_question
from .risk import region_confidence
from .types import DocumentLedger, Region, Verdict


@dataclass
class CaseResult:
    """一个案例的输出。"""

    case: str
    models: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    regions: list[dict[str, Any]] = field(default_factory=list)
    markdown: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case,
            "models": self.models,
            "summary": self.summary,
            "regions": self.regions,
            "markdown": self.markdown,
            "notes": self.notes,
        }


def run_ocr_case(parser, image: bytes, profile: ParseProfile | None = None) -> CaseResult:
    """案例一：只解析，不评分。

    输出每一块文字的坐标和内容，以及整页的 Markdown。适合做"把我的
    扫描件变成带坐标的结构化数据"这类需求的直接交付。
    """

    profile = profile or PROFILE_PRIMARY
    page = parser.parse_page(image, profile)

    regions = [
        {
            "id": region.id,
            "bbox": {
                "x0": region.bbox.x0,
                "y0": region.bbox.y0,
                "x1": region.bbox.x1,
                "y1": region.bbox.y1,
            },
            "category": region.category.value,
            "text": region.text,
        }
        for region in page.regions
    ]
    return CaseResult(
        case="ocr",
        models=["MiniCPM-OCR 1.0"],
        summary={
            "regions": len(regions),
            "chars": sum(len(r["text"]) for r in regions),
            "profile": profile.to_dict(),
        },
        regions=regions,
        markdown=page.markdown,
        notes=[
            "坐标为归一化值（0~1，左上原点），可直接叠加到原图上做高亮",
            "每个 region 对应原图上的一块内容，可单独裁出来送 Jev 评分",
        ],
    )


def run_jev_case(
    scorer,
    parser,
    image: bytes,
    threshold: float = 0.9,
    profile: ParseProfile | None = None,
) -> CaseResult:
    """案例二：只质检，不做决策。

    这是 Jev 最直接的用法，也是绝大多数团队接入 Jev 时的第一件事：把
    别人（或者自己）的 OCR 结果逐块打分，按阈值分成"可自动采用"和
    "需要人工看"两堆。

    和 TraceCell 本体的区别只在于：这里不做预算决策，也不重解析。
    """

    profile = profile or PROFILE_PRIMARY
    page = parser.parse_page(image, profile)

    regions: list[dict[str, Any]] = []
    auto_ok = 0
    need_review = 0
    total_score = 0.0
    scored_count = 0

    for region in page.regions:
        crop = layout_mod.crop(image, region.bbox)
        result = scorer.score(crop, region.text, CATEGORY_TO_JEV.get(region.category.value, "text"))
        entry = {
            "id": region.id,
            "bbox": {
                "x0": region.bbox.x0,
                "y0": region.bbox.y0,
                "x1": region.bbox.x1,
                "y1": region.bbox.y1,
            },
            "category": region.category.value,
            "text": region.text,
            "score": result.score,
            "verdict": result.verdict.value,
            "detail": result.detail,
        }
        regions.append(entry)
        if result.verdict is Verdict.SCORED and result.score is not None:
            scored_count += 1
            total_score += result.score
            if result.score >= threshold:
                auto_ok += 1
            else:
                need_review += 1
        else:
            # 评分没跑成的不能当成低分，只能单独标记
            need_review += 1

    return CaseResult(
        case="jev",
        models=["MiniCPM-OCR 1.0", "MiniCPM-OCR Jev"],
        summary={
            "regions": len(regions),
            "scored": scored_count,
            "auto_accept": auto_ok,
            "need_review": need_review,
            "mean_score": round(total_score / scored_count, 4) if scored_count else None,
            "threshold": threshold,
            "auto_accept_rate": round(auto_ok / len(regions), 4) if regions else 0.0,
        },
        regions=regions,
        notes=[
            f"阈值 {threshold} 是起点，不是结论——阈值必须在自己的标注数据上标",
            "verdict 不是 scored 的区域属于「没跑成」，不能当成低分，会单独列出",
            "这里没有做任何修复；需要修复请用 TraceCell 本体（核验闭环）",
        ],
    )


def run_vlm_case(
    vlm,
    image: bytes,
    question: str = "",
    region: Region | None = None,
) -> CaseResult:
    """案例三：看图问答与视觉定位。

    两种用法：

    * 给整张图提一个问题（默认问题是"把金额类字段抄出来"）；
    * 给一个区域提问，用来处理 OCR 和 Jev 都搞不定的疑难区域。
    """

    if region is not None:
        crop = layout_mod.crop(image, region.bbox)
        answer = vlm.review(crop, question or build_review_question(region.text, []))
        bbox = None
    else:
        crop = image
        answer = vlm.review(crop, question or "请把图中的金额、日期、编号类字段逐一抄写出来，每行一个。")
        bbox = vlm.locate(image, question or "金额")

    return CaseResult(
        case="vlm",
        models=["MiniCPM-V 4.7"],
        summary={
            "question": question,
            "answer": answer,
            "located_bbox": bbox,
            "scope": "region" if region is not None else "page",
        },
        notes=[
            "窄问题的答案可以直接变成新候选，再交给 Jev 复评——这是闭环的关键",
            "MiniCPM-V 4.7 是 35B-A3B 的 MoE 模型，每 token 只激活约 3B，端侧可跑",
        ],
    )


def case_ledger(result: CaseResult, doc_id: str) -> DocumentLedger:
    """把案例结果包成账本，方便复用同一套导出工具。"""

    ledger = DocumentLedger(doc_id=doc_id, source=doc_id, markdown=result.markdown)
    ledger.meta["case"] = result.to_dict()
    ledger.meta["models"] = result.models
    ledger.verdict = "undecided"
    return ledger
