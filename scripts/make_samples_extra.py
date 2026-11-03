#!/usr/bin/env python3
"""生成 TraceCell 的补充样例：财务报表 + 设备采购合同。

和 ``scripts/make_samples.py`` 是同一套思路：**画的时候就把坐标记下来，写
JSON 的时候直接用记下来的坐标**，不做任何二次估算。所以图片上每一段文字在
回放会话里都有一个严丝合缝的 bbox，点字段能直接跳到原图上那一格。

这两个样例驱动的是"跨场景核验"：

* 财务报表的「所有者权益合计」印得又浅又小 —— 印刷难点，重解析读出的数不一样；
* 财务报表的「营业收入」比当期发票金额少 600 元 —— 跨文档矛盾，只能靠对账发现；
* 购销合同的「合同总价」数字转置（146,318 应为 145,318）—— 内部勾稽和跨文档
  对账两条规则同时命中；
* 购销合同的「金额」字号小一号、颜色偏浅 —— 难读区域。

脚本可重复执行：所有随机量都走固定 seed，同一份代码每次产出同样的图片和
同样的 JSON。

    python3 scripts/make_samples_extra.py
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "samples"
SESSIONS_DIR = SAMPLES_DIR / "sessions"

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

PAPER = (250, 250, 247)
INK = (26, 28, 34)
INK_SOFT = (76, 80, 90)
INK_FAINT = (150, 150, 150)
"""难点行专用：浅得几乎要融进纸里，OCR 容易漏读/误读。"""

RULE = (120, 124, 132)
RULE_LIGHT = (186, 190, 198)
FRAME_INK = (66, 70, 78)
SEAL_RED = (200, 40, 40)

PAGE_W = 1240
PAGE_H = 1754

SEED = 20261017

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    key = (path, int(size))
    if key not in _FONT_CACHE:
        _FONT_CACHE[key] = ImageFont.truetype(path, int(size), index=0)
    return _FONT_CACHE[key]


def reg(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return font(FONT_BOLD if bold else FONT_REGULAR, size)


def candidate(
    text: str,
    origin: str,
    score: float | None = None,
    verdict: str = "scored",
    detail: str = "",
) -> dict:
    """一条候选。verdict 不是 scored 时分数没有意义，用 null 表示没评上。"""

    item: dict = {"text": text, "origin": origin, "verdict": verdict}
    item["score"] = float(score) if verdict == "scored" and score is not None else None
    if detail:
        item["detail"] = detail
    return item


class Page:
    """一张纸。画字的时候顺手把文字的实际墨迹范围记下来。"""

    def __init__(self, width: int = PAGE_W, height: int = PAGE_H, bg: tuple = PAPER):
        self.width = width
        self.height = height
        self.img = Image.new("RGB", (width, height), bg)
        self.draw = ImageDraw.Draw(self.img)
        self.regions: list[dict] = []

    # -- 画图基元 ---------------------------------------------------------
    def rect(self, box, outline=None, fill=None, width=1):
        self.draw.rectangle(box, outline=outline, fill=fill, width=width)

    def line(self, x0, y0, x1, y1, fill=RULE, width=1):
        self.draw.line([x0, y0, x1, y1], fill=fill, width=width)

    def text(
        self,
        xy,
        content: str,
        font_obj,
        fill=INK,
        anchor="la",
        category="text",
        candidates=None,
        score=0.95,
        pad=3,
        record=True,
    ):
        """画一段字，记成一个区域。record=False 时只画，不进版面块列表。"""

        self.draw.text(xy, content, font=font_obj, fill=fill, anchor=anchor)
        if not record:
            return None
        box = self.draw.textbbox(xy, content, font=font_obj, anchor=anchor)
        return self._record(box, content, category, candidates, score, pad)

    def text_spaced(
        self,
        center_x: float,
        y: float,
        content: str,
        font_obj,
        fill=INK,
        spacing: float = 12.0,
        category="text",
        candidates=None,
        score=0.95,
        pad=3,
    ):
        """字距拉开一点的标题。"""

        widths = [font_obj.getlength(ch) for ch in content]
        total = sum(widths) + spacing * (len(content) - 1)
        cursor = center_x - total / 2
        boxes = []
        for ch, width in zip(content, widths):
            self.draw.text((cursor, y), ch, font=font_obj, fill=fill, anchor="la")
            boxes.append(self.draw.textbbox((cursor, y), ch, font=font_obj, anchor="la"))
            cursor += width + spacing
        merged = (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
        return self._record(merged, content, category, candidates, score, pad)

    def kv_row(
        self,
        x0: float,
        x1: float,
        y: float,
        label: str,
        value: str,
        font_obj,
        fill=INK,
        value_font=None,
        category="text",
        candidates=None,
        score=0.95,
        pad=3,
        label_pad: float = 14.0,
        value_pad: float = 10.0,
    ):
        """表格里的一行：左边标签，右边数值（金额右对齐）。

        整行记成**一个**区域，文本是 ``标签 + 数值``（标签自带冒号），
        这样字段值可以直接从区域原文里切出来。
        """

        value_font = value_font or font_obj
        self.draw.text((x0 + label_pad, y), label, font=font_obj, fill=fill, anchor="la")
        self.draw.text((x1 - value_pad, y), value, font=value_font, fill=fill, anchor="ra")
        left_box = self.draw.textbbox((x0 + label_pad, y), label, font=font_obj, anchor="la")
        right_box = self.draw.textbbox((x1 - value_pad, y), value, font=value_font, anchor="ra")
        merged = (
            min(left_box[0], right_box[0]),
            min(left_box[1], right_box[1]),
            max(left_box[2], right_box[2]),
            max(left_box[3], right_box[3]),
        )
        return self._record(merged, label + value, category, candidates, score, pad)

    # -- 区域登记 ---------------------------------------------------------
    def _record(self, box, content, category, candidates, score, pad):
        left = max(0, int(round(box[0])) - pad)
        top = max(0, int(round(box[1])) - pad)
        right = min(self.width, int(round(box[2])) + pad)
        bottom = min(self.height, int(round(box[3])) + pad)
        if candidates is None:
            candidates = [candidate(content, "primary", score)]
        region = {
            "id": f"r{len(self.regions):03d}",
            "page": 1,
            "bbox": {
                "x0": round(left / self.width, 6),
                "y0": round(top / self.height, 6),
                "x1": round(right / self.width, 6),
                "y1": round(bottom / self.height, 6),
            },
            "category": category,
            "text": content,
            "candidates": candidates,
            "_pixels": (left, top, right, bottom),
        }
        self.regions.append(region)
        return region

    def seal_region(self, box, content, candidates=None, score=0.6):
        return self._record(box, content, "seal", candidates, score, 0)

    def composite(self, layer: Image.Image) -> None:
        self.img = Image.alpha_composite(self.img.convert("RGBA"), layer).convert("RGB")
        self.draw = ImageDraw.Draw(self.img)


# ---------------------------------------------------------------------------
# 印章
# ---------------------------------------------------------------------------
def star_polygon(cx, cy, outer, inner, points=5, rotation=-90.0):
    coords = []
    for i in range(points * 2):
        radius = outer if i % 2 == 0 else inner
        angle = math.radians(rotation + i * 180.0 / points)
        coords.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return coords


def arc_text(layer, cx, cy, radius, content, font_obj, color, start_deg, end_deg):
    """把文字沿圆弧摆开（章上环绕的单位名称就是这么排的）。"""

    count = len(content)
    for index, ch in enumerate(content):
        angle = start_deg + (end_deg - start_deg) * (index + 0.5) / count
        theta = math.radians(angle)
        x = cx + radius * math.sin(theta)
        y = cy - radius * math.cos(theta)
        bbox = font_obj.getbbox(ch)
        tile = Image.new(
            "RGBA", (bbox[2] - bbox[0] + 12, bbox[3] - bbox[1] + 12), (0, 0, 0, 0)
        )
        tile_draw = ImageDraw.Draw(tile)
        tile_draw.text((6 - bbox[0], 6 - bbox[1]), ch, font=font_obj, fill=color)
        tile = tile.rotate(-angle, resample=Image.BICUBIC, expand=True)
        layer.alpha_composite(tile, (int(x - tile.width / 2), int(y - tile.height / 2)))


def draw_seal(page: Page, cx, cy, radius, name, center_text=""):
    """红色圆形公章。"""

    layer = Image.new("RGBA", (page.width, page.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    red = SEAL_RED + (205,)
    ring = max(3, int(radius * 0.055))
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius], outline=red, width=ring
    )
    inner = radius - ring * 2
    draw.ellipse(
        [cx - inner, cy - inner, cx + inner, cy + inner], outline=red, width=max(1, ring // 3)
    )
    draw.polygon(star_polygon(cx, cy - radius * 0.08, radius * 0.26, radius * 0.11), fill=red)
    if center_text:
        draw.text(
            (cx, cy + radius * 0.46),
            center_text,
            font=font(FONT_BOLD, max(12, int(radius * 0.15))),
            fill=red,
            anchor="mm",
        )
    arc_text(
        layer,
        cx,
        cy,
        radius * 0.70,
        name,
        font(FONT_BOLD, max(13, int(radius * 0.165))),
        red,
        -74.0,
        74.0,
    )
    page.composite(layer)
    return (cx - radius, cy - radius, cx + radius, cy + radius)


# ---------------------------------------------------------------------------
# 纸张质感：轻微噪点 + 自上而下的光照衰减（不影响任何坐标）
# ---------------------------------------------------------------------------
def apply_paper_grain(page: Page, seed: int) -> None:
    array = np.asarray(page.img).astype(np.float32)
    height, width = array.shape[:2]

    yy = np.arange(height, dtype=np.float32).reshape(-1, 1)
    array *= (1.03 - 0.05 * (yy / height))[..., None]

    rng = np.random.default_rng(seed)
    array += rng.normal(0.0, 1.5, array.shape)

    page.img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    page.draw = ImageDraw.Draw(page.img)


# ---------------------------------------------------------------------------
# 一、资产负债表
# ---------------------------------------------------------------------------
FINANCIAL_MD = """# 资产负债表

编制单位：杭州明远网络科技有限公司
报表期间：2026年10月

| 项目 | 期末余额 |
| --- | --- |
| 流动资产合计 | ¥12,860,000.00 |
| 非流动资产合计 | ¥8,140,000.00 |
| 资产总计 | ¥21,000,000.00 |
| 负债合计 | ¥6,800,000.00 |
| 所有者权益合计 | ¥14,100,000.00 |
| 负债和所有者权益总计 | ¥21,000,000.00 |
| 营业收入 | ¥128,000.00 |
| 净利润 | ¥2,460,000.00 |

单位：人民币元"""


def build_financial_statement() -> tuple[Page, list[dict]]:
    page = Page()
    left, right = 130, 1098

    # 标题
    page.text_spaced(
        620,
        140,
        "资产负债表",
        reg(44, bold=True),
        fill=INK,
        spacing=18,
        candidates=[
            candidate("资产负债表", "primary", 0.99),
            candidate("资产负债表", "reparse-16x", 0.97),
        ],
    )
    page.line(left, 232, right, 232, fill=FRAME_INK, width=2)
    page.line(left, 240, right, 240, fill=RULE_LIGHT, width=1)

    # 编制单位 / 报表期间
    r_org = page.text(
        (left, 284),
        "编制单位：杭州明远网络科技有限公司",
        reg(22),
        fill=INK,
        candidates=[candidate("编制单位：杭州明远网络科技有限公司", "primary", 0.98)],
    )
    r_period = page.text(
        (right, 284),
        "报表期间：2026年10月",
        reg(22),
        fill=INK,
        anchor="ra",
        candidates=[candidate("报表期间：2026年10月", "primary", 0.98)],
    )
    page.line(left, 348, right, 348, fill=RULE_LIGHT, width=1)

    # 表头
    page.text(
        (left + 14, 386),
        "项目",
        reg(21, bold=True),
        fill=INK,
        category="table",
        candidates=[candidate("项目", "primary", 0.97)],
    )
    page.text(
        (right, 386),
        "期末余额",
        reg(21, bold=True),
        fill=INK,
        anchor="ra",
        category="table",
        candidates=[candidate("期末余额", "primary", 0.97)],
    )
    page.line(left, 428, right, 428, fill=RULE, width=2)
    # 资产/负债两栏之间的双竖线，财务报表的老规矩
    page.line(636, 428, 636, 1244, fill=RULE_LIGHT, width=1)
    page.line(644, 428, 644, 1244, fill=RULE_LIGHT, width=1)

    rows = [
        # (项目名, 金额, y, 粗体, 浅色, 字号, 额外候选)
        ("流动资产合计：", "¥12,860,000.00", 452, False, False, 22, None),
        ("非流动资产合计：", "¥8,140,000.00", 556, False, False, 22, None),
        ("资产总计：", "¥21,000,000.00", 660, True, False, 22, None),
        ("负债合计：", "¥6,800,000.00", 764, False, False, 22, None),
        (
            "所有者权益合计：",
            "¥14,100,000.00",
            868,
            False,
            True,
            20,
            [
                candidate("所有者权益合计：¥14,100,000.00", "primary", 0.93),
                candidate("所有者权益合计：¥14,100,900.00", "reparse-16x", 0.41),
                candidate(
                    "所有者权益合计：¥14,100,000.00",
                    "vlm-review",
                    0.95,
                    detail="多模态复核确认印字，浅色小字但读数无误",
                ),
            ],
        ),
        ("负债和所有者权益总计：", "¥21,000,000.00", 972, True, False, 22, None),
        (
            "营业收入：",
            "¥128,000.00",
            1080,
            False,
            False,
            22,
            [
                candidate("营业收入：¥128,000.00", "primary", 0.95),
                candidate("营业收入：¥128,600.00", "reparse-16x", 0.22),
            ],
        ),
        ("净利润：", "¥2,460,000.00", 1184, False, False, 22, None),
    ]

    # 分节线：资产总计 / 负债和所有者权益总计 两处加重
    page.line(left, 648, right, 648, fill=RULE, width=2)
    page.line(left, 960, right, 960, fill=RULE, width=2)
    page.line(left, 1052, right, 1052, fill=RULE, width=2)

    row_regions = []
    for label, value, y, bold, faint, size, cands in rows:
        fill = INK_FAINT if faint else INK
        row_regions.append(
            page.kv_row(
                left,
                right,
                y,
                label,
                value,
                reg(size, bold=bold),
                fill=fill,
                candidates=cands,
                score=0.95 if cands is None else 0.93,
            )
        )
    page.line(left, 1244, right, 1244, fill=RULE_LIGHT, width=1)

    # 右下角的单位说明
    page.text(
        (right, 1530),
        "单位：人民币元",
        reg(20),
        fill=INK_SOFT,
        anchor="ra",
        candidates=[candidate("单位：人民币元", "primary", 0.96)],
    )

    page.line(left, 1620, right, 1620, fill=RULE_LIGHT, width=1)
    page.text(
        (620, 1646),
        "第 1 页　共 1 页",
        reg(17),
        fill=INK_SOFT,
        anchor="ma",
        score=0.93,
        record=False,
    )

    (
        r_current,
        r_noncurrent,
        r_assets,
        r_liabilities,
        r_equity,
        r_equity_total,
        r_revenue,
        r_profit,
    ) = row_regions

    fields = [
        {"name": "编制单位", "value": "杭州明远网络科技有限公司", "region_ids": [r_org["id"]]},
        {"name": "报表期间", "value": "2026年10月", "region_ids": [r_period["id"]]},
        {"name": "流动资产合计", "value": "12,860,000.00", "region_ids": [r_current["id"]]},
        {"name": "非流动资产合计", "value": "8,140,000.00", "region_ids": [r_noncurrent["id"]]},
        {"name": "资产总计", "value": "21,000,000.00", "region_ids": [r_assets["id"]]},
        {"name": "负债合计", "value": "6,800,000.00", "region_ids": [r_liabilities["id"]]},
        {"name": "所有者权益合计", "value": "14,100,000.00", "region_ids": [r_equity["id"]]},
        {
            "name": "负债和所有者权益总计",
            "value": "21,000,000.00",
            "region_ids": [r_equity_total["id"]],
        },
        {"name": "营业收入", "value": "128,000.00", "region_ids": [r_revenue["id"]]},
        {"name": "净利润", "value": "2,460,000.00", "region_ids": [r_profit["id"]]},
    ]
    return page, fields


# ---------------------------------------------------------------------------
# 二、设备采购合同
# ---------------------------------------------------------------------------
CONTRACT_MD = """# 设备采购合同

合同编号：CG-2026-0917

需方（盖章）：杭州明远网络科技有限公司
供方（盖章）：南京海川软件股份有限公司

| 项目 | 内容 |
| --- | --- |
| 品名 | 智能网关设备 |
| 数量 | 50 台 |
| 单价 | ¥2,572.00 |
| 金额 | ¥128,600.00 |
| 税率 | 13% |
| 税额 | ¥16,718.00 |
| 合同总价 | ¥146,318.00 |

交付方式：合同生效后 30 日内一次性交付至需方指定地点
签署日期：2026年10月17日"""


def build_purchase_contract() -> tuple[Page, list[dict]]:
    page = Page()
    left, right = 130, 1098

    # 标题
    page.text_spaced(
        620,
        130,
        "设备采购合同",
        reg(42, bold=True),
        fill=INK,
        spacing=16,
        candidates=[
            candidate("设备采购合同", "primary", 0.99),
            candidate("设备采购合同", "reparse-16x", 0.97),
        ],
    )
    r_no = page.text(
        (right, 214),
        "合同编号：CG-2026-0917",
        reg(21),
        fill=INK_SOFT,
        anchor="ra",
        candidates=[candidate("合同编号：CG-2026-0917", "primary", 0.98)],
    )
    page.line(left, 262, right, 262, fill=FRAME_INK, width=2)
    page.line(left, 270, right, 270, fill=RULE_LIGHT, width=1)

    # 需方 / 供方
    r_buyer = page.text(
        (left, 300),
        "需方（盖章）：杭州明远网络科技有限公司",
        reg(21),
        fill=INK,
        candidates=[candidate("需方（盖章）：杭州明远网络科技有限公司", "primary", 0.98)],
    )
    r_seller = page.text(
        (660, 300),
        "供方（盖章）：南京海川软件股份有限公司",
        reg(21),
        fill=INK,
        candidates=[candidate("供方（盖章）：南京海川软件股份有限公司", "primary", 0.98)],
    )
    page.line(left, 364, right, 364, fill=RULE_LIGHT, width=1)

    # 明细表
    page.line(left, 404, right, 404, fill=RULE, width=2)
    page.text(
        (left + 14, 424),
        "项目",
        reg(20, bold=True),
        fill=INK,
        category="table",
        candidates=[candidate("项目", "primary", 0.97)],
    )
    page.text(
        (right, 424),
        "内容",
        reg(20, bold=True),
        fill=INK,
        anchor="ra",
        category="table",
        candidates=[candidate("内容", "primary", 0.97)],
    )
    page.line(left, 466, right, 466, fill=RULE_LIGHT, width=1)

    rows = [
        ("品名：", "智能网关设备", 496, False, False, 21, None),
        ("数量：", "50 台", 570, False, False, 21, None),
        ("单价：", "¥2,572.00", 644, False, False, 21, None),
        (
            "金额：",
            "¥128,600.00",
            718,
            False,
            True,
            19,
            [
                candidate("金额：¥128,600.00", "primary", 0.88),
                candidate("金额：¥128,600.00", "reparse-16x", 0.37),
            ],
        ),
        ("税率：", "13%", 792, False, False, 21, None),
        ("税额：", "¥16,718.00", 866, False, False, 21, None),
        (
            "合同总价：",
            "¥146,318.00",
            940,
            True,
            False,
            22,
            [
                candidate("合同总价：¥146,318.00", "primary", 0.61),
                candidate("合同总价：¥145,318.00", "reparse-16x", 0.18),
                candidate(
                    "合同总价：¥146,318.OO",
                    "reparse-16x",
                    None,
                    verdict="not_scored",
                    detail="Jev 评分服务超时，本轮未返回分数",
                ),
            ],
        ),
    ]

    page.line(left, 928, right, 928, fill=RULE, width=2)

    row_regions = []
    for label, value, y, bold, faint, size, cands in rows:
        fill = INK_FAINT if faint else INK
        row_regions.append(
            page.kv_row(
                left,
                right,
                y,
                label,
                value,
                reg(size, bold=bold),
                fill=fill,
                candidates=cands,
                score=0.95 if cands is None else 0.88,
            )
        )
    # 大写金额。合同以大写为准，这一行是核验小写有没有抄错的依据。
    # 放在合同总价正下方，不挪动其它元素。
    r_total_cn = page.text(
        (left, 976),
        "合同总价（大写）：壹拾肆万伍仟叁佰壹拾捌元整",
        reg(18),
        fill=INK,
        candidates=[
            candidate("合同总价（大写）：壹拾肆万伍仟叁佰壹拾捌元整", "primary", 0.91),
            candidate("合同总价（大写）：壹拾肆万伍仟叁佰壹拾捌圆整", "reparse-16x", 0.44),
        ],
    )

    page.line(left, 1006, right, 1006, fill=RULE, width=2)

    r_delivery = page.text(
        (left, 1046),
        "交付方式：合同生效后 30 日内一次性交付至需方指定地点",
        reg(21),
        fill=INK,
        candidates=[
            candidate("交付方式：合同生效后 30 日内一次性交付至需方指定地点", "primary", 0.96)
        ],
    )

    # 签署区
    page.line(left, 1100, right, 1100, fill=RULE_LIGHT, width=1)
    page.text((left, 1120), "需方（签字）：", reg(20), fill=INK_SOFT, score=0.93, record=False)
    page.text((660, 1120), "供方（签字）：", reg(20), fill=INK_SOFT, score=0.93, record=False)

    seal_buyer = draw_seal(page, 300, 1300, 96, "杭州明远网络科技有限公司", "合同专用章")
    page.seal_region(
        seal_buyer,
        "杭州明远网络科技有限公司 合同专用章",
        candidates=[
            candidate("杭州明远网络科技有限公司 合同专用章", "primary", 0.93),
            candidate("杭州明远网络科技有限公司 合同专用章", "reparse-16x", 0.91),
        ],
    )
    seal_seller = draw_seal(page, 952, 1300, 104, "南京海川软件股份有限公司", "合同专用章")
    page.seal_region(
        seal_seller,
        "南京海川软件股份有限公司 合同专用章",
        candidates=[
            candidate("南京海川软件股份有限公司 合同专用章", "primary", 0.92),
            candidate("南京海川软件股份有限公司 合同专用章", "reparse-16x", 0.90),
        ],
    )

    r_sign_date = page.text(
        (left, 1480),
        "签署日期：2026年10月17日",
        reg(21),
        fill=INK,
        candidates=[candidate("签署日期：2026年10月17日", "primary", 0.97)],
    )

    page.line(left, 1620, right, 1620, fill=RULE_LIGHT, width=1)
    page.text(
        (620, 1646),
        "第 1 页　共 1 页",
        reg(17),
        fill=INK_SOFT,
        anchor="ma",
        score=0.93,
        record=False,
    )

    (
        r_item,
        r_qty,
        r_price,
        r_amount,
        r_tax_rate,
        r_tax,
        r_total,
    ) = row_regions

    fields = [
        {"name": "合同编号", "value": "CG-2026-0917", "region_ids": [r_no["id"]]},
        {"name": "需方", "value": "杭州明远网络科技有限公司", "region_ids": [r_buyer["id"]]},
        {"name": "供方", "value": "南京海川软件股份有限公司", "region_ids": [r_seller["id"]]},
        {"name": "品名", "value": "智能网关设备", "region_ids": [r_item["id"]]},
        {"name": "数量", "value": "50", "region_ids": [r_qty["id"]]},
        {"name": "单价", "value": "2,572.00", "region_ids": [r_price["id"]]},
        {"name": "金额", "value": "128,600.00", "region_ids": [r_amount["id"]]},
        {"name": "税率", "value": "13%", "region_ids": [r_tax_rate["id"]]},
        {"name": "税额", "value": "16,718.00", "region_ids": [r_tax["id"]]},
        {"name": "合同总价", "value": "146,318.00", "region_ids": [r_total["id"]]},
        {
            "name": "合同总价大写",
            "value": "壹拾肆万伍仟叁佰壹拾捌元整",
            "region_ids": [r_total_cn["id"]],
        },
        {"name": "签署日期", "value": "2026年10月17日", "region_ids": [r_sign_date["id"]]},
    ]
    return page, fields


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def strip_pixels(regions: list[dict]) -> list[dict]:
    return [{key: value for key, value in region.items() if key != "_pixels"} for region in regions]


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    financial, financial_fields = build_financial_statement()
    contract, contract_fields = build_purchase_contract()

    apply_paper_grain(financial, SEED)
    apply_paper_grain(contract, SEED + 1)

    financial.img.save(SAMPLES_DIR / "financial_statement_001.png")
    contract.img.save(SAMPLES_DIR / "purchase_contract_001.png")

    session = {
        "documents": [
            {
                "doc_id": "financial-statement-001",
                "source": "samples/financial_statement_001.png",
                "label": "资产负债表",
                "schema": "financial_statement",
                "description": (
                    "所有者权益合计印得偏浅又偏小，重解析读出的数与首次不一致；"
                    "营业收入比当期发票金额少 600 元，属于跨文档矛盾。"
                ),
                "markdown": FINANCIAL_MD,
                "regions": strip_pixels(financial.regions),
                "fields": financial_fields,
            },
            {
                "doc_id": "purchase-contract-001",
                "source": "samples/purchase_contract_001.png",
                "label": "设备采购合同",
                "schema": "purchase_contract",
                "description": (
                    "合同总价数字转置，比金额加税额多 1000 元；"
                    "金额一行字号小一号、颜色偏浅，识别把握不足。"
                ),
                "markdown": CONTRACT_MD,
                "regions": strip_pixels(contract.regions),
                "fields": contract_fields,
            },
        ]
    }
    (SESSIONS_DIR / "extra.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for name, page in (
        ("financial_statement_001.png", financial),
        ("purchase_contract_001.png", contract),
    ):
        print(f"{name}: {page.width}x{page.height}")
    for doc in session["documents"]:
        candidates = sum(len(r["candidates"]) for r in doc["regions"])
        print(
            f"{doc['doc_id']}: regions={len(doc['regions'])} "
            f"fields={len(doc['fields'])} candidates={candidates}"
        )


if __name__ == "__main__":
    main()
