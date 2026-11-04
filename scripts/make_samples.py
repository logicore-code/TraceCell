#!/usr/bin/env python3
"""生成 TraceCell 的演示样例：合成单据图片 + 与之严格对应的回放会话。

图片和 JSON 必须严丝合缝：图片上每一段文字在 JSON 里都有一个同样大小的
bbox，点字段能直接跳到原图上那一格。所以这里的做法是——**画的时候就把
坐标记下来，写 JSON 的时候直接用记下来的坐标**，不做任何二次估算。

三个演示点：

* 印刷难点（浅灰小字）—— 机器读对了，但自己也拿不准，分数低；
* 候选打架 —— 同一块区域，两次识别给出两个互不相同的值；
* 跨文档矛盾 —— 合同上的乙方名称和发票上的销售方名称对不上。

脚本可重复执行：所有随机量都走固定 seed，同一份代码每次产出同样的
图片和同样的 JSON。

    python3 scripts/make_samples.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "src"))
from tracecell.recorded import apply_scores, load_scores  # noqa: E402

SAMPLES_DIR = ROOT / "samples"
SESSIONS_DIR = SAMPLES_DIR / "sessions"

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_HAND = "/usr/share/fonts/truetype/arphic/ukai.ttc"

PAPER = (250, 250, 247)
INK = (26, 28, 34)
INK_SOFT = (76, 80, 90)
INK_FAINT = (164, 168, 178)
RULE = (120, 124, 132)
RULE_LIGHT = (186, 190, 198)
FRAME_INK = (66, 70, 78)
SEAL_RED = (200, 40, 40)
SIGN_INK = (36, 52, 138)

PAGE_W = 1240
PAGE_H = 1754

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

    def _run(self, x, y, content, font_obj, fill, hand_seed=None, jitter=2.4):
        """画一段文字，返回 (推进宽度, 墨迹外框)。"""

        if hand_seed is None:
            self.draw.text((x, y), content, font=font_obj, fill=fill, anchor="la")
            box = self.draw.textbbox((x, y), content, font=font_obj, anchor="la")
            return font_obj.getlength(content), box

        rng = random.Random(hand_seed)
        boxes = []
        cursor = x
        for ch in content:
            dx = rng.uniform(-jitter, jitter)
            dy = rng.uniform(-jitter * 1.5, jitter * 1.5)
            tilt = rng.uniform(-2.8, 2.8)
            bbox = font_obj.getbbox(ch)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            tile = Image.new("RGBA", (width + 12, height + 12), (0, 0, 0, 0))
            tile_draw = ImageDraw.Draw(tile)
            tile_draw.text((6 - bbox[0], 6 - bbox[1]), ch, font=font_obj, fill=fill)
            tile = tile.rotate(tilt, resample=Image.BICUBIC, center=(width / 2 + 6, height / 2 + 6))
            px = int(cursor + dx)
            py = int(y + dy)
            self.img.paste(tile, (px, py), tile)
            self.draw = ImageDraw.Draw(self.img)
            boxes.append((px, py, px + tile.width, py + tile.height))
            cursor += font_obj.getlength(ch) * rng.uniform(0.97, 1.03)
        merged = (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
        return cursor - x, merged

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
        box = self.draw.textbbox(xy, content, font=font_obj, anchor=anchor)
        if not record:
            return None
        return self._record(box, content, category, candidates, score, pad)

    def text_runs(
        self,
        x: float,
        y: float,
        runs: list[dict],
        category="text",
        candidates=None,
        score=0.95,
        pad=3,
    ):
        """同一行里用不同字体/颜色/手写体拼出来的文字，整体记成一个区域。

        每个 run 是 {"text":..., "font":..., "fill":..., "hand": seed|None}。
        """

        boxes = []
        cursor = x
        for run in runs:
            advance, box = self._run(
                cursor,
                y,
                run["text"],
                run["font"],
                run.get("fill", INK),
                hand_seed=run.get("hand"),
            )
            boxes.append(box)
            cursor += advance
        merged = (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
        joined = "".join(run["text"] for run in runs)
        return self._record(merged, joined, category, candidates, score, pad)

    def text_lines(
        self,
        x: float,
        y: float,
        lines: list[str],
        font_obj,
        fill=INK,
        leading: float = 1.75,
        category="text",
        candidates=None,
        score=0.95,
        pad=3,
        record=True,
    ):
        """多行段落，整体记成一个区域（原文里带换行）。"""

        boxes = []
        step = leading * font_obj.size
        for index, line in enumerate(lines):
            top = y + index * step
            self.draw.text((x, top), line, font=font_obj, fill=fill, anchor="la")
            boxes.append(self.draw.textbbox((x, top), line, font=font_obj, anchor="la"))
        merged = (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
        if not record:
            return None
        return self._record(merged, "\n".join(lines), category, candidates, score, pad)

    def table_row(
        self,
        cells: list[tuple[str, float, float, str]],
        y: float,
        font_obj,
        fill=INK,
        category="text",
        candidates=None,
        score=0.95,
        pad=3,
        separator="　",
        record=True,
    ):
        """一行表格。每一格画在自己的列里，整行记成一个版面块。

        cells 是 [(文字, 列左边界, 列右边界, 对齐)]，对齐取 l/c/r。
        """

        boxes = []
        joined = []
        for content, x0, x1, align in cells:
            if align == "r":
                x, anchor = x1 - 10, "ra"
            elif align == "c":
                x, anchor = (x0 + x1) / 2, "ma"
            else:
                x, anchor = x0 + 8, "la"
            self.draw.text((x, y), content, font=font_obj, fill=fill, anchor=anchor)
            boxes.append(self.draw.textbbox((x, y), content, font=font_obj, anchor=anchor))
            joined.append(content)
        merged = (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
        if not record:
            return None
        return self._record(merged, separator.join(joined), category, candidates, score, pad)

    def text_spaced(
        self,
        center_x: float,
        y: float,
        content: str,
        font_obj,
        fill=INK,
        spacing: float = 10.0,
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
        for ch, width in zip(content, widths, strict=True):
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


def draw_rect_stamp(page: Page, cx, cy, width, height, content, angle=-7.0):
    """矩形审批章（用印泥章，不是公章）。"""

    layer = Image.new("RGBA", (page.width, page.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    red = SEAL_RED + (185,)
    draw.rounded_rectangle(
        [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
        radius=10,
        outline=red,
        width=4,
    )
    draw.text(
        (cx, cy),
        content,
        font=font(FONT_BOLD, max(14, int(height * 0.30))),
        fill=red,
        anchor="mm",
    )
    layer = layer.rotate(angle, resample=Image.BICUBIC, center=(cx, cy))
    page.composite(layer)
    return (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


# ---------------------------------------------------------------------------
# 翻拍增强（旋转 / 噪声 / 光照不均）
# ---------------------------------------------------------------------------
def rotate_box(box, angle_deg, width, height):
    cx, cy = width / 2.0, height / 2.0
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    points = []
    for x, y in ((box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])):
        dx, dy = x - cx, y - cy
        points.append((cx + dx * cos_t + dy * sin_t, cy - dx * sin_t + dy * cos_t))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (
        max(0.0, min(xs)),
        max(0.0, min(ys)),
        min(float(width), max(xs)),
        min(float(height), max(ys)),
    )


def apply_photo_effects(page: Page, angle_deg: float, noise: float, seed: int, light=(0.42, 0.3)):
    """模拟手机翻拍：轻微旋转 + 亮度不均 + 噪点。坐标跟着一起转。"""

    rotated = page.img.rotate(angle_deg, resample=Image.BICUBIC, fillcolor=PAPER, expand=False)
    array = np.asarray(rotated).astype(np.float32)
    height, width = array.shape[:2]

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = width * light[0]
    cy = height * light[1]
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (width * 0.9)
    gradient = 1.06 - 0.16 * distance ** 1.4
    array *= gradient[..., None]

    rng = np.random.default_rng(seed)
    array += rng.normal(0.0, noise, array.shape)

    page.img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
    page.draw = ImageDraw.Draw(page.img)

    for region in page.regions:
        box = rotate_box(region["_pixels"], angle_deg, width, height)
        region["_pixels"] = box
        region["bbox"] = {
            "x0": round(box[0] / width, 6),
            "y0": round(box[1] / height, 6),
            "x1": round(box[2] / width, 6),
            "y1": round(box[3] / height, 6),
        }


# ---------------------------------------------------------------------------
# 一、增值税专用发票
# ---------------------------------------------------------------------------
INVOICE_MD = """# 浙江增值税专用发票

发票代码：251170000000
发票号码：25117000000012345678

购买方名称：杭州明远网络科技有限公司
纳税人识别号：91330100MA2XXXXX1J
地址、电话：杭州市西湖区文三路 398 号 6 幢 12 层
开户行及账号：中国银行杭州文三路支行 3301010036000088

销售方名称：南京海川软件股份有限公司
纳税人识别号：91320100MA1XXXXX9K
地址、电话：南京市建邺区江东中路 106 号 8 幢 501 室
开户行及账号：中国银行南京建邺支行 1020001100880032

货物或应税劳务、服务名称：技术开发服务*信息技术服务
货物或应税劳务、服务名称：系统运维服务*信息技术服务
金额 ¥128,600.00　税率 13%　税额 ¥16,718.00
价税合计（大写）壹拾肆万伍仟叁佰壹拾捌圆整
价税合计（小写）¥145,318.00

备注：本次服务费对应合同 HT-2026-1043，账期 30 天。
开票日期：2026年10月18日
收款人：李敏　复核：王岚　开票人：陈涛"""


def build_invoice() -> tuple[Page, list[dict]]:
    page = Page()
    left, right = 110, 1130

    # 外框
    page.rect((72, 56, 1168, 1698), outline=FRAME_INK, width=3)
    page.rect((86, 70, 1154, 1684), outline=RULE_LIGHT, width=1)

    # 标题
    page.text_spaced(
        620,
        92,
        "浙江增值税专用发票",
        reg(46, bold=True),
        fill=INK,
        spacing=14,
        candidates=[
            candidate("浙江增值税专用发票", "primary", 0.99),
            candidate("浙江增值税专用发票", "reparse-16x", 0.97),
        ],
    )
    page.line(left, 168, right, 168, fill=FRAME_INK, width=3)
    page.line(left, 176, right, 176, fill=RULE_LIGHT, width=1)

    # 发票代码 / 发票号码
    r_code = page.text(
        (left + 2, 192),
        "发票代码：251170000000",
        reg(23),
        fill=INK_SOFT,
        candidates=[candidate("发票代码：251170000000", "primary", 0.98)],
    )
    r_number = page.text(
        (right - 2, 192),
        "发票号码：25117000000012345678",
        reg(23),
        fill=INK_SOFT,
        anchor="ra",
        candidates=[
            candidate("发票号码：25117000000012345678", "primary", 0.99),
            candidate("发票号码：25117000000012345678", "reparse-16x", 0.96),
        ],
    )
    page.line(left, 232, right, 232, fill=RULE, width=1)

    # 购销双方
    page.line(620, 244, 620, 520, fill=RULE_LIGHT, width=1)
    buyer = [
        ("购买方名称：杭州明远网络科技有限公司", 0.97),
        ("纳税人识别号：91330100MA2XXXXX1J", 0.96),
        ("地址、电话：杭州市西湖区文三路 398 号 6 幢 12 层", 0.94),
        ("开户行及账号：中国银行杭州文三路支行 3301010036000088", 0.95),
    ]
    seller = [
        (
            "销售方名称：南京海川软件股份有限公司",
            [
                # OCR 把「股份」自信地读成了「股分」。分数很高，读得很像样，
                # 但错——抓住它靠的是评分和跨文档对账，不是更长的上下文。
                candidate("销售方名称：南京海川软件股分有限公司", "primary", 0.91),
                candidate("销售方名称：南京海川软件股份有限公司", "reparse-16x", 0.46),
                candidate(
                    "销售方名称：南京海川软件股分有限公同",
                    "reparse-16x",
                    None,
                    verdict="not_scored",
                    detail="Jev 评分服务超时，本轮未返回分数",
                ),
            ],
        ),
        ("纳税人识别号：91320100MA1XXXXX9K", 0.97),
        ("地址、电话：南京市建邺区江东中路 106 号 8 幢 501 室", 0.93),
        ("开户行及账号：中国银行南京建邺支行 1020001100880032", 0.92),
    ]
    r_buyer_name = r_buyer_tax = r_seller_name = None
    for index in range(4):
        y = 254 + index * 62
        text, score = buyer[index]
        region = page.text(
            (left + 2, y), text, reg(18), fill=INK, score=score, record=index < 3
        )
        if index == 0:
            r_buyer_name = region
        elif index == 1:
            r_buyer_tax = region
        text, score_or_cands = seller[index]
        cands = score_or_cands if isinstance(score_or_cands, list) else None
        score = score_or_cands if not isinstance(score_or_cands, list) else None
        region = page.text(
            (648, y), text, reg(18), fill=INK, candidates=cands, score=score, record=index < 3
        )
        if index == 0:
            r_seller_name = region
    page.line(left, 520, right, 520, fill=RULE, width=1)

    # 货物明细表
    cols = [
        (110, 380),
        (380, 470),
        (470, 530),
        (530, 600),
        (600, 710),
        (710, 880),
        (880, 960),
        (960, 1130),
    ]
    table_top, table_bottom = 528, 792
    page.line(left, 564, right, 564, fill=RULE, width=1)
    headers = [
        ("货物或应税劳务、服务名称", cols[0], "l"),
        ("规格型号", cols[1], "c"),
        ("单位", cols[2], "c"),
        ("数量", cols[3], "c"),
        ("单价", cols[4], "r"),
        ("金额", cols[5], "c"),
        ("税率", cols[6], "c"),
        ("税额", cols[7], "r"),
    ]
    page.table_row(
        [(text, col[0], col[1], align) for text, col, align in headers],
        table_top + 4,
        reg(17),
        fill=INK,
        category="table",
        score=0.95,
        record=False,
    )
    for index in range(1, len(cols)):
        x = cols[index][0]
        page.line(x, table_top, x, table_bottom, fill=RULE_LIGHT, width=1)

    item_rows = [
        ("技术开发服务*信息技术服务", "—", "项", "1", "86,000.00", "¥86,000.00", "13%", "¥11,180.00"),
        ("系统运维服务*信息技术服务", "—", "项", "1", "42,600.00", "¥42,600.00", "13%", "¥5,538.00"),
    ]
    aligns = ["l", "c", "c", "c", "r", "r", "c", "r"]
    for row_index, row in enumerate(item_rows):
        y = 582 + row_index * 78
        page.line(left, y + 60, right, y + 60, fill=RULE_LIGHT, width=1)
        page.table_row(
            [
                (value, cols[index][0], cols[index][1], aligns[index])
                for index, value in enumerate(row)
            ],
            y,
            reg(18),
            fill=INK,
            category="table",
            score=0.96 if row_index == 0 else 0.95,
        )

    # 合计行
    total_y = 738
    page.line(left, 736, right, 736, fill=RULE, width=1)
    page.text((118, total_y), "合　计", reg(18), fill=INK, score=0.96, record=False)
    r_amount = page.text(
        (866, total_y),
        "金额 ¥128,600.00",
        reg(17),
        fill=INK,
        anchor="ra",
        candidates=[
            candidate("金额 ¥128,600.00", "primary", 0.95),
            candidate("金额 ¥128,600.00", "reparse-16x", 0.93),
        ],
    )
    page.text((886, total_y), "税率 13%", reg(17), fill=INK, score=0.94, record=False)
    r_tax = page.text(
        (1122, total_y),
        "税额 ¥16,718.00",
        reg(17),
        fill=INK,
        anchor="ra",
        candidates=[
            candidate("税额 ¥16,718.00", "primary", 0.94),
            candidate("税额 ¥16,718.00", "reparse-16x", 0.92),
        ],
    )

    # 价税合计
    page.rect((left, 800, right, 940), outline=RULE_LIGHT, width=1)
    page.text(
        (128, 816),
        "价税合计（大写）壹拾肆万伍仟叁佰壹拾捌圆整",
        reg(24),
        fill=INK,
        score=0.92,
    )
    r_total = page.text_runs(
        128,
        866,
        [
            {"text": "价税合计（小写）", "font": reg(24), "fill": INK},
            {"text": "¥145,318.00", "font": reg(20), "fill": INK_FAINT},
        ],
        candidates=[
            candidate("价税合计（小写）¥145,318.00", "primary", 0.53),
            candidate("价税合计（小写）¥145,818.00", "reparse-16x", 0.19),
            candidate("价税合计（小写）¥145,318.00", "vlm-review", 0.96),
            candidate(
                "价税合计（小写）¥145,318.OO",
                "reparse-16x",
                None,
                verdict="not_scored",
                detail="Jev 评分服务超时，本轮未返回分数",
            ),
        ],
    )

    # 备注
    page.rect((left, 960, right, 1078), outline=RULE_LIGHT, width=1)
    page.text(
        (128, 976),
        "备注：本次服务费对应合同 HT-2026-1043，账期 30 天。",
        reg(19),
        fill=INK,
        candidates=[
            candidate("备注：本次服务费对应合同 HT-2026-1043，账期 30 天。", "primary", 0.95),
            candidate(
                "备注：本次服务费对应合同 HT-2026-1043，账期为 30 天。",
                "reparse-16x",
                None,
                verdict="failed",
                detail="Jev 渲染分块失败，未产出分数",
            ),
        ],
    )

    # 开票日期 / 经办人
    r_date = page.text(
        (128, 1112),
        "开票日期：2026年10月18日",
        reg(21),
        fill=INK,
        candidates=[candidate("开票日期：2026年10月18日", "primary", 0.98)],
    )
    for index, (label, value, score) in enumerate(
        [("收款人", "李　敏", 0.93), ("复核", "王　岚", 0.92), ("开票人", "陈　涛", 0.91)]
    ):
        page.text(
            (790, 1100 + index * 36),
            f"{label}：{value}",
            reg(19),
            fill=INK,
            score=score,
            record=False,
        )

    # 公章
    seal_box = draw_seal(
        page, 946, 1400, 126, "南京海川软件股份有限公司", center_text="发票专用章"
    )
    page.seal_region(
        seal_box,
        "南京海川软件股份有限公司 发票专用章",
        candidates=[
            candidate("南京海川软件股份有限公司 发票专用章", "primary", 0.93),
            candidate("南京海川软件股份有限公司 发票专用章", "reparse-16x", 0.91),
        ],
    )

    fields = [
        {"name": "发票号码", "value": "25117000000012345678", "region_ids": [r_number["id"]]},
        {"name": "发票代码", "value": "251170000000", "region_ids": [r_code["id"]]},
        {"name": "开票日期", "value": "2026年10月18日", "region_ids": [r_date["id"]]},
        {"name": "购买方名称", "value": "杭州明远网络科技有限公司", "region_ids": [r_buyer_name["id"]]},
        {"name": "销售方名称", "value": "南京海川软件股份有限公司", "region_ids": [r_seller_name["id"]]},
        {"name": "纳税人识别号", "value": "91330100MA2XXXXX1J", "region_ids": [r_buyer_tax["id"]]},
        {"name": "金额", "value": "128,600.00", "region_ids": [r_amount["id"]]},
        {"name": "税额", "value": "16,718.00", "region_ids": [r_tax["id"]]},
        {"name": "价税合计", "value": "145,318.00", "region_ids": [r_total["id"]]},
    ]
    return page, fields


# ---------------------------------------------------------------------------
# 二、技术服务合同
# ---------------------------------------------------------------------------
CONTRACT_MD = """# 技术开发与运维服务合同

合同编号：HT-2026-1043

甲方：杭州明远网络科技有限公司
乙方：南京海川信息技术有限公司

第一条 服务内容
乙方为甲方提供订单中心与结算系统的定制开发服务，包括需求调研、系统设计、
编码实现、联调测试与上线支持；并在验收合格后提供为期十二个月的运行维护
服务，含系统巡检、故障响应与版本升级。

第二条 服务期限与交付
服务期自 2026年11月01日 起至 2027年10月31日 止。乙方应于服务期起始后四十五
日内完成首期交付，并提交阶段验收报告。

第三条 合同金额
合同金额（大写）：伍拾捌万元整
合同金额：¥580,000.00

第四条 付款方式与期限
付款期限：合同签订后 30 日内支付合同金额的 40%，验收合格之日起 60 日内支付余款 60%。
甲方付款以乙方开具等额增值税专用发票为前提。

第五条 违约责任
任何一方违反本合同约定，应向守约方支付相当于合同金额 5% 的违约金；违约金
不足以弥补守约方损失的，还应赔偿差额。

第六条 其他约定
本合同一式肆份，甲乙双方各执贰份，具有同等法律效力。未尽事宜，由双方另行
协商并签订补充协议。

甲方（盖章）：杭州明远网络科技有限公司
乙方（盖章）：南京海川信息技术有限公司
签署日期：2026年10月21日"""


def build_contract() -> tuple[Page, list[dict]]:
    page = Page()
    left, right = 112, 1128

    page.text_spaced(
        620,
        92,
        "技术开发与运维服务合同",
        reg(38, bold=True),
        fill=INK,
        spacing=12,
        candidates=[
            candidate("技术开发与运维服务合同", "primary", 0.99),
            candidate("技术开发与运维服务合同", "reparse-16x", 0.96),
        ],
    )
    r_contract_no = page.text(
        (right, 174),
        "合同编号：HT-2026-1043",
        reg(20),
        fill=INK_SOFT,
        anchor="ra",
        candidates=[candidate("合同编号：HT-2026-1043", "primary", 0.98)],
    )
    page.line(left, 214, right, 214, fill=RULE, width=1)

    r_party_a = page.text(
        (left, 238),
        "甲方：杭州明远网络科技有限公司",
        reg(21),
        fill=INK,
        candidates=[candidate("甲方：杭州明远网络科技有限公司", "primary", 0.97)],
    )
    r_party_b = page.text(
        (left, 278),
        "乙方：南京海川信息技术有限公司",
        reg(21),
        fill=INK,
        candidates=[candidate("乙方：南京海川信息技术有限公司", "primary", 0.96)],
    )

    page.text_lines(
        left,
        330,
        [
            "鉴于甲方拟委托乙方承担其业务系统的定制开发与后续运维服务，双方本着平等自愿、",
            "诚实信用的原则，经友好协商订立本合同，以资共同信守。",
        ],
        reg(19),
        leading=1.85,
        fill=INK,
        score=0.94,
        record=False,
    )

    def clause(
        top,
        title,
        lines,
        title_score,
        body_score,
        body_candidates=None,
        record=True,
    ):
        page.text(
            (left, top), title, reg(21, bold=True), fill=INK, score=title_score, record=record
        )
        return page.text_lines(
            left,
            top + 42,
            lines,
            reg(19),
            leading=1.85,
            fill=INK,
            candidates=body_candidates,
            score=body_score,
            record=record,
        )

    clause(
        424,
        "第一条　服务内容",
        [
            "乙方为甲方提供订单中心与结算系统的定制开发服务，包括需求调研、系统设计、",
            "编码实现、联调测试与上线支持；并在验收合格后提供为期十二个月的运行维护",
            "服务，含系统巡检、故障响应与版本升级。",
        ],
        0.97,
        0.93,
    )
    clause(
        588,
        "第二条　服务期限与交付",
        [
            "服务期自 2026年11月01日 起至 2027年10月31日 止。乙方应于服务期起始后四十五",
            "日内完成首期交付，并提交阶段验收报告。",
        ],
        0.96,
        0.92,
    )
    clause(
        712,
        "第三条　合同金额",
        ["合同金额（大写）：伍拾捌万元整"],
        0.98,
        0.41,
        body_candidates=[
            candidate("合同金额（大写）：伍拾捌万圆整", "primary", 0.24),
            candidate("合同金额（大写）：伍拾捌万元整", "reparse-16x", 0.41),
            candidate(
                "合同金额（大写）：伍拾捌万圆整。",
                "reparse-16x",
                None,
                verdict="not_scored",
                detail="Jev 评分服务超时，本轮未返回分数",
            ),
        ],
    )
    r_amount = page.text(
        (left, 790),
        "合同金额：¥580,000.00",
        reg(20),
        fill=INK_FAINT,
        candidates=[
            candidate("合同金额：¥580,000.00", "primary", 0.38),
            candidate("合同金额：¥580,800.00", "reparse-16x", 0.19),
            candidate(
                "合同金额：¥580,000.OO",
                "reparse-16x",
                None,
                verdict="not_scored",
                detail="Jev 评分服务超时，本轮未返回分数",
            ),
        ],
    )
    r_payment = clause(
        852,
        "第四条　付款方式与期限",
        [
            "付款期限：合同签订后 30 日内支付合同金额的 40%，验收合格之日起 60 日内支付余款 60%。",
            "甲方付款以乙方开具等额增值税专用发票为前提。",
        ],
        0.96,
        0.93,
    )
    # 签署区
    page.line(left, 1090, right, 1090, fill=RULE, width=1)
    page.text(
        (left, 1110),
        "甲方（盖章）：杭州明远网络科技有限公司",
        reg(20),
        fill=INK,
        candidates=[candidate("甲方（盖章）：杭州明远网络科技有限公司", "primary", 0.95)],
    )
    page.text(
        (672, 1110),
        "乙方（盖章）：南京海川信息技术有限公司",
        reg(20),
        fill=INK,
        candidates=[candidate("乙方（盖章）：南京海川信息技术有限公司", "primary", 0.95)],
    )
    page.text((left, 1156), "授权代表（签字）：", reg(19), fill=INK_SOFT, score=0.93, record=False)
    page.text((672, 1156), "授权代表（签字）：", reg(19), fill=INK_SOFT, score=0.93, record=False)
    page.line(300, 1182, 560, 1182, fill=RULE_LIGHT, width=1)
    page.line(860, 1182, 1120, 1182, fill=RULE_LIGHT, width=1)
    r_sign_date = page.text(
        (left, 1204),
        "签署日期：2026年10月21日",
        reg(20),
        fill=INK,
        candidates=[candidate("签署日期：2026年10月21日", "primary", 0.97)],
    )
    page.text(
        (672, 1204),
        "签署日期：2026年10月21日",
        reg(20),
        fill=INK,
        candidates=[candidate("签署日期：2026年10月21日", "primary", 0.97)],
    )

    seal_a = draw_seal(
        page, 352, 1348, 96, "杭州明远网络科技有限公司", center_text="合同专用章"
    )
    page.seal_region(
        seal_a,
        "杭州明远网络科技有限公司 合同专用章",
        candidates=[
            candidate("杭州明远网络科技有限公司 合同专用章", "primary", 0.93),
            candidate("杭州明远网络科技有限公司 合同专用章", "reparse-16x", 0.92),
        ],
    )
    seal_b = draw_seal(
        page, 912, 1348, 96, "南京海川信息技术有限公司", center_text="合同专用章"
    )
    page.seal_region(
        seal_b,
        "南京海川信息技术有限公司 合同专用章",
        candidates=[
            candidate("南京海川信息技术有限公司 合同专用章", "primary", 0.92),
            candidate("南京海川信息技术有限公司 合同专用章", "reparse-16x", 0.90),
        ],
    )

    page.line(left, 1520, right, 1520, fill=RULE_LIGHT, width=1)
    page.text(
        (620, 1538),
        "第 1 页　共 2 页",
        reg(17),
        fill=INK_SOFT,
        anchor="ma",
        score=0.93,
        record=False,
    )

    fields = [
        {"name": "合同编号", "value": "HT-2026-1043", "region_ids": [r_contract_no["id"]]},
        {"name": "甲方", "value": "杭州明远网络科技有限公司", "region_ids": [r_party_a["id"]]},
        {"name": "乙方", "value": "南京海川信息技术有限公司", "region_ids": [r_party_b["id"]]},
        {"name": "合同金额", "value": "580,000.00", "region_ids": [r_amount["id"]]},
        {
            "name": "付款期限",
            "value": "合同签订后 30 日内支付合同金额的 40%，验收合格之日起 60 日内支付余款 60%。",
            "region_ids": [r_payment["id"]],
        },
        {"name": "签署日期", "value": "2026年10月21日", "region_ids": [r_sign_date["id"]]},
    ]
    return page, fields


# ---------------------------------------------------------------------------
# 三、员工报销单
# ---------------------------------------------------------------------------
EXPENSE_MD = """# 费用报销单

单据号：BX-20261025-007
报销人：李思远
所属部门：平台研发部
日期：2026年10月25日

费用类型：差旅费、办公用品、业务招待费
差旅费　杭州至南京项目现场实施差旅　1,480.00
办公用品　研发团队耗材采购　386.50
业务招待费　客户现场沟通餐费　1,098.00
合计金额：¥2,864.50

审批意见：同意报销，请补充差旅报销明细。"""


def build_expense() -> tuple[Page, list[dict]]:
    page = Page()
    left, right = 112, 1128

    page.text_spaced(
        620,
        180,
        "费用报销单",
        reg(40, bold=True),
        fill=INK,
        spacing=16,
        candidates=[
            candidate("费用报销单", "primary", 0.98),
            candidate("费用报销单", "reparse-16x", 0.95),
        ],
    )
    page.line(360, 256, 880, 256, fill=RULE, width=1)

    r_no = page.text(
        (left, 300),
        "单据号：BX-20261025-007",
        reg(21),
        fill=INK,
        candidates=[candidate("单据号：BX-20261025-007", "primary", 0.97)],
    )
    r_dept = page.text(
        (600, 300),
        "所属部门：平台研发部",
        reg(21),
        fill=INK,
        candidates=[candidate("所属部门：平台研发部", "primary", 0.95)],
    )
    r_person = page.text_runs(
        left,
        352,
        [
            {"text": "报销人：", "font": reg(21), "fill": INK},
            {"text": "李思远", "font": font(FONT_HAND, 30), "fill": SIGN_INK, "hand": 21},
        ],
        candidates=[
            candidate("报销人：李思远", "primary", 0.52),
            candidate("报销人：李思源", "reparse-16x", 0.27),
            candidate(
                "报销人：李思沅",
                "reparse-16x",
                None,
                verdict="not_scored",
                detail="Jev 评分服务超时，本轮未返回分数",
            ),
        ],
    )
    r_date = page.text(
        (600, 352),
        "日期：2026年10月25日",
        reg(21),
        fill=INK,
        candidates=[candidate("日期：2026年10月25日", "primary", 0.98)],
    )
    r_types = page.text(
        (left, 420),
        "费用类型：差旅费、办公用品、业务招待费",
        reg(21),
        fill=INK,
        candidates=[candidate("费用类型：差旅费、办公用品、业务招待费", "primary", 0.93)],
    )

    # 明细表
    table_top = 486
    page.line(left, table_top, right, table_top, fill=RULE, width=2)
    page.text((132, table_top + 20), "费用类型", reg(19, bold=True), fill=INK, category="table", score=0.94)
    page.table_row(
        [
            ("说明", 380, 900, "l"),
            ("金额", 900, 1128, "r"),
        ],
        table_top + 20,
        reg(19, bold=True),
        fill=INK,
        category="table",
        score=0.94,
    )
    page.line(left, table_top + 70, right, table_top + 70, fill=RULE, width=1)
    for x in (360, 900):
        page.line(x, table_top, x, table_top + 348, fill=RULE_LIGHT, width=1)

    rows = [
        ("差旅费", "杭州至南京项目现场实施差旅", "1,480.00", 0.92),
        ("办公用品", "研发团队耗材采购", "386.50", 0.91),
        ("业务招待费", "客户现场沟通餐费", "1,098.00", 0.90),
    ]
    for index, (kind, note, amount, score) in enumerate(rows):
        y = table_top + 94 + index * 82
        page.table_row(
            [
                (kind, 120, 360, "l"),
                (note, 380, 900, "l"),
                (amount, 900, 1128, "r"),
            ],
            y,
            reg(19),
            fill=INK,
            category="table",
            score=score,
        )
        page.line(left, y + 60, right, y + 60, fill=RULE_LIGHT, width=1)

    r_sum = page.text_runs(
        left,
        table_top + 362,
        [
            {"text": "合计金额：", "font": reg(21), "fill": INK},
            {"text": "¥2,864.50", "font": reg(19), "fill": INK_FAINT},
        ],
        candidates=[
            candidate("合计金额：¥2,864.50", "primary", 0.37),
            candidate("合计金额：¥2,964.50", "reparse-16x", 0.19),
            candidate(
                "合计金额：¥2,864.5O",
                "reparse-16x",
                None,
                verdict="not_scored",
                detail="Jev 评分服务超时，本轮未返回分数",
            ),
        ],
    )
    page.line(left, table_top + 350, right, table_top + 350, fill=RULE, width=2)

    # 审批区
    page.line(left, 926, right, 926, fill=RULE, width=1)
    page.text_runs(
        left,
        954,
        [
            {"text": "审批意见：", "font": reg(20), "fill": INK},
            {
                "text": "同意报销，请补充差旅报销明细。",
                "font": font(FONT_HAND, 27),
                "fill": SIGN_INK,
                "hand": 33,
            },
        ],
        candidates=[
            candidate("审批意见：同意报销，请补充差旅报销明细。", "primary", 0.94),
            candidate("审批意见：同意报销，请补充差旅报销明细。", "reparse-16x", 0.92),
        ],
    )
    page.text_runs(
        left,
        1074,
        [
            {"text": "李思远", "font": font(FONT_HAND, 34), "fill": SIGN_INK, "hand": 41},
        ],
        candidates=[candidate("李思远", "primary", 0.92)],
    )
    page.text((left, 1128), "报销人签字", reg(17), fill=INK_SOFT, score=0.9, record=False)

    stamp_box = draw_rect_stamp(page, 820, 1090, 300, 116, "同意报销", angle=-8.0)
    page.seal_region(
        stamp_box,
        "同意报销",
        candidates=[candidate("同意报销", "primary", 0.93)],
    )

    # 凭证粘贴区
    paste_box = (112, 1196, 1128, 1470)
    for x in range(paste_box[0], paste_box[2], 18):
        page.line(x, paste_box[1], min(x + 9, paste_box[2]), paste_box[1], fill=RULE_LIGHT, width=1)
        page.line(x, paste_box[3], min(x + 9, paste_box[2]), paste_box[3], fill=RULE_LIGHT, width=1)
    for y in range(paste_box[1], paste_box[3], 18):
        page.line(paste_box[0], y, paste_box[0], min(y + 9, paste_box[3]), fill=RULE_LIGHT, width=1)
        page.line(paste_box[2], y, paste_box[2], min(y + 9, paste_box[3]), fill=RULE_LIGHT, width=1)
    page._record(
        paste_box,
        "费用凭证粘贴区",
        "figure",
        [candidate("费用凭证粘贴区", "primary", 0.92)],
        0.92,
        0,
    )
    page.draw.text(
        (132, 1212),
        "费用凭证粘贴区",
        font=reg(17),
        fill=INK_FAINT,
        anchor="la",
    )

    # 底部签批
    page.line(left, 1512, right, 1512, fill=RULE, width=1)
    page.table_row(
        [
            ("部门负责人：周　立", 112, 456, "l"),
            ("财务审核：陈　涛", 456, 800, "l"),
            ("出纳：孙　敏", 800, 1128, "l"),
        ],
        1540,
        reg(19),
        fill=INK,
        score=0.92,
    )
    page.text(
        (left, 1610),
        "本单据由员工手工填写，经部门负责人审批后作为财务凭证。",
        reg(17),
        fill=INK_FAINT,
        score=0.91,
        record=False,
    )

    fields = [
        {"name": "单据号", "value": "BX-20261025-007", "region_ids": [r_no["id"]]},
        {"name": "报销人", "value": "李思远", "region_ids": [r_person["id"]]},
        {"name": "部门", "value": "平台研发部", "region_ids": [r_dept["id"]]},
        {"name": "费用类型", "value": "差旅费、办公用品、业务招待费", "region_ids": [r_types["id"]]},
        {"name": "金额", "value": "2,864.50", "region_ids": [r_sum["id"]]},
        {"name": "日期", "value": "2026年10月25日", "region_ids": [r_date["id"]]},
    ]
    return page, fields


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def strip_pixels(regions: list[dict]) -> list[dict]:
    return [{key: value for key, value in region.items() if key != "_pixels"} for region in regions]


def add_digests(session: dict) -> dict:
    """给每份文档记上源文件的哈希。

    上传一张图进来时按内容匹配已录制的样例，靠的就是这个哈希。
    """

    import hashlib

    for document in session["documents"]:
        source = ROOT / document["source"]
        if source.exists():
            document["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return session


def main() -> None:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    invoice, invoice_fields = build_invoice()
    contract, contract_fields = build_contract()
    expense, expense_fields = build_expense()

    # 报销单是手机翻拍：轻微旋转 + 噪点 + 亮度不均
    apply_photo_effects(expense, angle_deg=0.9, noise=3.2, seed=20261025)

    invoice.img.save(SAMPLES_DIR / "invoice_001.png")
    contract.img.save(SAMPLES_DIR / "contract_001.png")
    expense.img.save(SAMPLES_DIR / "expense_001.png")

    session = {
        "documents": [
            {
                "doc_id": "invoice-001",
                "source": "samples/invoice_001.png",
                "label": "增值税专用发票",
                "schema": "invoice",
                "description": (
                    "价税合计的小写金额印得偏浅又偏小，识别结果拿不准；"
                    "合计行的金额与税额两次识别给出不同的数字，需要人工确认。"
                ),
                "markdown": INVOICE_MD,
                "regions": strip_pixels(invoice.regions),
                "fields": invoice_fields,
            },
            {
                "doc_id": "contract-001",
                "source": "samples/contract_001.png",
                "label": "技术服务合同",
                "schema": "contract",
                "description": (
                    "合同金额的大小写两处都印得偏淡，识别结果互相打架；"
                    "乙方名称与发票上的销售方名称对不上，属于跨文档矛盾。"
                ),
                "markdown": CONTRACT_MD,
                "regions": strip_pixels(contract.regions),
                "fields": contract_fields,
            },
            {
                "doc_id": "expense-001",
                "source": "samples/expense_001.png",
                "label": "员工报销单",
                "schema": "expense",
                "description": (
                    "手机翻拍，纸面有噪点与光照不均；手写签名和审批意见识别不稳；"
                    "单据上的合计金额与三条明细的实际之和差了一百元。"
                ),
                "markdown": EXPENSE_MD,
                "regions": strip_pixels(expense.regions),
                "fields": expense_fields,
            },
        ]
    }
    # Jev 分数用录制的真实分数，不是手写的
    recorded = load_scores(SAMPLES_DIR / "jev_scores.json")
    applied = 0
    for document in session["documents"]:
        doc_scores = recorded.get(document["doc_id"], {})
        for region in document["regions"]:
            region.setdefault("truth", region["text"])
            applied += apply_scores(region.get("candidates", []), doc_scores.get(region["id"]))

    add_digests(session)
    (SESSIONS_DIR / "demo.json").write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for name, page in (
        ("invoice_001.png", invoice),
        ("contract_001.png", contract),
        ("expense_001.png", expense),
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
