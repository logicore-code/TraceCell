"""把候选文本渲染成图像。

Jev 判断"这段识别结果和原图对不对得上"，光看原图和文字不够——公式和表格
的对错往往在结构上，文字对但排版不对的情况也得判出来。所以官方流程会把
候选文本再渲染回一张图，和原图一起送进模型比对。

这个模块提供渲染能力。有 Playwright 就用 Chromium 渲染（和官方一致，
字体、排版最接近）；没有就退回 PIL，用本机字体画。两条路出来的图不一样，
**评分会跟着变**，所以部署时最好固定其中一条。
"""

from __future__ import annotations

import html
import io
import os
import threading
from typing import Any

# 官方 SDK 固定的字号与边距。改动这些会让分数和官方对不上。
FONT_SIZE = 22
LINE_HEIGHT = 1.5
PADDING = 24
BACKGROUND = "#ffffff"
FOREGROUND = "#000000"

_BROWSER_LOCK = threading.Lock()
_BROWSER: Any = None
_BROWSER_FAILED = False


def render_text(
    text: str,
    width: int = 1024,
    font_family: str = "Noto Sans CJK SC, Noto Sans SC, Source Han Sans SC, sans-serif",
) -> bytes:
    """把一段文本渲染成 PNG。"""

    global _BROWSER_FAILED

    if not _BROWSER_FAILED:
        try:
            return _render_with_chromium(text, width, font_family)
        except Exception:  # noqa: BLE001 - 没有浏览器就退回 PIL
            _BROWSER_FAILED = True
    return _render_with_pil(text, width)


def close_browser() -> None:
    """关掉复用的浏览器。进程退出前调一下。"""

    global _BROWSER
    with _BROWSER_LOCK:
        if _BROWSER is not None:
            try:
                _BROWSER.close()
            except Exception:  # noqa: BLE001
                pass
            _BROWSER = None


def _get_browser():
    """起一个浏览器并复用。

    每个候选都新开一个 Chromium 的话，渲染这一项就能吃掉大部分时间——
    浏览器启动比渲染本身还贵。
    """

    global _BROWSER
    with _BROWSER_LOCK:
        if _BROWSER is None:
            from playwright.sync_api import sync_playwright

            _BROWSER = sync_playwright().start().chromium.launch(args=["--no-sandbox"])
        return _BROWSER


def _render_with_chromium(text: str, width: int, font_family: str) -> bytes:
    document = (
        "<!doctype html><html><head><meta charset='utf-8'><style>"
        f"html,body{{margin:0;padding:0;background:{BACKGROUND};color:{FOREGROUND};}}"
        f"body{{padding:{PADDING}px;font-family:'{font_family}';"
        f"font-size:{FONT_SIZE}px;line-height:{LINE_HEIGHT};"
        "white-space:pre-wrap;word-break:break-all;}"
        "</style></head><body>"
        f"<div>{html.escape(text)}</div>"
        "</body></html>"
    )
    browser = _get_browser()
    page = browser.new_page(viewport={"width": width, "height": 200})
    try:
        page.set_content(document, wait_until="load")
        element = page.query_selector("body")
        return element.screenshot(type="png")
    finally:
        page.close()


def _render_with_pil(text: str, width: int) -> bytes:
    """PIL 渲染。字体从本机找。"""

    from PIL import Image, ImageDraw, ImageFont

    font = _load_font(FONT_SIZE)
    lines: list[str] = []
    for raw_line in text.split("\n"):
        if not raw_line:
            lines.append("")
            continue
        current = ""
        for char in raw_line:
            candidate = current + char
            if font.getlength(candidate) > width - PADDING * 2:
                lines.append(current)
                current = char
            else:
                current = candidate
        lines.append(current)

    line_height = int(FONT_SIZE * LINE_HEIGHT)
    height = max(line_height, line_height * len(lines)) + PADDING * 2
    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    y = PADDING
    for line in lines:
        draw.text((PADDING, y), line, font=font, fill=FOREGROUND)
        y += line_height
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _load_font(size: int):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()
