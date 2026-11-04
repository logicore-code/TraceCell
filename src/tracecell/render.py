"""把候选文本渲染成图像。

Jev 判断"这段识别结果和原图对不对得上"，光看原图和文字不够——公式和表格
的对错往往在结构上，文字对但排版不对的情况也得判出来。所以官方流程会把
候选文本再渲染回一张图，和原图一起送进模型比对。

渲染参数照着官方 `render_label.py` 的纯文本路径抄：字号 18px、行高 2.0、
字体栈一致、容器按内容收缩、白边裁掉 16px、宽高比超过 8 就收窄换行。
抄这些不是为了让图好看，是为了让**分数跟官方可比**——渲染环境和官方差得
远，绝对分数就会整体偏移。

官方渲染器还支持 LaTeX 公式（KaTeX）和 HTML 表格。这两样这里没有实现：
样例单据的识别结果是纯文本，公式和表格走不通会退化成和公式长得很像的
字符串。遇到公式或表格的区域，Jev 的分数不可信，别拿来当判据。

有 Playwright 就用 Chromium 渲染（和官方一致，字体、排版最接近）；没有就退回
PIL，用本机字体画。两条路出来的图不一样，**评分会跟着变**，所以部署时最好
固定其中一条。
"""

from __future__ import annotations

import html
import io
import re
import threading
from typing import Any

# 官方渲染器的参数（scripts/minicpm-ocr-jev/renderer/render_label.py）
FONT_SIZE = 18
LINE_HEIGHT = 2.0
FONT_FAMILY = (
    '-apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans CJK SC", '
    '"Source Han Sans SC", "WenQuanYi Micro Hei", "Microsoft YaHei", SimSun, sans-serif'
)
TRIM_PADDING = 16
MAX_ASPECT_RATIO = 8
MIN_TEXT_WIDTH = 480
MAX_TEXT_WIDTH = 1200

_BROWSER_LOCK = threading.Lock()
# Playwright 的同步接口不支持多线程共用一个浏览器。渲染本身很快（一张图
# 几十毫秒），串行化不影响速度，但少了它并发调用会直接把浏览器搞坏。
_RENDER_LOCK = threading.Lock()
_BROWSER: Any = None
_BROWSER_FAILED = False


def render_text(text: str, width: int | None = None) -> bytes:
    """把一段文本渲染成 PNG。

    ``width`` 是原图块的宽度，只作参考——官方对普通文本不强行拉伸到原图宽度，
    而是保持自然宽度，只在宽高比过大时收窄换行。
    """

    global _BROWSER_FAILED

    if not _BROWSER_FAILED:
        try:
            return _render_with_chromium(text)
        except Exception:  # noqa: BLE001 - 没有浏览器就退回 PIL
            _BROWSER_FAILED = True
    return _render_with_pil(text)


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


def _to_html(text: str) -> str:
    """纯文本转 HTML。空行分段，其余按行断开——和官方 nl2br 的效果一致。"""

    escaped = html.escape(text or "")
    paragraphs = re.split(r"\n\s*\n", escaped)
    blocks = []
    for para in paragraphs:
        if not para.strip():
            continue
        blocks.append("<p>" + para.strip("\n").replace("\n", "<br>") + "</p>")
    return "".join(blocks) or "<p></p>"


def _page_html(content: str) -> str:
    return (
        '<!DOCTYPE html><html><head><meta charset="UTF-8"><style>'
        f"body {{ font-family: {FONT_FAMILY}; padding: 0; margin: 0; background: #fff;"
        f"font-size: {FONT_SIZE}px; line-height: {LINE_HEIGHT}; }}"
        "p { margin: 0.2em 0; }"
        "#capture-area { display: inline-block; }"
        "</style></head><body>"
        f'<div id="capture-area">{content}</div>'
        "</body></html>"
    )


def _render_with_chromium(text: str) -> bytes:
    with _RENDER_LOCK:
        browser = _get_browser()
        page = browser.new_page(viewport={"width": 1400, "height": 900})
        try:
            page.set_content(_page_html(_to_html(text)), wait_until="load")
            page.evaluate(_FIT_SCRIPT)
            element = page.query_selector("#capture-area")
            raw = element.screenshot(type="png")
        finally:
            page.close()
    return _trim_white_border(raw)


# 官方 TABLE_LAYOUT_SCRIPT 里纯文本那一支：宽高比过大就收窄换行，二分找宽度。
_FIT_SCRIPT = """
() => {
  const area = document.getElementById('capture-area');
  if (!area) return;
  const measure = () => {
    const rect = area.getBoundingClientRect();
    const width = Math.max(1, rect.width || area.scrollWidth || 0);
    const height = Math.max(1, rect.height || area.scrollHeight || 0);
    return {width, height, ratio: width / height};
  };
  const apply = (width) => {
    const px = `${Math.round(width)}px`;
    area.style.width = px;
    area.style.maxWidth = px;
    area.style.whiteSpace = 'normal';
    area.style.wordBreak = 'break-word';
    area.style.overflowWrap = 'anywhere';
    area.style.boxSizing = 'border-box';
  };
  const natural = measure();
  if (natural.ratio <= 8 || natural.width <= 480) return;
  let high = Math.min(1200, natural.width);
  apply(high);
  if (measure().ratio <= 8) return;
  let low = 480;
  apply(low);
  if (measure().ratio > 8) return;
  let best = low;
  for (let i = 0; i < 12; i += 1) {
    const mid = (low + high) / 2;
    apply(mid);
    if (measure().ratio <= 8) { best = mid; low = mid; } else { high = mid; }
  }
  apply(best);
}
"""


def _trim_white_border(raw: bytes) -> bytes:
    """裁掉渲染图外面的白边，留 16px 余量。"""

    from PIL import Image, ImageChops

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    background = Image.new(image.mode, image.size, "white")
    bbox = ImageChops.difference(image, background).getbbox()
    if not bbox:
        return raw
    left = max(bbox[0] - TRIM_PADDING, 0)
    top = max(bbox[1] - TRIM_PADDING, 0)
    right = min(bbox[2] + TRIM_PADDING, image.width)
    bottom = min(bbox[3] + TRIM_PADDING, image.height)
    cropped = image.crop((left, top, right, bottom))
    buffer = io.BytesIO()
    cropped.save(buffer, format="PNG")
    return buffer.getvalue()


def _render_with_pil(text: str) -> bytes:
    """PIL 渲染。字体从本机找。"""

    from PIL import Image, ImageDraw

    font = _load_font(FONT_SIZE)
    lines: list[str] = []
    for raw_line in (text or "").split("\n"):
        current = ""
        for char in raw_line:
            candidate = current + char
            if font.getlength(candidate) > MAX_TEXT_WIDTH - TRIM_PADDING * 2:
                lines.append(current)
                current = char
            else:
                current = candidate
        lines.append(current)

    line_height = int(FONT_SIZE * LINE_HEIGHT)
    height = max(line_height, line_height * len(lines)) + TRIM_PADDING * 2
    image = Image.new("RGB", (MAX_TEXT_WIDTH, height), "white")
    draw = ImageDraw.Draw(image)
    y = TRIM_PADDING
    for line in lines:
        draw.text((TRIM_PADDING, y), line, font=font, fill="black")
        y += line_height
    return _trim_white_border(_to_png(image))


def _to_png(image) -> bytes:
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
