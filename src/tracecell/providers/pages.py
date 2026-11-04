"""文档输入。

真实业务拿过来的东西有三样：单页扫描图、多页 PDF、已经是图片的凭证。
这一层把它们统一成"一页一张 PNG"，后面的流程就只认页，不关心来源。

PDF 渲染用 pypdfium2（PDFium，BSD-3 许可）。没用 PyMuPDF 是因为它是
AGPL-3.0，混进 Apache-2.0 的项目里会有许可问题。
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 图片魔数，用来在不看扩展名的情况下判断来源
_IMAGE_MAGIC = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF8", b"BM", b"RIFF", b"II*\x00", b"MM\x00*")


@dataclass
class Page:
    """一页图像。"""

    index: int
    png: bytes
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "width": self.width, "height": self.height, "bytes": len(self.png)}


@dataclass
class Document:
    """一份待核验的文档。"""

    doc_id: str
    source: str
    pages: list[Page] = field(default_factory=list)
    kind: str = "image"          # image / pdf
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def page(self, index: int) -> Page | None:
        for page in self.pages:
            if page.index == index:
                return page
        return None


def is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def is_image(data: bytes) -> bool:
    return data[:4] == b"%PDF-" == False and data.startswith(_IMAGE_MAGIC)


def load_image(data: bytes) -> tuple[bytes, int, int]:
    """把任意图片格式转成 PNG，返回 (png, width, height)。"""

    from PIL import Image

    with Image.open(io.BytesIO(data)) as source:
        converted = source.convert("RGB")
        buffer = io.BytesIO()
        converted.save(buffer, format="PNG")
        return buffer.getvalue(), converted.width, converted.height


def load_document(
    source: str | Path | bytes,
    doc_id: str | None = None,
    dpi: int = 150,
    max_pages: int = 50,
) -> Document:
    """加载文档，自动判断是图片还是 PDF。

    dpi 150 是文档扫描的常规档位：分辨率够看清小字，页图又不会大到
    拖慢视觉编码器。
    """

    if isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
        name = path.name
    else:
        data = source
        name = doc_id or "upload"

    document = Document(doc_id=doc_id or Path(name).stem or "document", source=name)

    if is_pdf(data):
        document.kind = "pdf"
        document.pages = _render_pdf(data, dpi, max_pages)
        document.meta["dpi"] = dpi
    elif not data:
        # 回放模式下没有原图。给一张空白页，流程照样能跑完。
        png, width, height = load_image(_blank_png())
        document.pages = [Page(index=0, png=png, width=width, height=height)]
    else:
        document.kind = "image"
        png, width, height = load_image(data)
        document.pages = [Page(index=0, png=png, width=width, height=height)]

    if not document.pages:
        raise ValueError(f"没能从 {name} 里读出任何页面")
    return document


def _blank_png() -> bytes:
    """一张 1×1 的白图。"""

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), (255, 255, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


def _render_pdf(data: bytes, dpi: int, max_pages: int) -> list[Page]:
    """PDF 逐页渲染成 PNG。"""

    import pypdfium2

    scale = dpi / 72.0
    pages: list[Page] = []
    pdf = pypdfium2.PdfDocument(data)
    try:
        count = min(len(pdf), max_pages)
        for index in range(count):
            page = pdf[index]
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil().convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            pages.append(
                Page(index=index, png=buffer.getvalue(), width=image.width, height=image.height)
            )
            page.close()
    finally:
        pdf.close()
    return pages


def as_document(source: str | Path | bytes | Document, doc_id: str | None = None, dpi: int = 150) -> Document:
    """把各种形态的输入统一成 Document。"""

    if isinstance(source, Document):
        return source
    return load_document(source, doc_id=doc_id, dpi=dpi)
