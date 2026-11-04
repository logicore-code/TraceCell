"""轻量版面切分。

TraceCell 需要知道"一块文字在原图的哪里"，否则证据坐标无从谈起。
官方流水线用 PP-DocLayoutV3 做这件事，但那是一个额外的 ONNX 运行时和
一份模型文件。TraceCell 自带一个纯 numpy 的实现作为默认，理由是：

* 依赖为零，pip 装完就能跑；
* 切分结果足够支撑"把这一块裁出来给 Jev 评分"和"在界面上高亮原图"；
* 真要更高精度时，可以整个换掉——接口是 `detect`。

算法走的是经典的投影法：二值化 → 水平投影找文本行 → 按行间距合并成块
→ 块内再做一次垂直投影判断是不是表格。
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..types import BBox, Category

# 判定文本行的最小墨迹比例（占整行宽度）
INK_RATIO_MIN = 0.008
# 相邻文本行之间超过这个高度（相对图像高度）就认为是两个块
LINE_GAP_RATIO = 0.006
# 判定表格分隔线的长度比例
TABLE_RULE_RATIO = 0.45


@dataclass
class LayoutBlock:
    """一个版面块。"""

    bbox: BBox
    category: Category = Category.TEXT
    ink_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": {
                "x0": self.bbox.x0,
                "y0": self.bbox.y0,
                "x1": self.bbox.x1,
                "y1": self.bbox.y1,
            },
            "category": self.category.value,
            "ink_ratio": round(self.ink_ratio, 5),
        }


def _decode(image: bytes):
    from PIL import Image

    return Image.open(io.BytesIO(image)).convert("L")


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu 二值化，返回 True 表示有墨。"""

    flat = gray.reshape(-1)
    histogram = np.bincount(flat, minlength=256).astype(np.float64)
    total = flat.size
    sum_all = float(np.dot(np.arange(256), histogram))
    sum_background = 0.0
    weight_background = 0.0
    best_threshold = 128
    best_variance = -1.0
    for threshold in range(256):
        weight_background += histogram[threshold]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break
        sum_background += threshold * histogram[threshold]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return gray < best_threshold


def _row_runs(ink: np.ndarray, min_ratio: float) -> list[tuple[int, int]]:
    """找出有墨的行段。"""

    height, width = ink.shape
    projection = ink.sum(axis=1)
    threshold = max(1.0, width * min_ratio)
    active = projection > threshold
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(active):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, height))
    return runs


def detect(image: bytes, padding: float = 0.004) -> list[LayoutBlock]:
    """切分版面。

    返回按阅读顺序（从上到下）排列的块。
    """

    gray = np.asarray(_decode(image), dtype=np.uint8)
    height, width = gray.shape
    ink = binarize(gray)

    lines = _row_runs(ink, INK_RATIO_MIN)
    if not lines:
        return []

    gap_limit = max(2, int(height * LINE_GAP_RATIO))
    blocks: list[tuple[int, int]] = []
    block_start = lines[0][0]
    block_end = lines[0][1]
    for index in range(1, len(lines)):
        gap = lines[index][0] - lines[index - 1][1]
        if gap > gap_limit:
            blocks.append((block_start, block_end))
            block_start, block_end = lines[index]
        else:
            block_end = lines[index][1]
    blocks.append((block_start, block_end))

    pad_y = max(1, int(height * padding))
    pad_x = max(1, int(width * padding))

    result: list[LayoutBlock] = []
    for top, bottom in blocks:
        strip = ink[top:bottom]
        column_projection = strip.sum(axis=0)
        left = int(np.argmax(column_projection > 0))
        right = int(len(column_projection) - np.argmax(column_projection[::-1] > 0))
        if right <= left:
            continue
        block = strip[:, left:right]
        ratio = float(block.sum()) / max(1, block.size)

        category = _classify(block)
        bbox = BBox.from_pixels(
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(width, right + pad_x),
            min(height, bottom + pad_y),
            width,
            height,
        )
        result.append(LayoutBlock(bbox=bbox, category=category, ink_ratio=ratio))
    return result


def _classify(block: np.ndarray) -> Category:
    """按块内特征粗分类别。"""

    height, width = block.shape
    if width == 0 or height == 0:
        return Category.TEXT

    # 表格：存在横贯大部分宽度的实心行
    row_fill = block.mean(axis=1)
    long_rules = int(np.sum(row_fill > 0.5))
    if long_rules >= 2 and height > 6:
        return Category.TABLE

    # 公式：数学符号密度高
    ink = block.sum()
    if ink > 0:
        # 粗略估计：公式块的墨迹分布更"稀"但行高更低
        density = ink / block.size
        if density < 0.12 and height < 0.06 * 1000:
            return Category.FORMULA
    return Category.TEXT


@lru_cache(maxsize=4)
def _decoded(image: bytes):
    """解码图像并缓存。

    一页文档要裁几十个区域，每次都重新解码整张 PNG 会把时间全花在
    解码上——1240×1754 的发票解一次要 0.2 秒，40 个区域就是 8 秒。
    缓存之后只解一次。
    """

    from PIL import Image

    try:
        return Image.open(io.BytesIO(image)).convert("RGB")
    except Exception:  # noqa: BLE001 - 任何解码失败都退化成空白图
        return None


# PP-DocLayoutV3 的标签到 TraceCell 版面类别的映射
PADDLE_LABEL_TO_CATEGORY = {
    "table": Category.TABLE,
    "display_formula": Category.FORMULA,
    "inline_formula": Category.FORMULA,
    "algorithm": Category.FORMULA,
    "seal": Category.SEAL,
    "chart": Category.FIGURE,
    "image": Category.FIGURE,
    "figure_title": Category.FIGURE,
    "footer_image": Category.FIGURE,
    "header_image": Category.FIGURE,
}


class PaddleLayoutDetector:
    """官方 PP-DocLayoutV3 版面检测（ONNX 版）。

    自带的投影法切分器不依赖任何模型，缺点是复杂表格和多栏排版会偏。
    真要处理扫描件，这个检测器更靠谱——它是 MiniCPM-OCR 官方流水线里
    用的那一个。

    模型文件用 ``scripts/fetch_layout_model.py`` 下载（约 136MB），然后：

        export TRACECELL_LAYOUT_MODEL=/path/to/PP-DocLayoutV3

    onnxruntime 不在默认依赖里，装了才会用：

        pip install onnxruntime
    """

    INPUT_SIZE = 800

    def __init__(self, model_dir: str, threshold: float = 0.4) -> None:
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        self.threshold = threshold
        self.session = ort.InferenceSession(
            str(self.model_dir / "PP-DocLayoutV3.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self.labels = self._load_labels()

    def _load_labels(self) -> list[str]:
        """标签表。优先用 config.json 的 id2label，没有就退回 yml。"""

        config_path = self.model_dir / "config.json"
        if config_path.exists():
            data = json.loads(config_path.read_text(encoding="utf-8"))
            id2label = data.get("id2label") or {}
            if id2label:
                size = max(int(k) for k in id2label) + 1
                labels = [""] * size
                for key, value in id2label.items():
                    labels[int(key)] = value
                return labels
        yml_path = self.model_dir / "inference.yml"
        if yml_path.exists():
            for line in yml_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("label_list:"):
                    continue
                if line.startswith("- ") and not line.startswith("- -"):
                    pass
            labels = []
            inside = False
            for line in yml_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("label_list:"):
                    inside = True
                    continue
                if inside:
                    if line.startswith("- "):
                        labels.append(line[2:].strip())
                    elif line.strip():
                        break
            if labels:
                return labels
        raise ValueError(f"{self.model_dir} 里找不到标签表")

    def detect(self, image: bytes, padding: float = 0.004) -> list[LayoutBlock]:
        from PIL import Image

        with Image.open(io.BytesIO(image)) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            resized = rgb.resize((self.INPUT_SIZE, self.INPUT_SIZE), Image.BILINEAR)
            pixels = np.asarray(resized, dtype=np.float32) / 255.0

        tensor = np.transpose(pixels, (2, 0, 1))[None, ...]
        logits, pred_boxes = self.session.run(
            ["logits", "pred_boxes"], {"pixel_values": tensor}
        )

        scores = _softmax(logits[0])
        best_scores = scores.max(axis=1)
        best_labels = scores.argmax(axis=1)

        blocks: list[LayoutBlock] = []
        for index in range(scores.shape[0]):
            score = float(best_scores[index])
            if score < self.threshold:
                continue
            cx, cy, w, h = (float(v) for v in pred_boxes[0][index])
            x0 = (cx - w / 2) * width
            y0 = (cy - h / 2) * height
            x1 = (cx + w / 2) * width
            y1 = (cy + h / 2) * height
            x0 = max(0.0, min(1.0, x0 / width))
            y0 = max(0.0, min(1.0, y0 / height))
            x1 = max(0.0, min(1.0, x1 / width))
            y1 = max(0.0, min(1.0, y1 / height))
            if x1 - x0 < 0.002 or y1 - y0 < 0.002:
                continue
            label = self.labels[int(best_labels[index])]
            blocks.append(
                LayoutBlock(
                    bbox=BBox(x0, y0, x1, y1),
                    category=PADDLE_LABEL_TO_CATEGORY.get(label, Category.TEXT),
                    ink_ratio=score,
                )
            )
        blocks.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))
        return blocks


def _softmax(array: np.ndarray) -> np.ndarray:
    shifted = array - array.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def get_detector(model_dir: str | None = None):
    """拿版面检测器。

    给了模型目录就用官方的 PP-DocLayoutV3，没给或者 onnxruntime 没装就
    退回自带的投影法——两条路都能跑，只是复杂版面的效果不一样。
    """

    candidate = model_dir or os.environ.get("TRACECELL_LAYOUT_MODEL")
    if candidate:
        try:
            return PaddleLayoutDetector(candidate)
        except Exception:  # noqa: BLE001 - 模型有问题就退回内置的
            pass
    return detect


def crop(image: bytes, bbox: BBox, padding: float = 0.0) -> bytes:
    """按归一化坐标裁剪图像，返回 PNG 字节。

    图像为空或无法解码时返回一张 1×1 的空白图。回放模式本来就没有原图，
    裁剪只是为了把调用链跑通，不应该在这里炸掉。
    """

    from PIL import Image

    source = _decoded(image)
    if source is None:
        buffer = io.BytesIO()
        Image.new("RGB", (1, 1), (255, 255, 255)).save(buffer, format="PNG")
        return buffer.getvalue()

    width, height = source.size
    left, top, right, bottom = bbox.to_pixels(width, height)
    if padding > 0:
        pad_x = int((right - left) * padding)
        pad_y = int((bottom - top) * padding)
        left = max(0, left - pad_x)
        top = max(0, top - pad_y)
        right = min(width, right + pad_x)
        bottom = min(height, bottom + pad_y)
    # 裁剪框至少留 1×1。bbox 落在图像右边界或下边界时，朴素的钳位会算出
    # 一个宽或高为 0 的框，PIL 保存时会抛 "cannot write empty image"。
    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    buffer = io.BytesIO()
    source.crop((left, top, right, bottom)).save(buffer, format="PNG")
    return buffer.getvalue()
