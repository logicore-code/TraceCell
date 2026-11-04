"""文本归一化。

对账、候选一致性判断都要回答同一个问题："这两段文字说的是不是一回事？"
直接比字符串会把全角半角、空格、标点的差异当成矛盾，所以统一放在这里。

只做保守的归一化：去掉空白、统一全角、常见的金额/日期写法归一。不会
做同义词映射——那属于业务层，不该由底层工具猜。
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_FULLWIDTH_PUNCT = "．，：；（）【】％－～"


def to_halfwidth(text: str) -> str:
    """全角转半角，顺带把中文标点换成 ASCII 对应物。"""

    out = []
    for ch in text:
        if ch in _FULLWIDTH_PUNCT:
            out.append(
                {
                    "．": ".",
                    "，": ",",
                    "：": ":",
                    "；": ";",
                    "（": "(",
                    "）": ")",
                    "【": "[",
                    "】": "]",
                    "％": "%",
                    "－": "-",
                    "～": "~",
                }[ch]
            )
            continue
        code = ord(ch)
        if code == 0x3000:  # 全角空格
            out.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            out.append(chr(code - 0xFEE0))
        else:
            out.append(ch)
    return "".join(out)


def squeeze(text: str) -> str:
    """去掉所有空白字符。"""

    return re.sub(r"\s+", "", text or "")


def normalize(text: str) -> str:
    """通用归一化：全角转半角、去空白、小写。

    用于判断两段文字"是否同一个值"。返回的字符串只适合做相等性比较，
    不要拿它去展示给用户。
    """

    return squeeze(to_halfwidth(text or "")).lower()


def normalize_money(text: str) -> str | None:
    """把金额归一化成 "1234.56" 形式。

    支持 "¥18,680.00"、"18,680.00元"、"18680" 这类写法。
    大写金额（壹万捌仟...）不做转换，只在纯数字找不到时返回 None。
    """

    if not text:
        return None
    cleaned = to_halfwidth(text).replace(",", "").replace("，", "")
    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    try:
        value = Decimal(match.group(0))
    except InvalidOperation:
        return None
    return f"{value.quantize(Decimal('0.01')):f}"


_DATE_PATTERNS = (
    (r"(\d{4})\s*[-/年.]\s*(\d{1,2})\s*[-/月.]\s*(\d{1,2})\s*日?", (1, 2, 3)),
    (r"(\d{4})\s*[-/年.]\s*(\d{1,2})\s*月?$", (1, 2)),
)


def normalize_date(text: str) -> str | None:
    """把日期归一化成 ISO 形式，识别不了就返回 None。"""

    if not text:
        return None
    cleaned = to_halfwidth(text).strip()
    for pattern, groups in _DATE_PATTERNS:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        parts = [match.group(i) for i in groups]
        year, month = int(parts[0]), int(parts[1])
        day = int(parts[2]) if len(parts) > 2 else 1
        try:
            return f"{year:04d}-{month:02d}-{day:02d}"
        except ValueError:
            return None
    return None


def normalize_entity(text: str) -> str:
    """主体名称归一化：去掉公司后缀里的空格和常见标点。"""

    return normalize(text).replace("·", "").replace("（", "(").replace("）", ")")


def similarity(a: str, b: str) -> float:
    """两段文字的字符级相似度（Jaccard on 字符 bigram）。

    用来在对账时区分"完全矛盾"和"写法略有差异"。不是语义相似度，
    只在归一化之后仍然分不出来的时候做辅助判断。
    """

    left, right = normalize(a), normalize(b)
    if left == right:
        return 1.0
    if not left or not right:
        return 0.0
    if len(left) == 1 or len(right) == 1:
        return 1.0 if left == right else 0.0
    grams_left = {left[i : i + 2] for i in range(len(left) - 1)}
    grams_right = {right[i : i + 2] for i in range(len(right) - 1)}
    intersection = grams_left & grams_right
    union = grams_left | grams_right
    return len(intersection) / len(union) if union else 0.0


def diff_positions(left: str, right: str) -> list[int]:
    """返回两个字符串第一个不同的字符下标，方便在界面上高亮差异。"""

    left, right = normalize(left), normalize(right)
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return [index]
    if len(left) != len(right):
        return [min(len(left), len(right))]
    return []
