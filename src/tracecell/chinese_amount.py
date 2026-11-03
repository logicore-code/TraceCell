"""中文金额解析。

合同里金额写两遍：阿拉伯数字的小写，和中文的大写。法律上以大写为准，
因为小写可以改，大写改起来麻烦。核验时两遍必须对得上——这是一件纯靠
OCR 解决不了的事：大写那一行识别成"伍拾捌万元整"还是"伍拾捌万圆整"，
机器读对了也只能说"读到了这串字"，没法判断它到底值多少钱。

所以要把中文数字解析成数值，再跟小写那遍比。

规则按《正确填写票据和结算凭证的基本规定》那套：
    零壹贰叁肆伍陆柒捌玖  拾佰仟  万亿  元圆整角分
十到十九前面要带"壹"（拾 = 10，壹拾 = 10，壹拾伍 = 15）。
"""

from __future__ import annotations

import re
from decimal import Decimal

# 大写数字。括号里的是口语里也常见的写法，一并认。
CN_DIGITS: dict[str, int] = {
    "零": 0, "〇": 0, "壹": 1, "一": 1, "贰": 2, "二": 2, "叁": 3, "三": 3,
    "肆": 4, "四": 4, "伍": 5, "五": 5, "陆": 6, "六": 6, "柒": 7, "七": 7,
    "捌": 8, "八": 8, "玖": 9, "九": 9,
}

# 数量单位
CN_UNITS: dict[str, int] = {"拾": 10, "十": 10, "佰": 100, "百": 100, "仟": 1000, "千": 1000}

# 节权位
CN_SECTIONS: dict[str, int] = {"万": 10**4, "亿": 10**8}

# 货币单位。圆是元的异体。
CN_CURRENCY = {"元", "圆", "圆整", "元整", "整"}

# 角、分
CN_DECIMALS: dict[str, int] = {"角": 1, "分": 1}

AMOUNT_PATTERN = re.compile(r"[零〇壹一贰二叁三肆四伍五陆六柒七捌八玖九拾十佰百仟千万亿圓圆元角分整]+")


def normalize_amount_text(text: str) -> str:
    """把金额文本里的大写单位整理成规范写法。

    "圆" 换成 "元"，去掉多余的 "整"，方便解析。
    """

    if not text:
        return ""
    cleaned = text.replace("圓", "圆").replace("圆", "元")
    cleaned = re.sub(r"元+整+", "元", cleaned)
    cleaned = re.sub(r"整+$", "", cleaned)
    return cleaned


def parse_chinese_amount(text: str) -> Decimal | None:
    """把中文金额解析成数值。

    解析不出来就返回 None——调用方据此说明"这一项没法核"，而不是当成 0。

        >>> parse_chinese_amount("伍拾捌万元整")
        Decimal('580000')
        >>> parse_chinese_amount("壹拾肆万伍仟叁佰壹拾捌元整")
        Decimal('145318')
        >>> parse_chinese_amount("壹佰零伍元")
        Decimal('105')
    """

    if not text:
        return None

    cleaned = normalize_amount_text(text)
    if not cleaned:
        return None

    # 只取金额主体，丢掉前面的标签
    match = AMOUNT_PATTERN.search(cleaned)
    if not match:
        return None
    body = match.group(0)

    # 角、分：数字写在单位前面，格式是「肆角伍分」
    jiao = fen = 0
    decimal_match = re.search(
        r"([零〇壹一贰二叁三肆四伍五陆六柒七捌八玖九])\s*角"
        r"(?:([零〇壹一贰二叁三肆四伍五陆六柒七捌八玖九])\s*分)?",
        body,
    )
    if decimal_match:
        jiao = CN_DIGITS[decimal_match.group(1)]
        fen = CN_DIGITS[decimal_match.group(2)] if decimal_match.group(2) else 0
        body = body[: decimal_match.start()] + body[decimal_match.end() :]

    body = body.replace("元", "").replace("整", "")

    total = 0
    section = 0
    number = 0
    zero_pending = False

    for char in body:
        if char in CN_DIGITS:
            number = CN_DIGITS[char]
            zero_pending = False
        elif char in CN_UNITS:
            if number == 0 and not zero_pending:
                # 拾单独出现是 10（拾 = 壹拾）
                number = 1
            section += number * CN_UNITS[char]
            number = 0
            zero_pending = False
        elif char in CN_SECTIONS:
            section = (section + number) * CN_SECTIONS[char]
            total += section
            section = 0
            number = 0
            zero_pending = False
        elif char == "零" or char == "〇":
            number = 0
            zero_pending = True
        else:
            return None

    value = total + section + number
    if jiao or fen:
        value += Decimal(jiao) / 10 + Decimal(fen) / 100

    if value == 0 and not any(ch in CN_DIGITS for ch in body):
        return None
    return Decimal(value).quantize(Decimal("0.01"))


def find_chinese_amount(text: str) -> Decimal | None:
    """在一段文字里找出中文金额。

    用于正文里的数字引用——合同条款里写"甲方应支付人民币伍拾捌万元整"，
    这句话里的金额同样可以拿来跟结构化字段比对。
    """

    if not text:
        return None
    match = AMOUNT_PATTERN.search(normalize_amount_text(text))
    if not match:
        return None
    return parse_chinese_amount(match.group(0))


def amounts_agree(left: str, right: str, tolerance: float = 0.01) -> bool | None:
    """比较中文金额与阿拉伯数字金额是否一致。

    返回 None 表示有一边解析不出来，没法比。
    """

    chinese = parse_chinese_amount(left)
    arabic = _parse_arabic(right)
    if chinese is None or arabic is None:
        return None
    return abs(float(chinese - arabic)) <= tolerance


def _parse_arabic(text: str) -> Decimal | None:
    cleaned = re.sub(r"[^\d.]", "", text or "")
    if not cleaned:
        return None
    try:
        return Decimal(cleaned).quantize(Decimal("0.01"))
    except Exception:  # noqa: BLE001 - 解析不了就当没有
        return None
