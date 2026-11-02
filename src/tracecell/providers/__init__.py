"""模型接入层。

每一类模型对应一个协议，实现可以互换：

* Parser    —— MiniCPM-OCR 1.0（解析）、ReplayParser（回放）
* Scorer    —— MiniCPM-OCR Jev（评分）、ReplayScorer（回放）
* Escalator —— MiniCPM-V 4.7（升级复核）、ReplayEscalator（回放）
"""

from .base import (  # noqa: F401
    CATEGORY_TO_JEV,
    PROFILE_ALT,
    PROFILE_PRIMARY,
    Escalator,
    PageParse,
    ParseProfile,
    Parser,
    ScoreResult,
    Scorer,
)

__all__ = [
    "CATEGORY_TO_JEV",
    "PROFILE_ALT",
    "PROFILE_PRIMARY",
    "Escalator",
    "PageParse",
    "ParseProfile",
    "Parser",
    "ScoreResult",
    "Scorer",
]
