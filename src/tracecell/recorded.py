"""录制数据。

样例里 Jev 的分数是真实模型的输出，不是手写的。手写的分数只能让演示按
预期走，证明不了任何事——所以分数单独录在 ``samples/jev_scores.json`` 里，
生成脚本把它读回来填进会话，生成过程依然可复现。

``samples/real_ocr/`` 是同一回事：那是 MiniCPM-OCR 1.0 的真实输出记录。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_scores(path: str | Path) -> dict[str, Any]:
    """读入录制的 Jev 分数。没有文件就返回空 dict。"""

    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def apply_scores(
    candidates: list[dict[str, Any]],
    region_scores: dict[str, float] | None,
) -> int:
    """把录制的分数填到候选上。返回实际填上的条数。"""

    if not region_scores:
        return 0
    applied = 0
    for candidate in candidates:
        value = region_scores.get(candidate.get("text", ""))
        if value is None:
            continue
        candidate["score"] = round(float(value), 6)
        candidate["verdict"] = "scored"
        applied += 1
    return applied
