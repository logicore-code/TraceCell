"""TraceCell —— 让每一个被机器读出来的值，都能指回它的出处。

一句话概括：**别人给你一个 OCR 结果，TraceCell 给你一份"可以签字"的结论**。

仓库里的四条主线：

* ``types``     数据契约（区域、候选、字段、账本）
* ``risk``      置信度与风险
* ``budget``    逐格预算决策
* ``routing``   文档级路由（免检 / 抽检 / 全检）

以及把三个 MiniCPM 模型串起来的 ``agent.loop``。
"""

from __future__ import annotations

__version__ = "0.1.0"

from .types import (  # noqa: F401
    BBox,
    Candidate,
    Category,
    DocumentLedger,
    FieldValue,
    Finding,
    Region,
    Severity,
    Status,
    Verdict,
)

__all__ = [
    "__version__",
    "BBox",
    "Candidate",
    "Category",
    "DocumentLedger",
    "FieldValue",
    "Finding",
    "Region",
    "Severity",
    "Status",
    "Verdict",
]
