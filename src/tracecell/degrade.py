"""降级策略。

三个模型任何一个不可用，系统都得给出**明确的**行为，而不是崩掉或者
悄悄给出错误结论：

* 解析服务挂了 —— 没有原文就没有任何结论，直接失败，但要说清是哪个服务。
* 评分服务挂了 —— 拿不到分数就意味着"不知道可不可信"，这时候必须全部
  转人工，绝不能把"没评分"当成"没问题"放过去。
* 升级复核挂了 —— 这一格转人工，不要重试到预算耗尽。

包装器负责捕获异常并记账，核验闭环读这份账，在账本里如实写明"本次核验
有一处降级"。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .providers.base import ScoreResult
from .types import Verdict


@dataclass
class Degradation:
    """一次降级事件。"""

    component: str
    reason: str
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"component": self.component, "reason": self.reason, "at": self.at}


class DegradationLog:
    """降级记录。"""

    def __init__(self) -> None:
        self.events: list[Degradation] = []

    def record(self, component: str, reason: str) -> None:
        self.events.append(Degradation(component=component, reason=reason))

    @property
    def failed(self) -> bool:
        return bool(self.events)

    def components(self) -> list[str]:
        return sorted({event.component for event in self.events})

    def to_dict(self) -> dict[str, Any]:
        return {
            "degraded": self.failed,
            "components": self.components(),
            "events": [event.to_dict() for event in self.events],
        }


class ResilientScorer:
    """评分服务的包装。

    先重试几次——网络抖动和模型服务重启是常事，一次失败就放弃会把本来
    能核验的单据推进人工队列。重试用完还不行，才降级成"没评上分"。
    """

    def __init__(
        self,
        scorer,
        log: DegradationLog | None = None,
        retries: int = 2,
        backoff: float = 1.5,
    ) -> None:
        self.scorer = scorer
        self.log = log or DegradationLog()
        self.retries = retries
        self.backoff = backoff

    def score(self, image: bytes, text: str, category: str = "text") -> ScoreResult:
        try:
            return _with_retry(
                lambda: self.scorer.score(image, text, category), self.retries, self.backoff
            )
        except Exception as exc:  # noqa: BLE001 - 任何故障都要降级而不是崩
            self.log.record("jev", f"评分失败：{exc}")
            return ScoreResult(None, Verdict.FAILED, f"评分服务不可用：{exc}")

    def score_best(
        self, image: bytes, candidates: list[str], category: str = "text"
    ) -> list[ScoreResult]:
        try:
            return _with_retry(
                lambda: self.scorer.score_best(image, candidates, category),
                self.retries,
                self.backoff,
            )
        except Exception as exc:  # noqa: BLE001
            self.log.record("jev", f"择优失败：{exc}")
            return [ScoreResult(None, Verdict.FAILED, f"评分服务不可用：{exc}") for _ in candidates]


def _with_retry(action, retries: int, backoff: float):
    """执行一次，失败就退避重试。最后一次失败把异常抛出去。"""

    delay = backoff
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 - 重试是这里的目的
            last = exc
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
    raise last  # type: ignore[misc]


class ResilientEscalator:
    """升级复核的包装。挂了返回空串，调用方据此转人工。"""

    def __init__(
        self,
        escalator,
        log: DegradationLog | None = None,
        retries: int = 1,
        backoff: float = 1.5,
    ) -> None:
        self.escalator = escalator
        self.log = log or DegradationLog()
        self.retries = retries
        self.backoff = backoff

    def review(self, image: bytes, question: str, hint: str = "") -> str:
        if self.escalator is None:
            return ""
        try:
            return _with_retry(
                lambda: self.escalator.review(image, question, hint), self.retries, self.backoff
            )
        except Exception as exc:  # noqa: BLE001
            self.log.record("minicpm-v", f"升级复核失败：{exc}")
            return ""

    def locate(self, image: bytes, description: str) -> dict[str, Any] | None:
        locate = getattr(self.escalator, "locate", None)
        if not callable(locate):
            return None
        try:
            return locate(image, description)
        except Exception as exc:  # noqa: BLE001
            self.log.record("minicpm-v", f"视觉定位失败：{exc}")
            return None


class ResilientParser:
    """解析服务的包装。

    解析失败没有降级空间——没有原文就没有任何结论。这里只做一件事：
    把异常翻译成一句人能看懂的话。
    """

    def __init__(self, parser, log: DegradationLog | None = None) -> None:
        self.parser = parser
        self.log = log or DegradationLog()

    def parse_page(self, image: bytes, profile=None):
        try:
            return self.parser.parse_page(image, profile)
        except Exception as exc:  # noqa: BLE001
            self.log.record("minicpm-ocr", f"解析失败：{exc}")
            raise RuntimeError(
                f"文档解析失败，无法继续核验。检查 OCR 服务是否可用（{exc}）"
            ) from exc

    def parse_region(self, image: bytes, bbox=None, profile=None) -> str:
        try:
            return self.parser.parse_region(image, bbox, profile)
        except Exception as exc:  # noqa: BLE001
            self.log.record("minicpm-ocr", f"区域解析失败：{exc}")
            return ""
