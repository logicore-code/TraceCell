"""MiniCPM-OCR Jev 客户端。

Jev 是一个 2B 的判别模型：给它一块原文区域图和一份候选识别结果，它输出
0~1 的分数，越高表示这份结果越可能是对的。

这里有三件事值得注意，都是踩过的坑：

1. **必须检查 verdict。** 官方服务在图像解码失败、渲染失败、前向失败
   时会返回固定的分数（0.0 或 -1.0 之类）。这些是"没跑成"，不是"跑出来
   说不对"。本客户端一律把这种情况标成 `failed`，绝不把固定值当成低分
   塞进置信度计算。
2. **图片是裸 base64。** 官方 Docker 接口不收 `data:image/png;base64,`
   前缀，多加一个前缀会导致解码失败，而失败的表现又是固定分数。
3. **分数不是概率。** 0.9 不等于 90% 正确。阈值必须在自己的数据上标，
   见 calibration.py。
"""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from typing import Any, Iterable

import requests

from ..types import Verdict
from .base import ScoreResult

DEFAULT_TIMEOUT = 120


class JevError(RuntimeError):
    """Jev 服务调用失败。"""


def encode_image(image: bytes) -> str:
    """裸 base64，不带 data URI 前缀。"""

    return base64.b64encode(image).decode("ascii")


class HttpJevScorer:
    """对接官方 Docker / SDK 服务。

    启动方式见 docs/deployment.md。服务本身带渲染器，所以只需要传原图、
    候选文本和区域类别，渲染由服务端完成。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8030",
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = 2,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()

    # -- 健康检查 ---------------------------------------------------------
    def health(self) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}/healthz", timeout=10)
        response.raise_for_status()
        return response.json()

    def wait_ready(self, attempts: int = 30, interval: float = 5.0) -> bool:
        """等待服务就绪。返回是否成功。"""

        for _ in range(attempts):
            try:
                if self.health().get("status") == "ok":
                    return True
            except Exception:
                pass
            time.sleep(interval)
        return False

    # -- 单条评分 ---------------------------------------------------------
    def score(self, image: bytes, text: str, category: str = "text") -> ScoreResult:
        results = self.score_many([(image, text, category)])
        return results[0] if results else ScoreResult(None, Verdict.FAILED, "服务返回空结果")

    def score_many(self, items: Iterable[tuple[bytes, str, str]]) -> list[ScoreResult]:
        """批量评分。每一项是 (区域图, 候选文本, 类别)。"""

        payload_items = []
        for index, (image, text, category) in enumerate(items):
            payload_items.append(
                {
                    "case_id": f"case-{index:06d}",
                    "image": encode_image(image),
                    "ocr_text": text,
                    "category": category,
                }
            )
        if not payload_items:
            return []

        body = {"input_form": "ternary", "items": payload_items}
        response = self._post("/v1/rm/score", body)
        raw_results = response.get("results") or []
        return [self._to_result(item) for item in raw_results]

    # -- 多候选择优 -------------------------------------------------------
    def score_best(
        self, image: bytes, candidates: list[str], category: str = "text"
    ) -> list[ScoreResult]:
        """对同一区域的多个候选逐一评分。

        官方接口一次调用就能完成，比循环调用 score 省不少往返。
        """

        if not candidates:
            return []
        body = {
            "input_form": "ternary",
            "image": encode_image(image),
            "category": category,
            "candidates": candidates,
        }
        response = self._post("/v1/rm/score_best", body)
        scores = response.get("scores") or []
        verdicts = response.get("verdicts") or []
        details = response.get("details") or []
        results: list[ScoreResult] = []
        for index in range(len(candidates)):
            verdict = verdicts[index] if index < len(verdicts) else "not_scored"
            detail = details[index] if index < len(details) else ""
            results.append(self._to_result({"verdict": verdict, "detail": detail, "score": (scores[index] if index < len(scores) else None)}))
        return results

    def score_best_batch(
        self, blocks: list[tuple[bytes, list[str], str]]
    ) -> list[list[ScoreResult]]:
        """跨多个区域批量择优。"""

        if not blocks:
            return []
        payload_blocks = []
        for index, (image, candidates, category) in enumerate(blocks):
            payload_blocks.append(
                {
                    "case_id": f"block-{index:06d}",
                    "image": encode_image(image),
                    "candidates": candidates,
                    "category": category,
                }
            )
        body = {"input_form": "ternary", "blocks": payload_blocks}
        response = self._post("/v1/rm/score_best_batch", body)
        raw_blocks = response.get("blocks") or response.get("results") or []
        results: list[list[ScoreResult]] = []
        for block in raw_blocks:
            scores = block.get("scores") or []
            verdicts = block.get("verdicts") or []
            results.append(
                [
                    self._to_result(
                        {
                            "verdict": verdicts[i] if i < len(verdicts) else "not_scored",
                            "score": scores[i] if i < len(scores) else None,
                            "detail": block.get("detail", ""),
                        }
                    )
                    for i in range(len(block.get("candidates", scores)))
                ]
            )
        return results

    # -- 内部 -------------------------------------------------------------
    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(url, json=body, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except Exception as exc:  # noqa: BLE001 - 统一转成 JevError
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise JevError(f"调用 {url} 失败：{last_error}")

    @staticmethod
    def _to_result(item: dict[str, Any]) -> ScoreResult:
        """把服务端返回统一成 ScoreResult。

        关键在这里：verdict 不是 scored 时，分数一律丢弃。
        """

        verdict_raw = str(item.get("verdict", "not_scored"))
        detail = str(item.get("detail", "") or "")
        if verdict_raw == "scored":
            score = item.get("prob_yes", item.get("score"))
            try:
                return ScoreResult(float(score), Verdict.SCORED, detail)
            except (TypeError, ValueError):
                return ScoreResult(None, Verdict.FAILED, detail or "分数字段无法解析")
        if verdict_raw in ("failed", "error"):
            return ScoreResult(None, Verdict.FAILED, detail or "评分服务执行失败")
        return ScoreResult(None, Verdict.NOT_SCORED, detail or "评分未执行")


class LlamaCppJevScorer:
    """跑在 llama.cpp 上的 Jev（纯 CPU 端侧）。

    官方给的两条部署路里，Docker 那条要 NVIDIA GPU；SDK 那条虽然支持
    llama.cpp 后端，但要另起一个 Python 服务层。这里直接对 llama-server
    说话，把中间那层省掉：

        llama-server \
          -m MiniCPM-OCR-Jev-Q4_K_M.gguf \
          --mmproj mmproj-model-f16.gguf \
          -c 8192 --port 18499

    两个要点：

    * **图像顺序是有讲究的**：原图在前、候选文本的渲染图在后。反过来问的
      就不是同一个问题了，而且不会报错，只会给一个错的数。
    * **渲染图要自己提供**：官方 SDK 用 Chromium 加中文字体渲染候选文本，
      这里用 `tracecell.render` 做同样的事。渲染环境不同，绝对分数会差，
      排序是稳的。
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18499",
        timeout: int = 600,
        renderer=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._renderer = renderer

    def _render(self, text: str) -> bytes:
        if self._renderer is not None:
            return self._renderer(text)
        from ..render import render_text

        return render_text(text)

    def score(self, image: bytes, text: str, category: str = "text") -> ScoreResult:
        """给一个候选打分。类别在这里用不上，模型只判断视觉一致性。"""

        try:
            body = {
                "state": text,
                "files": [
                    _data_url(image),
                    _data_url(self._render(text)),
                ],
                "questions": {
                    "consistency": {"type": "choice", "criteria": {"0": None, "1": None}}
                },
            }
            request = urllib.request.Request(
                f"{self.base_url}/v1/systemone",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.load(response)
            probabilities = payload["answers"]["consistency"]["probabilities"]
            value = float(probabilities["1"])
        except Exception as exc:  # noqa: BLE001 - 失败要能降级而不是崩
            return ScoreResult(None, Verdict.FAILED, f"Jev 评分失败：{exc}")
        return ScoreResult(value, Verdict.SCORED)

    def score_best(
        self, image: bytes, candidates: list[str], category: str = "text"
    ) -> list[ScoreResult]:
        return [self.score(image, text, category) for text in candidates]


def _data_url(image: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(image).decode()
