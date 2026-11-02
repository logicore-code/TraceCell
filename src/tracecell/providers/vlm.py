"""MiniCPM-V 4.7 升级复核后端。

当 OCR 解析和 Jev 评分都拿不准的时候，把这一块单独拎出来，问一个更
强的视觉模型。4.7 是 35B-A3B 的 MoE 模型，在 1B 激活参数下就能在手机上
跑，端侧部署完全可行。

升级复核不是让模型"重新读一遍"，而是问一个**可以证伪的窄问题**：

    这块区域里的金额是 18680.00 还是 18880.00？只回答数字。

窄问题的好处是答案短、可判定、可直接变成新候选，再交给 Jev 复评。宽泛
的"请描述这块内容"会产生一堆无法自动核验的散文。
"""

from __future__ import annotations

from typing import Any

import requests

from .ocr import encode_data_uri

DEFAULT_TIMEOUT = 300


class OpenAICompatVlm:
    """OpenAI 兼容接口的视觉复核器。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8001",
        model: str = "openbmb/MiniCPM-V-4.7-35B-A3B",
        timeout: int = DEFAULT_TIMEOUT,
        temperature: float = 0.1,
        max_tokens: int = 512,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session = session or requests.Session()

    def review(self, image: bytes, question: str, hint: str = "") -> str:
        """对一块区域提问，返回模型的回答。"""

        prompt = question
        if hint:
            prompt = f"{question}\n参考信息：{hint}"

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encode_data_uri(image)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions", json=body, timeout=self.timeout
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content", "").strip()

    def locate(self, image: bytes, description: str) -> dict[str, Any] | None:
        """视觉定位：在图里找一段描述，返回归一化坐标。

        返回 {"x0": .., "y0": .., "x1": .., "y1": ..} 或 None。
        """

        prompt = (
            f"请在图中找到「{description}」所在的位置，"
            "以归一化坐标 (x0, y0, x1, y1) 返回，坐标范围 0 到 1，"
            "只输出 JSON，不要其它文字。"
        )
        answer = self.review(image, prompt)
        return _parse_bbox(answer)


def _parse_bbox(text: str) -> dict[str, Any] | None:
    """从模型回答里抠出坐标。模型偶尔会加前缀说明，容错一下。"""

    import re

    match = re.search(
        r"\(?\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*,\s*(-?\d*\.?\d+)\s*\)?",
        text or "",
    )
    if not match:
        return None
    try:
        x0, y0, x1, y1 = (float(value) for value in match.groups())
    except ValueError:
        return None
    if max(x0, y0, x1, y1) > 1.5:  # 模型给了像素坐标
        return None
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def build_review_question(region_text: str, candidates: list[str]) -> str:
    """根据候选生成一个窄问题。

    两个候选只差一两个字符时，问题就是"到底哪个是对的"；只有一个候选时，
    问题变成"把这一块原样抄一遍"。
    """

    distinct = []
    for candidate in candidates:
        if candidate not in distinct:
            distinct.append(candidate)
    if len(distinct) >= 2:
        options = "，还是 ".join(distinct[:3])
        return f"这块区域的内容是「{options}」中的哪一个？只回答正确的那个，不要解释。"
    return "请把图中的文字原样抄写出来，不要添加任何说明。"
