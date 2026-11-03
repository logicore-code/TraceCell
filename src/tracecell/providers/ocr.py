"""MiniCPM-OCR 1.0 客户端。

MiniCPM-OCR 1.0 是 1.3B 的文档解析模型：输入扫描件、翻拍件、弯曲页面，
输出带版面结构的 Markdown（表格转 HTML、公式转 LaTeX、印章和图表也能
处理）。官方流水线是三步：版面检测 → 逐块识别 → 按阅读顺序拼回。

TraceCell 沿用同样的三步，但把"版面检测"换成了自带的轻量切分器（见
layout.py），并且把每一步都做成可替换的：

* ``OpenAICompatOcr``：对接任何 OpenAI 兼容的多模态服务。官方推荐的
  ``transformers serve``、vLLM、SGLang 都走这条路。生产环境首选。
* ``LlamaCppOcr``：直接调 llama.cpp 的多模态 CLI，端侧 CPU 部署用。
* ``ReplayOcr``：回放录制的会话，用于离线演示和回归测试。

为什么逐块识别，不整页丢进去？为了拿到证据坐标。整页解析只给你一个
Markdown，看不出每个字段来自原图哪里；Jev 的输入是"一块区域图 + 一份候选
文本"，没有区域就没法评分。
"""

from __future__ import annotations

import base64
import json
import subprocess
from typing import Any, Callable

import requests

from ..types import BBox, Category, Region
from . import layout as layout_mod
from .base import TASK_PROMPTS, PageParse, ParseProfile

DEFAULT_TIMEOUT = 300


def encode_data_uri(image: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(image).decode("ascii")


class OpenAICompatOcr:
    """OpenAI 兼容接口的解析器。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        model: str = "openbmb/MiniCPM-OCR-1.0",
        timeout: int = DEFAULT_TIMEOUT,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        session: requests.Session | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.session = session or requests.Session()
        self.extra_body = extra_body or {}
        self._layout: Callable[[bytes], list[layout_mod.LayoutBlock]] | None = None

    def use_layout(self, detector: Callable[[bytes], list[layout_mod.LayoutBlock]]) -> None:
        """替换版面切分器。默认用内置的投影法实现。"""

        self._layout = detector

    # -- 单块识别 ---------------------------------------------------------
    def parse_region(
        self,
        image: bytes,
        bbox: BBox | None = None,
        profile: ParseProfile | None = None,
        category: str | None = None,
    ) -> str:
        profile = profile or ParseProfile()
        payload_image = image if bbox is None else layout_mod.crop(image, bbox, profile.padding)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encode_data_uri(payload_image)}},
                        {"type": "text", "text": prompt_for(profile, category)},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        body.update(self.extra_body)
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions", json=body, timeout=self.timeout
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("message") or {}).get("content", "").strip()

    # -- 整页解析 ---------------------------------------------------------
    def parse_page(self, image: bytes, profile: ParseProfile | None = None) -> PageParse:
        profile = profile or ParseProfile()
        detector = self._layout or layout_mod.detect
        blocks = detector(image)

        regions: list[Region] = []
        for index, block in enumerate(blocks):
            text = self.parse_region(image, block.bbox, profile)
            if not text:
                continue
            regions.append(
                Region(
                    id=f"r{index:03d}",
                    page=1,
                    bbox=block.bbox,
                    category=block.category,
                    text=text,
                    candidates=[],
                )
            )

        markdown = "\n\n".join(region.text for region in regions)
        return PageParse(regions=regions, markdown=markdown)


class LlamaCppOcr:
    """llama.cpp 多模态 CLI 后端。

    端侧部署用：模型权重和 mmproj 都在本地，CPU 也能跑。命令行的参数名
    跟 llama.cpp 的 mtmd CLI 一致。
    """

    def __init__(
        self,
        binary: str = "llama-mtmd-cli",
        model_path: str = "MiniCPM-OCR-1_0-Q4_K_M.gguf",
        mmproj_path: str = "mmproj-model-f16-4x.gguf",
        timeout: int = DEFAULT_TIMEOUT,
        extra_args: list[str] | None = None,
    ) -> None:
        self.binary = binary
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.timeout = timeout
        self.extra_args = extra_args or []

    def parse_region(
        self,
        image: bytes,
        bbox: BBox | None = None,
        profile: ParseProfile | None = None,
        category: str | None = None,
    ) -> str:
        profile = profile or ParseProfile()
        payload_image = image if bbox is None else layout_mod.crop(image, bbox, profile.padding)
        with open("/tmp/tracecell-region.png", "wb") as handle:
            handle.write(payload_image)
        command = [
            self.binary,
            "-m",
            self.model_path,
            "--mmproj",
            self.mmproj_path,
            "--image",
            "/tmp/tracecell-region.png",
            "--prompt",
            profile.prompt,
            "--temp",
            "0.1",
            "-n",
            str(profile.max_slice_nums * 40),
            *self.extra_args,
        ]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout)
        if completed.returncode != 0:
            raise RuntimeError(f"llama.cpp 调用失败：{completed.stderr[-500:]}")
        return _clean_cli_output(completed.stdout)

    def parse_page(self, image: bytes, profile: ParseProfile | None = None) -> PageParse:
        profile = profile or ParseProfile()
        blocks = layout_mod.detect(image)
        regions: list[Region] = []
        for index, block in enumerate(blocks):
            text = self.parse_region(image, block.bbox, profile)
            if not text:
                continue
            regions.append(
                Region(
                    id=f"r{index:03d}",
                    page=1,
                    bbox=block.bbox,
                    category=block.category,
                    text=text,
                )
            )
        return PageParse(
            regions=regions,
            markdown="\n\n".join(region.text for region in regions),
        )


def prompt_for(profile: ParseProfile, category: str | None = None) -> str:
    """按版面类别选任务前缀。

    profile.prompt 显式给定时用它；否则按类别取 MiniCPM-OCR 1.0 的标准
    任务指令。
    """

    if profile.prompt and profile.prompt != TASK_PROMPTS["text"]:
        return profile.prompt
    return TASK_PROMPTS.get(category or "text", TASK_PROMPTS["text"])


def _clean_cli_output(text: str) -> str:
    """去掉 llama.cpp 输出里的日志行。"""

    kept: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("llama_", "ggml_", "load_", "print_info", "init: ")):
            continue
        if stripped.startswith(">"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def regions_from_markdown(markdown: str, blocks: list[layout_mod.LayoutBlock]) -> list[Region]:
    """把整页解析出来的 Markdown 切成与版面块对齐的区域。

    有些服务端只支持整页输入（比如只暴露一个 chat 接口的部署）。这时
    按空行把 Markdown 拆段，再按顺序对应回版面块，尽量保住坐标。
    """

    paragraphs = [part.strip() for part in markdown.split("\n\n") if part.strip()]
    regions: list[Region] = []
    for index, paragraph in enumerate(paragraphs):
        block = blocks[index] if index < len(blocks) else None
        regions.append(
            Region(
                id=f"r{index:03d}",
                page=1,
                bbox=block.bbox if block else BBox(0.0, 0.0, 1.0, 1.0),
                category=block.category if block else Category.TEXT,
                text=paragraph,
            )
        )
    return regions


def save_session(path: str, session: dict[str, Any]) -> None:
    """把一次解析结果存成可回放的会话。"""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(session, handle, ensure_ascii=False, indent=2)
