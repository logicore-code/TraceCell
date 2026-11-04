"""离线回放后端。

这个模块让整套流水线在**没有 GPU、没有模型权重**的机器上也能完整跑通：
界面、预算决策、路由、对账、报表，全部是真实代码在跑，只是把三个模型
的调用换成了录制好的数据。

录制数据长这样（``samples/sessions/*.json``）：

    {
      "documents": [
        {
          "doc_id": "invoice-001",
          "source": "samples/invoice_001.png",
          "regions": [
            {"id": "r000", "bbox": {...}, "category": "text",
             "text": "价税合计（小写）￥18680.00",
             "candidates": [
               {"text": "价税合计（小写）￥18680.00", "origin": "primary",
                "score": 0.97, "verdict": "scored"},
               {"text": "价税合计（小写）￥18880.00", "origin": "reparse-16x",
                "score": 0.22, "verdict": "scored"}
             ]}
          ],
          "fields": [
            {"name": "价税合计", "value": "18680.00", "region_ids": ["r000"]}
          ]
        }
      ]
    }

需要说明的是：回放里的分数是录制时的真实分数，**不是**现场算出来的。
真要重新算，用 ``--provider minicpm`` 接上真实模型。回放模式的定位是
"让评审和 CI 不用装模型也能验证整条链路"，不是"假装有模型"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..types import BBox, Candidate, Category, Region, Verdict
from .base import PageParse, ParseProfile, ScoreResult


class ReplaySession:
    """一次录制的解析会话。"""

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.documents: dict[str, dict[str, Any]] = {
            doc["doc_id"]: doc for doc in data.get("documents", [])
        }

    @classmethod
    def load(cls, path: str | Path) -> "ReplaySession":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_dir(
        cls, directory: str | Path, include_variants: bool = False
    ) -> "ReplaySession":
        """把一个目录下的所有会话文件合并。

        样例按场景分文件存放（发票、合同、报销单一份，财务报表和购销合同
        一份），跑完整链路时需要一起加载。

        标了 ``variant_of`` 的文档是同一份单据的另一种形态（比如同一份合同
        的 PNG 版和两页 PDF 版）。默认不参与汇总，免得统计里出现两份一样的
        单据；要单独跑可以直接按 doc_id 取。
        """

        merged: dict[str, Any] = {"documents": []}
        for path in sorted(Path(directory).glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for document in data.get("documents", []):
                # 只收能回放的会话（带 regions）。同目录下的 real_ocr.json
                # 记的是模型原始输出，doc_id 会跟会话撞车，不能混进来。
                if "regions" not in document:
                    continue
                if document.get("variant_of") and not include_variants:
                    continue
                merged["documents"].append(document)
        return cls(merged)

    def document(self, doc_id: str) -> dict[str, Any] | None:
        return self.documents.get(doc_id)

    def doc_ids(self) -> list[str]:
        return list(self.documents)

    def find_by_hash(self, digest: str) -> str | None:
        """按文件内容的哈希反查文档标识。

        上传一张图进来时，文件名未必对得上，但内容一样就应该是同一份。
        """

        for doc_id, document in self.documents.items():
            if document.get("sha256") == digest:
                return doc_id
        return None

    def find_by_source(self, filename: str) -> str | None:
        """按源文件名反查文档标识。

        批量处理时文件名和会话里的标识未必一致（contract_001.png 对应的
        标识可能是 contract-001），对不上就按文件名找。
        """

        target = Path(str(filename)).name
        for doc_id, document in self.documents.items():
            source = str(document.get("source", ""))
            if source and Path(source).name == target:
                return doc_id
        # 再退一步：只比主干名字，去掉下划线和短横线的差别
        stem = Path(target).stem.replace("_", "").replace("-", "").lower()
        for doc_id, document in self.documents.items():
            source = str(document.get("source", ""))
            if source:
                other = Path(source).stem.replace("_", "").replace("-", "").lower()
                if other == stem:
                    return doc_id
        return None


class ReplayParser:
    """回放解析结果。"""

    def __init__(self, session: ReplaySession, doc_id: str) -> None:
        self.session = session
        self.doc_id = doc_id

    def parse_page(self, image: bytes, profile: ParseProfile | None = None) -> PageParse:
        document = self.session.document(self.doc_id)
        if document is None:
            raise KeyError(f"回放会话里没有文档 {self.doc_id}")

        regions: list[Region] = []
        for raw in document.get("regions", []):
            candidates = [
                Candidate(
                    text=item["text"],
                    origin=item.get("origin", "primary"),
                    score=item.get("score"),
                    verdict=Verdict(item.get("verdict", "not_scored")),
                    detail=item.get("detail", ""),
                )
                for item in raw.get("candidates", [])
            ]
            regions.append(
                Region(
                    id=raw["id"],
                    page=raw.get("page", 1),
                    bbox=BBox(**raw["bbox"]),
                    category=Category(raw.get("category", "text")),
                    text=raw.get("text", ""),
                    candidates=candidates,
                )
            )
        return PageParse(
            regions=regions,
            markdown=document.get("markdown", ""),
            meta={
                "replayed": True,
                "doc_id": self.doc_id,
                "scenario": document.get("schema", ""),
                "scenario_label": document.get("label", ""),
            },
        )

    def parse_region(self, image: bytes, bbox: BBox | None = None, profile: ParseProfile | None = None) -> str:
        page = self.parse_page(image, profile)
        if bbox is None:
            return page.markdown
        for region in page.regions:
            if region.bbox.iou(bbox) > 0.5:
                return region.text
        return ""


class ReplayScorer:
    """回放 Jev 评分。"""

    def __init__(self, session: ReplaySession, doc_id: str) -> None:
        self.session = session
        self.doc_id = doc_id

    def _candidates(self) -> list[Candidate]:
        document = self.session.document(self.doc_id)
        if document is None:
            return []
        result: list[Candidate] = []
        for raw in document.get("regions", []):
            for item in raw.get("candidates", []):
                result.append(
                    Candidate(
                        text=item["text"],
                        origin=item.get("origin", "primary"),
                        score=item.get("score"),
                        verdict=Verdict(item.get("verdict", "not_scored")),
                        detail=item.get("detail", ""),
                    )
                )
        return result

    def score(self, image: bytes, text: str, category: str = "text") -> ScoreResult:
        from .base import CATEGORY_TO_JEV  # noqa: F401 - 保持类别映射可用

        for candidate in self._candidates():
            if candidate.text == text:
                if candidate.scored:
                    return ScoreResult(candidate.score, Verdict.SCORED, candidate.detail)
                return ScoreResult(None, candidate.verdict, candidate.detail)
        return ScoreResult(None, Verdict.NOT_SCORED, "回放数据里没有这条候选")

    def score_best(
        self, image: bytes, candidates: list[str], category: str = "text"
    ) -> list[ScoreResult]:
        return [self.score(image, text, category) for text in candidates]


class ReplayEscalator:
    """回放多模态升级复核。"""

    def __init__(self, session: ReplaySession, doc_id: str) -> None:
        self.session = session
        self.doc_id = doc_id

    def review(self, image: bytes, question: str, hint: str = "") -> str:
        document = self.session.document(self.doc_id)
        if document is None:
            return ""
        for raw in document.get("regions", []):
            for item in raw.get("candidates", []):
                if item.get("origin") == "vlm-review":
                    return item["text"]
        return ""

    def locate(self, image: bytes, description: str) -> dict[str, Any] | None:
        """回放数据里没有定位信息。"""

        return None
