"""用真实的 MiniCPM-V 4.7 做区域级升级复核。

闭环里最难的一环是「OCR 和 Jev 都搞不定，交给多模态模型问一个窄问题」。
这一步必须用真模型验证过，否则整条链只是纸上谈兵。

**两个模型没法同时驻留**（沙箱 3GB 内存，V 4.7 单独就占 2.3GB），所以录制
分两个阶段跑：

    # 阶段一：V 4.7 回答窄问题
    python3 scripts/record_vlm.py --phase vlm

    # 阶段二：Jev 给 V 4.7 的答案复评
    python3 scripts/record_vlm.py --phase jev

结果写在 `samples/vlm_review/`，格式和 `samples/real_ocr/` 一样，是录制数据。

阶段一需要 V 4.7 的服务：

    llama-server -m v47-1b-q4km.gguf --mmproj v47-mmproj.gguf \\
      -c 4096 --port 18500

阶段二需要 Jev 的服务：

    llama-server -m jev-q4.gguf --mmproj jev-mmproj.gguf \\
      -c 8192 --port 18499
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracecell.agent.loop import _align_to_candidate  # noqa: E402
from tracecell.degrade import ResilientEscalator  # noqa: E402
from tracecell.providers import layout  # noqa: E402
from tracecell.providers.jev import LlamaCppJevScorer  # noqa: E402
from tracecell.providers.pages import is_pdf, load_document  # noqa: E402
from tracecell.providers.vlm import OpenAICompatVlm, build_review_question  # noqa: E402
from tracecell.types import BBox  # noqa: E402

OUT_DIR = ROOT / "samples" / "vlm_review"

# 挑了几块有代表性的疑难区域：形近字、大写金额、浅色小字、金额转置。
# 每一块都不是「随便找的图」——都是 OCR 和 Jev 会打架或者都判不了的地方。
DEFAULT_TARGETS = [
    ("invoice-001", "r004"),  # 销售方名称：股份 / 股分
    ("invoice-001", "r014"),  # 价税合计：145,318 / 145,818
    ("expense-001", "r003"),  # 报销人：李思远 / 李思源 / 李思沅
    ("contract-001", "r009"),  # 合同金额大写：伍拾捌万元整 / 伍拾捌万圆整
    ("financial-statement-001", "r009"),  # 所有者权益合计：浅色小字
    ("purchase-contract-001", "r010"),  # 合同总价大写：元整 / 圆整
]


def load_session(path: Path) -> dict:
    """把一个会话文件或一个目录读进来。

    目录的话把所有会话文件合并——样例分两个文件存，区域分散在里面。
    """

    if path.is_dir():
        documents: list[dict] = []
        for file in sorted(path.glob("*.json")):
            documents.extend(json.loads(file.read_text(encoding="utf-8")).get("documents", []))
        return {"documents": documents}
    return json.loads(path.read_text(encoding="utf-8"))


def crop_for(session: dict, doc_id: str, region: dict) -> bytes:
    document = next(d for d in session["documents"] if d["doc_id"] == doc_id)
    raw = (ROOT / document["source"]).read_bytes()
    if is_pdf(raw):
        pages = [page.png for page in load_document(raw, dpi=150).pages]
        index = int(region.get("page", 1)) - 1
        raw = pages[index]
    bbox = region["bbox"]
    return layout.crop(raw, BBox(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="录制 V 4.7 的区域级复核")
    parser.add_argument("--phase", choices=["vlm", "jev"], required=True)
    parser.add_argument("--session", default=str(ROOT / "samples" / "sessions"))
    parser.add_argument("--vlm-url", default="http://127.0.0.1:18500")
    parser.add_argument("--vlm-model", default="MiniCPM-V-4.7-1B")
    parser.add_argument("--jev-url", default="http://127.0.0.1:18499")
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    session = load_session(Path(args.session))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "reviews.json"

    records: dict[str, Any] = {}
    if record_path.exists():
        records = json.loads(record_path.read_text(encoding="utf-8"))

    if args.phase == "vlm":
        # 服务刚起来时第一请求可能 503，包一层重试
        vlm = ResilientEscalator(
            OpenAICompatVlm(base_url=args.vlm_url, model=args.vlm_model), retries=3
        )
        for doc_id, region_id in DEFAULT_TARGETS:
            document = next(d for d in session["documents"] if d["doc_id"] == doc_id)
            region = next(r for r in document["regions"] if r["id"] == region_id)
            scored = [
                {"text": c["text"], "origin": c.get("origin"), "score": c.get("score")}
                for c in region.get("candidates", [])
            ]
            question = build_review_question(
                region.get("text", ""), [c["text"] for c in scored]
            )
            crop = crop_for(session, doc_id, region)
            answer = vlm.review(crop, question)
            # 模型答的可能是片段，对齐回完整候选文本——闭环里就是这么做的
            answer = _align_to_candidate(answer, [c["text"] for c in scored])
            key = f"{doc_id}:{region_id}"
            records[key] = {
                "doc_id": doc_id,
                "region_id": region_id,
                "category": region.get("category"),
                "truth": region.get("truth"),
                "candidates": scored,
                "question": question,
                "vlm_answer": answer,
            }
            print(f"  {key}\n    问题：{question}\n    回答：{answer}")
            record_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        scorer = LlamaCppJevScorer(base_url=args.jev_url)
        for key, record in records.items():
            crop = crop_for(session, record["doc_id"], {"id": record["region_id"], **_region_of(session, record)})
            result = scorer.score(crop, record["vlm_answer"])
            record["jev_score"] = result.score
            record["jev_verdict"] = result.verdict.value
            best = max(record["candidates"], key=lambda c: c.get("score") or 0.0, default=None)
            best_score = (best or {}).get("score")
            best_score = f"{best_score:.4f}" if isinstance(best_score, float) else "无"
            score_text = f"{result.score:.4f}" if result.score is not None else "失败"
            print(f"  {key}  V4.7 答案复评 {score_text}（原最佳候选 {best_score}）")
            record_path.write_text(
                json.dumps(records, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    print(f"  已写入 {record_path}")
    return 0


def _region_of(session: dict, record: dict) -> dict:
    document = next(d for d in session["documents"] if d["doc_id"] == record["doc_id"])
    return next(r for r in document["regions"] if r["id"] == record["region_id"])


if __name__ == "__main__":
    raise SystemExit(main())
