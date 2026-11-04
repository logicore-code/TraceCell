"""用真实的 MiniCPM-OCR Jev 给样例区域打分。

样例里的 Jev 分数一开始是手工设的——为了让演示里那几个难点按预期触发。
但手工的分数不是证据。这一脚本用真模型把每个候选读数重新打一遍，结果写进
``samples/jev_scores.json``，那是**录制数据**，和 ``samples/real_ocr/`` 一个性质。

需要先起 Jev 服务（llama.cpp 的 decision model）：

    llama-server \\
      -m MiniCPM-OCR-Jev-Q4_K_M.gguf \\
      --mmproj mmproj-model-f16.gguf \\
      -c 8192 --port 18499

然后：

    python3 scripts/record_jev.py --session samples/sessions/demo.json
    python3 scripts/record_jev.py --session samples/sessions/extra.json

图像顺序是有讲究的：原图在前、候选文本的渲染图在后，反过来问的就不是同一个
问题了，而且不会报错，只会给一个错的数。

CPU 上每个候选二十来秒，服务默认四个槽，用并发跑。
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracecell.providers import layout  # noqa: E402
from tracecell.providers.pages import is_pdf, load_document  # noqa: E402
from tracecell.render import render_text  # noqa: E402
from tracecell.types import BBox  # noqa: E402


def score_pair(original_png: bytes, text: str, url: str) -> float | None:
    """给一组（原图, 候选文本）打分。返回与原图一致的��率，失败返回 None。"""

    render_png = render_text(text)
    body = {
        "state": text,
        "files": [
            "data:image/png;base64," + base64.b64encode(original_png).decode(),
            "data:image/png;base64," + base64.b64encode(render_png).decode(),
        ],
        "questions": {"consistency": {"type": "choice", "criteria": {"0": None, "1": None}}},
    }
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            payload = json.load(response)
        probabilities = payload["answers"]["consistency"]["probabilities"]
        return float(probabilities["1"])
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, ValueError) as exc:
        print(f"    评分失败：{exc}", file=sys.stderr)
        return None


def _flush(results: dict[str, Any], out: str) -> None:
    """每条结果都落盘。

    一次录制要跑几十分钟，中途挂掉不该全丢。
    """

    target = Path(out)
    existing: dict[str, Any] = {}
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
    # 按区域合并，不能按文档整个替换——续跑的时候只处理了部分区域，
    # 直接替换会把之前录好的区域冲掉。
    for doc_id, regions in results.items():
        for region_id, scores in regions.items():
            existing.setdefault(doc_id, {}).setdefault(region_id, {}).update(scores)
    target.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="用真实的 Jev 给样例区域打分")
    parser.add_argument("--session", required=True, help="会话 JSON 路径")
    parser.add_argument("--url", default="http://127.0.0.1:18499", help="Jev 服务地址")
    parser.add_argument("--out", default=str(ROOT / "samples" / "jev_scores.json"))
    parser.add_argument("--jobs", type=int, default=4, help="并发数，服务默认四个槽")
    args = parser.parse_args()

    session_path = Path(args.session)
    session = json.loads(session_path.read_text(encoding="utf-8"))

    # 已经打过分的跳过。一轮录制要跑几十分钟，中间断了得能接着来。
    done: dict[str, dict[str, dict[str, float]]] = {}
    out_path = Path(args.out)
    if out_path.exists():
        done = json.loads(out_path.read_text(encoding="utf-8"))

    tasks: list[tuple[str, str, str, bytes, str]] = []
    skipped = 0
    for document in session["documents"]:
        source = ROOT / document["source"]
        if not source.exists():
            print(f"  跳过 {document['doc_id']}：找不到 {source}")
            continue
        raw = source.read_bytes()
        # PDF 的区域坐标是相对那一页的，得先把页渲染出来再裁。
        # 直接拿 PDF 字节去裁会裁出空白，出来的分数是废的。
        pages: list[bytes] = []
        if is_pdf(raw):
            pages = [page.png for page in load_document(raw, dpi=150).pages]
        for region in document["regions"]:
            bbox = region["bbox"]
            image = raw
            if pages:
                index = int(region.get("page", 1)) - 1
                if not 0 <= index < len(pages):
                    continue
                image = pages[index]
            crop = layout.crop(
                image, BBox(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
            )
            for candidate in region.get("candidates", []):
                text = candidate["text"]
                if text in done.get(document["doc_id"], {}).get(region["id"], {}):
                    skipped += 1
                    continue
                tasks.append(
                    (
                        document["doc_id"],
                        region["id"],
                        text,
                        crop,
                        candidate.get("origin", "primary"),
                    )
                )
    if skipped:
        print(f"  已有 {skipped} 个候选打过分，跳过")

    print(f"  待打分的候选 {len(tasks)} 个，并发 {args.jobs}")
    started = time.time()
    results: dict[str, dict[str, dict[str, float]]] = {}

    def run_one(task):
        doc_id, region_id, text, crop, origin = task
        value = score_pair(crop, text, args.url)
        return doc_id, region_id, text, origin, value

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for index, (doc_id, region_id, text, origin, value) in enumerate(
            pool.map(run_one, tasks), start=1
        ):
            if value is None:
                continue
            results.setdefault(doc_id, {}).setdefault(region_id, {})[text] = round(value, 6)
            flag = "✓" if value > 0.5 else " "
            print(f"    [{index}/{len(tasks)}] {flag} {doc_id}:{region_id} {origin:12s} {value:.4f}")
            _flush(results, args.out)

    elapsed = time.time() - started
    print(f"  用时 {elapsed / 60:.1f} 分钟")

    _flush(results, args.out)
    print(f"  已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
