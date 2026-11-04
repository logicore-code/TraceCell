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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracecell.providers import layout  # noqa: E402
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


def main() -> int:
    parser = argparse.ArgumentParser(description="用真实的 Jev 给样例区域打分")
    parser.add_argument("--session", required=True, help="会话 JSON 路径")
    parser.add_argument("--url", default="http://127.0.0.1:18499", help="Jev 服务地址")
    parser.add_argument("--out", default=str(ROOT / "samples" / "jev_scores.json"))
    parser.add_argument("--jobs", type=int, default=4, help="并发数，服务默认四个槽")
    args = parser.parse_args()

    session_path = Path(args.session)
    session = json.loads(session_path.read_text(encoding="utf-8"))

    tasks: list[tuple[str, str, str, bytes, str]] = []
    for document in session["documents"]:
        source = ROOT / document["source"]
        if not source.exists():
            print(f"  跳过 {document['doc_id']}：找不到 {source}")
            continue
        image = source.read_bytes()
        for region in document["regions"]:
            bbox = region["bbox"]
            crop = layout.crop(
                image, BBox(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
            )
            for candidate in region.get("candidates", []):
                tasks.append(
                    (
                        document["doc_id"],
                        region["id"],
                        candidate["text"],
                        crop,
                        candidate.get("origin", "primary"),
                    )
                )

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

    elapsed = time.time() - started
    print(f"  用时 {elapsed / 60:.1f} 分钟")

    out_path = Path(args.out)
    existing: dict[str, Any] = {}
    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    existing.update(results)
    out_path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"  已写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
