"""拿官方样例验证这条链路。

Jev 官方仓库里放了三组样例，每组给出：原图块、参考渲染图，以及官方自己跑
出来的概率（`observed_prob_yes`）。样例文件收在 `references/jev_official/`，
来源：

    https://github.com/OpenBMB/MiniCPM-OCR/tree/main/scripts/minicpm-ocr-jev/examples

Apache-2.0。

**第一项检查是打分链路本身**：用官方给的参考渲染图打分，和官方自己跑出来的
概率对。差得远就说明请求格式、图像顺序、概率取值有问题。

**第二项是自渲染真值**：真值的渲染图我们自己渲染。官方渲染器用 KaTeX 渲染
公式、用 HTML 表格渲染表格，这里没有实现，所以公式和表格两类会有明显差距。
这一项能差多少，取决于渲染器缺了什么。

**第三项是排序**：官方案例 03 自己写了——表格的正确转写只有 0.04 分，绝对分
不可用，要比的是候选之间的排序。但公式那个案例官方自己的两个候选就几乎
同分（0.968 对 0.966），本来就不该要求排开，所以只在官方明确拉开差距时才
判定。

需要一个跑着的 Jev 服务：

    llama-server -m jev-q4.gguf --mmproj jev-mmproj.gguf -c 8192 --port 18499

    python3 scripts/verify_against_official.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracecell.providers.jev import LlamaCppJevScorer  # noqa: E402
from tracecell.render import render_text  # noqa: E402

CASES_DIR = ROOT / "references" / "jev_official"

# 打分链路这一项的容差。官方样本之间差 0.005 以内。
CHAIN_TOLERANCE = 0.02
# 官方自己把两个候选拉开多少，才值得要求本机也排开。
RANKING_GAP = 0.1


def main() -> int:
    parser = argparse.ArgumentParser(description="用官方样例验证 Jev 链路")
    parser.add_argument("--url", default="http://127.0.0.1:18499", help="Jev 服务地址")
    args = parser.parse_args()

    scorer = LlamaCppJevScorer(base_url=args.url)
    failures = 0
    rows = []

    for case_dir in sorted(p for p in CASES_DIR.iterdir() if p.is_dir()):
        case = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
        block = (case_dir / "block.png").read_bytes()
        reference = (case_dir / "reference_render.png").read_bytes()
        official = case["observed_prob_yes"]

        ocr_candidate = next(c for c in case["candidates"] if c["role"] == "ocr_output")
        truth_candidate = next(c for c in case["candidates"] if c["role"] == "ground_truth")

        # 官方自己的渲染图，配上官方给的概率——这是打分链路的对照
        with_official_render = scorer.score_with_render(block, ocr_candidate["text"], reference)
        # 真值的渲染图我们自己渲染
        with_own_render = scorer.score_with_render(
            block, truth_candidate["text"], render_text(truth_candidate["text"])
        )

        if with_official_render.score is None or with_own_render.score is None:
            print(f"  {case['case_id']}：评分失败")
            failures += 1
            continue

        official_ocr = float(official["ocr_output"])
        official_truth = float(official["ground_truth"])
        chain_delta = abs(with_official_render.score - official_ocr)
        chain_ok = chain_delta <= CHAIN_TOLERANCE

        gap = official_truth - official_ocr
        if abs(gap) >= RANKING_GAP:
            ranking_ok = with_own_render.score > with_official_render.score
            ranking_text = "✓ 一致" if ranking_ok else "✗ 不一致"
        else:
            ranking_ok = True
            ranking_text = "— 官方两者本身几乎同分，不作要求"

        print(f"\n  {case['case_id']}（{case['category']}）")
        print(
            f"    打分链路：官方渲染图 → 官方 {official_ocr:.5f}，本机 "
            f"{with_official_render.score:.5f}，差 {chain_delta:.5f}"
            f"　{'✓' if chain_ok else '✗'}"
        )
        print(
            f"    自渲染真值：官方 {official_truth:.5f}，本机 {with_own_render.score:.5f}，"
            f"差 {abs(with_own_render.score - official_truth):.5f}"
        )
        print(f"    排序：{ranking_text}")

        if not chain_ok or not ranking_ok:
            failures += 1
        rows.append(
            {
                "case_id": case["case_id"],
                "category": case["category"],
                "official_ocr_output": official_ocr,
                "local_ocr_output": round(with_official_render.score, 5),
                "chain_delta": round(chain_delta, 5),
                "official_ground_truth": official_truth,
                "local_ground_truth": round(with_own_render.score, 5),
                "ranking_ok": ranking_ok,
            }
        )

    out = CASES_DIR / "verification.json"
    out.write_text(
        json.dumps({"cases": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n  结果已写入 {out.relative_to(ROOT)}")

    if failures:
        print(f"  {failures} 组样例没过。")
        return 1
    print("  全部样例通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
