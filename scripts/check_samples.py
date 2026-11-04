"""检查样例数据是不是自洽。

CI 里跑的「重新生成一遍再 `git diff`」这条检查有个前提：生成样例的机器上
装的中文字体和当初生成仓库里那批图时**完全一致**。字体版本不一样，画出来的
像素就不一样，sha256 也就跟着变——那条检查会在完全正常的仓库上失败。

所以分两层：

* **字节一致**（本地做）：在生成样例的那台机器上重新生成，`git diff` 应该是
  空的。见 CONTRIBUTING。
* **语义一致 + 内部自洽**（CI 做）：换台机器也该成立的部分。

本脚本查两类：

1. **内部自洽**——仓库里这批数据自己跟自己对不对得上：源文件的 sha256 与记录
   一致、候选分数与录制数据一致、真值非空、字段引用的区域存在、变体标记指向
   真实存在的文档。
2. **语义一致**（`--regenerated <目录>`）——拿重新生成的会话和仓库里的比，
   只比不依赖字体的部分：区域数量、文本、真值、候选文本与分数、字段。
   坐标和 sha256 是图像派生的，字体一变就会变，不在这里面比。

    python3 scripts/check_samples.py --regenerated /tmp/regen
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
SESSIONS = SAMPLES / "sessions"


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def check_internal(problems: list[str]) -> None:
    """仓库里的数据自己跟自己对不对得上。"""

    scores = load(SAMPLES / "jev_scores.json")
    sessions = sorted(SESSIONS.glob("*.json"))
    if not sessions:
        problems.append("samples/sessions/ 下没有会话文件")

    doc_ids: set[str] = set()
    for path in sessions:
        session = load(path)
        for document in session.get("documents", []):
            doc_id = document["doc_id"]
            doc_ids.add(doc_id)

    for path in sessions:
        session = load(path)
        for document in session.get("documents", []):
            doc_id = document["doc_id"]
            where = f"{path.name}:{doc_id}"

            source = ROOT / document.get("source", "")
            if not source.exists():
                problems.append(f"{where}：源文件不存在 {source}")
            elif document.get("sha256"):
                # 真实 OCR 记录那种只有文本没有区域的会话不带哈希，跳过
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if digest != document["sha256"]:
                    problems.append(f"{where}：sha256 对不上（文件被改过？）")

            variant = document.get("variant_of")
            if variant and variant not in doc_ids:
                problems.append(f"{where}：variant_of 指向不存在的文档 {variant}")

            for region in document.get("regions", []):
                if not region.get("truth"):
                    problems.append(f"{where}:{region['id']}：没有真值标注")
                recorded = scores.get(doc_id, {}).get(region["id"], {})
                for candidate in region.get("candidates", []):
                    text = candidate.get("text", "")
                    if text not in recorded:
                        problems.append(
                            f"{where}:{region['id']}：候选没有对应的录制分数——"
                            f"{text[:24]}"
                        )
                    elif abs(recorded[text] - float(candidate.get("score") or 0)) > 1e-6:
                        problems.append(
                            f"{where}:{region['id']}：候选分数与录制数据不一致"
                        )

            known = {r["id"] for r in document.get("regions", [])}
            for field in document.get("fields", []):
                for rid in field.get("region_ids", []):
                    if rid not in known:
                        problems.append(f"{where}：字段 {field.get('name')} 引用了不存在的区域 {rid}")


# 这些字段是图像派生的：字体版本一变，坐标和文件哈希就会变。
IMAGE_DERIVED = {"_pixels", "sha256", "bbox"}


def semantic_view(session: dict[str, Any]) -> Any:
    """把会话里不依赖字体的部分抽出来。"""

    out = []
    for document in session.get("documents", []):
        regions = []
        for region in document.get("regions", []):
            regions.append(
                {
                    "id": region.get("id"),
                    "page": region.get("page", 1),
                    "category": region.get("category"),
                    "text": region.get("text"),
                    "truth": region.get("truth"),
                    "candidates": [
                        {
                            "text": c.get("text"),
                            "origin": c.get("origin"),
                            "score": c.get("score"),
                            "verdict": c.get("verdict"),
                        }
                        for c in region.get("candidates", [])
                    ],
                }
            )
        out.append(
            {
                "doc_id": document.get("doc_id"),
                "variant_of": document.get("variant_of"),
                "schema": document.get("schema"),
                "label": document.get("label"),
                "fields": [
                    {
                        "name": f.get("name"),
                        "value": f.get("value"),
                        "region_ids": f.get("region_ids"),
                    }
                    for f in document.get("fields", [])
                ],
                "regions": regions,
            }
        )
    return out


def check_regenerated(regenerated: Path, problems: list[str]) -> None:
    """重新生成的会话和仓库里的比，语义层面。"""

    for name in ("demo.json", "extra.json"):
        committed = SESSIONS / name
        fresh = regenerated / "sessions" / name
        if not committed.exists() or not fresh.exists():
            problems.append(f"找不到会话文件 {name}")
            continue
        left = semantic_view(load(committed))
        right = semantic_view(load(fresh))
        if left != right:
            problems.append(f"{name}：重新生成的内容和仓库里的不一致")
            for a, b in zip(left, right, strict=False):
                if a != b:
                    for key in a:
                        if a.get(key) != b.get(key):
                            problems.append(f"  {name}:{a.get('doc_id')} 的 {key} 对不上")


def main() -> int:
    parser = argparse.ArgumentParser(description="检查样例数据是否自洽")
    parser.add_argument("--regenerated", help="重新生成的样例目录，用来比对语义")
    args = parser.parse_args()

    problems: list[str] = []
    check_internal(problems)
    if args.regenerated:
        check_regenerated(Path(args.regenerated), problems)

    if problems:
        print(f"发现 {len(problems)} 个问题：")
        for problem in problems:
            print(f"  ✗ {problem}")
        return 1
    print("样例数据自洽。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
