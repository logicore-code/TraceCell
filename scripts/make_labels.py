"""从录制会话导出标定样本。

标定需要"分数 + 人工判断"。样例会话里每个区域都带一个 ``text`` 字段，
那是生成样例时写入的**构造真值**（也就是这块区域上正确的读法）。候选是否
正确，只要跟它比对就知道。

这个判定规则不看分数，因此不会循环论证：在争议区域里，得分高的那个候选
恰恰可能是错的，而真值比对会把它标成错误。

真实业务里这一列要由人来标，方法见 ``docs/calibration.md``。

    python3 scripts/make_labels.py            # 生成 samples/labelled.json
    tracecell calibrate samples/labelled.json  # 标定阈值
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tracecell.textnorm import normalize  # noqa: E402

SESSION = ROOT / "samples" / "sessions" / "demo.json"
OUTPUT = ROOT / "samples" / "labelled.json"


def main() -> int:
    session = json.loads(SESSION.read_text(encoding="utf-8"))
    samples: list[dict] = []

    for document in session.get("documents", []):
        doc_id = document.get("doc_id", "?")
        for region in document.get("regions", []):
            for candidate in region.get("candidates", []):
                score = candidate.get("score")
                if score is None:
                    continue
                truth = normalize(region.get("text", ""))
                correct = normalize(candidate.get("text", "")) == truth
                samples.append(
                    {
                        "score": round(float(score), 4),
                        "correct": correct,
                        "category": region.get("category", "text"),
                        "note": f"{doc_id}:{region.get('id')}:{candidate.get('origin')}",
                    }
                )

    OUTPUT.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    correct = sum(1 for s in samples if s["correct"])
    print(f"已写出 {OUTPUT.relative_to(ROOT)}：{len(samples)} 条样本（正确 {correct} / 错误 {len(samples) - correct}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
