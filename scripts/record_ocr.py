"""录制真实模型输出。

这个脚本把 MiniCPM-OCR 1.0 在样例单据上的**真实输出**存下来，供
``samples/real_ocr/`` 引用。这些不是手写的，是模型跑出来的，可以拿去做
"这个链路到底认不认得准"的对照。

复现方式（llama.cpp，纯 CPU 即可）：

    # 模型权重
    git clone https://huggingface.co/openbmb/MiniCPM-OCR-1.0-gguf
    #   MiniCPM-OCR-1_0-Q4_K_M.gguf        主模型
    #   mmproj-model-f16-4x.gguf           4x 视觉编码器

    llama-mtmd-cli \\
      -m MiniCPM-OCR-1_0-Q4_K_M.gguf \\
      --mmproj mmproj-model-f16-4x.gguf \\
      --image samples/invoice_001.png \\
      --prompt "Text Recognition:" \\
      -n 900 -t 2 -c 4096 -b 256 -ub 128

注意任务前缀不能改。MiniCPM-OCR 1.0 的前缀是训练时固定的任务指令，
换成自然语言问句模型会答非所问（实测会输出"请提供需要的信息"）。

用法：

    python3 scripts/record_ocr.py /path/to/llama-output-dir
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "samples" / "real_ocr"
SESSION = ROOT / "samples" / "sessions" / "real_ocr.json"

DOCS = {
    "invoice_001": ("invoice-001", "增值税专用发票"),
    "contract_001": ("contract-001", "技术服务合同"),
    "expense_001": ("expense-001", "员工报销单"),
}

# llama.cpp 的 chat 模板会先吐出几行示例对话，剥掉它们，只留模型的实际回答。
PREAMBLE = re.compile(
    r"^(system|user|assistant)\s*$|^You are a helpful assistant\s*$|^Hello\s*$"
    r"|^Hi there\s*$|^How are you\?\s*$"
)


def clean(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if PREAMBLE.match(stripped):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    source = Path(sys.argv[1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = []
    for filename, (doc_id, label) in DOCS.items():
        path = source / f"{filename}.txt"
        if not path.exists():
            print(f"跳过 {filename}：找不到 {path}")
            continue
        text = clean(path.read_text(encoding="utf-8", errors="ignore"))
        (OUT_DIR / f"{filename}.md").write_text(text + "\n", encoding="utf-8")
        documents.append(
            {
                "doc_id": doc_id,
                "label": label,
                "source": f"samples/{filename}.png",
                "markdown": text,
            }
        )
        print(f"{filename}: {len(text)} 字")

    SESSION.write_text(
        json.dumps({"documents": documents}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"已写出 {SESSION.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
