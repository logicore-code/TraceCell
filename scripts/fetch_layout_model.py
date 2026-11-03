"""下载 PP-DocLayoutV3 版面检测模型。

TraceCell 自带的投影法切分器不依赖任何模型，复杂版面会偏。真要处理扫描件，
换成官方的 PP-DocLayoutV3——这也是 MiniCPM-OCR 官方流水线用的那一个。

模型是 ONNX 版（约 136MB），跑在 onnxruntime 上，不需要 PyTorch 也不需要
PaddlePaddle。

    python3 scripts/fetch_layout_model.py
    export TRACECELL_LAYOUT_MODEL=$(python3 scripts/fetch_layout_model.py --path-only)

模型文件不进仓库（太大），用环境变量指路。没设这个变量、或者 onnxruntime
没装，TraceCell 就用自带的投影法，流程照跑不误。
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

MODEL_REPO = "Bei0001/PP-DocLayoutV3-ONNX"
ENDPOINTS = [
    f"https://hf-mirror.com/{MODEL_REPO}/resolve/main/{name}"
    for name in (
        "PP-DocLayoutV3.onnx",
        "PP-DocLayoutV3.onnx.data",
        "config.json",
        "preprocessor_config.json",
        "inference.yml",
    )
]
FALLBACK = f"https://huggingface.co/{MODEL_REPO}/resolve/main"


def download(urls: list[str], target: Path) -> bool:
    for url in urls:
        try:
            print(f"  下载 {target.name} …", flush=True)
            with urllib.request.urlopen(url, timeout=300) as response:
                target.write_bytes(response.read())
            return True
        except Exception as exc:  # noqa: BLE001 - 换下一个镜像继续
            print(f"    {url} 失败：{exc}", file=sys.stderr)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 PP-DocLayoutV3 版面检测模型")
    parser.add_argument(
        "--dir",
        default=os.environ.get("TRACECELL_LAYOUT_DIR", "models/PP-DocLayoutV3"),
        help="模型存放目录",
    )
    parser.add_argument("--path-only", action="store_true", help="只打印目录路径")
    args = parser.parse_args()

    target_dir = Path(args.dir).expanduser()
    if args.path_only:
        print(target_dir.resolve())
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    for url in ENDPOINTS:
        name = url.rsplit("/", 1)[-1]
        target = target_dir / name
        if target.exists() and target.stat().st_size > 0:
            print(f"  已有 {name}，跳过")
            continue
        if not download([url, f"{FALLBACK}/{name}"], target):
            print(f"  下载 {name} 失败", file=sys.stderr)
            return 1

    print(f"\n模型已就绪：{target_dir.resolve()}")
    print(f"  export TRACECELL_LAYOUT_MODEL={target_dir.resolve()}")
    print("  pip install onnxruntime")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
