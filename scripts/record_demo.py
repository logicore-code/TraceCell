"""录一段 35 秒左右的演示视频。

静态截图讲不清"点一下能看到证据"这件事——那段交互是整个工具的关键，视频
能讲明白。

    python3 scripts/record_demo.py

产出 `docs/demo.mp4`（先录 webm，再用 ffmpeg 转 mp4）。

脚本自己起 Web 服务、用 Playwright 走一遍主要视图，每一步叠一句字幕，
免得看的人不知道现在在看什么。字幕用界面上的中文，跟界面是一致的。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

VIDEO_DIR = ROOT / "docs" / "videos"

# (字幕, 停多久)
STEPS = [
    ("TraceCell · 溯格 —— 把 OCR 结果变成可追责的证据账本", 3.0),
    ("① 字段台账：每个字段带原图坐标、置信度、判定状态", 4.0),
    ("② 点一下字段，看它是从原图哪一格读出来的", 4.0),
    ("③ 复核清单：致命 / 高危 / 一般分级，每条都带处理建议", 4.0),
    ("④ 跨文档对账：合同总价和发票金额对不上，只有跨字段才能发现", 4.0),
    ("⑤ 路由结论：免检 / 抽检 / 全检，按风险决定人工看多少", 4.0),
]

CAPTION_JS = """
(text) => {
  const old = document.getElementById('__demo_caption');
  if (old) old.remove();
  const box = document.createElement('div');
  box.id = '__demo_caption';
  box.textContent = text;
  Object.assign(box.style, {
    position: 'fixed', left: '0', right: '0', top: '0', zIndex: '99999',
    padding: '14px 24px', background: 'rgba(16,19,26,0.92)', color: '#f4f6fa',
    font: '600 19px/1.4 system-ui, "Noto Sans CJK SC", sans-serif',
    letterSpacing: '0.02em', boxShadow: '0 6px 18px rgba(0,0,0,0.28)',
    pointerEvents: 'none',
  });
  document.body.appendChild(box);
}
"""


def free_port() -> int:
    """挑一个没人占的端口。"""

    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(port: int) -> subprocess.Popen:
    import os

    environ = dict(os.environ)
    environ["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.Popen(
        [sys.executable, "-m", "tracecell.cli", "serve", "--port", str(port)],
        cwd=str(ROOT),
        env=environ,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> int:
    if shutil.which("ffmpeg") is None:
        print("需要 ffmpeg 才能转 mp4", file=sys.stderr)
        return 1

    from playwright.sync_api import sync_playwright

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for old in VIDEO_DIR.glob("*"):
        old.unlink()

    port = free_port()
    server = start_server(port)
    try:
        time.sleep(6)
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox", "--force-device-scale-factor=1"])
            context = browser.new_context(
                viewport={"width": 1440, "height": 900},
                record_video_dir=str(VIDEO_DIR),
                record_video_size={"width": 1440, "height": 900},
            )
            page = context.new_page()
            page.goto(f"http://127.0.0.1:{port}/", wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(2500)

            def caption(text: str, hold: float = 0.0) -> None:
                page.evaluate(CAPTION_JS, text)
                if hold:
                    page.wait_for_timeout(int(hold * 1000))

            caption(STEPS[0][0], 2.5)

            # 选一份单据，等台账加载出来
            options = page.eval_on_selector_all(
                "#docSelect option", "els => els.map(e => e.value)"
            )
            if options:
                page.select_option("#docSelect", options[0])
                page.wait_for_timeout(2000)

            caption(STEPS[1][0], 3.0)

            # 点一个字段，看证据抽屉。字幕先亮，动作后做——顺序反了
            # 字幕就和画面对不上了。
            caption(STEPS[2][0], 0.9)
            rows = page.query_selector_all("#fieldBody tr[data-index]")
            if rows:
                rows[min(3, len(rows) - 1)].click()
                page.wait_for_timeout(2800)
                page.click("#drawerClose")
                page.wait_for_timeout(500)

            # 问题清单
            caption(STEPS[3][0], 0.9)
            page.eval_on_selector(
                "#findingsBody", "el => el.closest('.card').scrollIntoView({block: 'center'})"
            )
            page.wait_for_timeout(2400)

            # 跨文档对账。这个接口每次要算 6 秒左右，等够
            caption(STEPS[4][0], 0.9)
            page.click('[data-view="reconcile"]')
            page.wait_for_timeout(9000)

            # 回到台账收尾
            caption(STEPS[5][0], 0.9)
            page.click('[data-view="ledger"]')
            page.wait_for_timeout(3000)

            page.evaluate("() => { const b = document.getElementById('__demo_caption'); if (b) b.remove(); }")
            page.wait_for_timeout(1200)
            context.close()
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    webm = next(VIDEO_DIR.glob("*.webm"), None)
    if webm is None:
        print("没录到视频", file=sys.stderr)
        return 1
    mp4 = ROOT / "docs" / "demo.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(webm),
            "-vf", "scale=1440:900:flags=lanczos,fps=30",
            "-c:v", "libx264", "-preset", "slow", "-crf", "21",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(mp4),
        ],
        check=True,
    )
    webm.unlink()
    size = mp4.stat().st_size / 1024 / 1024
    print(f"  已写入 {mp4.relative_to(ROOT)}（{size:.1f}MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
