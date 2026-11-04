"""命令行的冒烟测试。

CLI 的参数解析和子命令装配最容易在重构时漏改——构造函数多返回一个值、
参数改名，单元测试都发现不了，只有真的跑一遍才会炸。这里把常用子命令
都跑一遍。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tracecell.cli import main

ROOT = Path(__file__).resolve().parents[1]
SESSIONS = str(ROOT / "samples" / "sessions")
IMAGE = str(ROOT / "samples" / "invoice_001.png")


def run(capsys, *args) -> tuple[int, str]:
    code = main(list(args))
    return code, capsys.readouterr().out


class TestCli:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0

    def test_scenarios_lists_all_of_them(self, capsys):
        code, out = run(capsys, "scenarios")
        assert code == 0
        for name in ("invoice", "financial_statement", "purchase_contract"):
            assert name in out

    def test_demo_runs_every_sample(self, capsys):
        code, out = run(capsys, "demo", "--session", SESSIONS)
        assert code == 0
        assert out.count("→") == 5

    def test_demo_reports_consistency_failures(self, capsys):
        """三处勾稽错都要在输出里出现。"""

        code, out = run(capsys, "demo", "--session", SESSIONS)
        assert code == 0
        assert out.count("勾稽不符") == 3

    @pytest.mark.parametrize("case", ["ocr", "jev", "vlm"])
    def test_standalone_cases_run(self, capsys, case):
        code, out = run(capsys, "case", case, IMAGE, "--doc-id", "invoice-001")
        assert code == 0
        assert "案例：" in out

    def test_calibrate_reports_a_threshold(self, capsys):
        code, out = run(
            capsys, "calibrate", str(ROOT / "samples" / "labelled.json"), "--target-accuracy", "0.99"
        )
        assert code == 0
        assert "阈值" in out
        assert "AUROC" in out

    def test_reconcile_across_scenarios(self, capsys, tmp_path):
        """先把两份账本跑出来再对账，矛盾要被抓到。"""

        ledgers = []
        for doc_id in ("invoice-001", "purchase-contract-001"):
            path = tmp_path / f"{doc_id}.json"
            code = main(
                [
                    "parse",
                    str(ROOT / "samples" / f"{doc_id.replace('-', '_')}.png"),
                    "--doc-id",
                    doc_id,
                    "--session",
                    SESSIONS,
                    "--format",
                    "json",
                    "--out",
                    str(path),
                ]
            )
            assert code == 0
            ledgers.append(str(path))

        code, out = run(capsys, "reconcile", *ledgers)
        assert "矛盾" in out
