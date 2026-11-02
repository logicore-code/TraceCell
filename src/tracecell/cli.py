"""命令行入口。

    tracecell demo                       # 跑内置样例，看完整链路
    tracecell parse 票据.png --schema invoice
    tracecell serve                      # 起 Web 界面
    tracecell mcp                        # 作为 MCP 服务运行
    tracecell calibrate samples.json      # 标定阈值
    tracecell reconcile a.json b.json     # 跨文档对账
    tracecell doctor                     # 检查模型服务是否就绪
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .agent.loop import VerificationLoop
from .calibration import calibrate, load_samples, per_category_stats
from .extractors import build_extractor, schema_by_name
from .ledger import export_csv, export_markdown, ledger_summary
from .policy import BudgetPolicy, RiskPolicy
from .providers.jev import HttpJevScorer
from .providers.ocr import OpenAICompatOcr
from .providers.replay import ReplayEscalator, ReplayParser, ReplayScorer, ReplaySession
from .providers.vlm import OpenAICompatVlm
from .reconcile import merge_ledgers, reconcile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION = ROOT / "samples" / "sessions" / "demo.json"


def _read_bytes(path: str) -> bytes:
    data = Path(path).read_bytes()
    if not data:
        raise SystemExit(f"文件是空的：{path}")
    return data


def _build_components(args, doc_id: str):
    """按 provider 组装三个模型的客户端。"""

    if args.provider == "minicpm":
        parser = OpenAICompatOcr(base_url=args.ocr_url, model=args.ocr_model)
        scorer = HttpJevScorer(base_url=args.jev_url)
        escalator = OpenAICompatVlm(base_url=args.vlm_url, model=args.vlm_model)
    else:
        session = ReplaySession.load(args.session)
        if doc_id not in session.doc_ids():
            raise SystemExit(
                f"回放会话里没有文档 {doc_id}，可选：{', '.join(session.doc_ids())}"
            )
        parser = ReplayParser(session, doc_id)
        scorer = ReplayScorer(session, doc_id)
        escalator = ReplayEscalator(session, doc_id)
    return parser, scorer, escalator


def _build_loop(args, doc_id: str, schema: str) -> VerificationLoop:
    parser, scorer, escalator = _build_components(args, doc_id)
    return VerificationLoop(
        parser=parser,
        scorer=scorer,
        escalator=escalator,
        policy=BudgetPolicy(),
        risk_policy=RiskPolicy(),
        extractor=build_extractor(schema),
        schema=schema,
    )


def cmd_parse(args) -> int:
    image = _read_bytes(args.image)
    doc_id = args.doc_id or Path(args.image).stem
    loop = _build_loop(args, doc_id, args.schema)
    report = loop.run(image, doc_id, args.image)
    ledger = report.ledger

    if args.format == "json":
        print(ledger.to_json())
    elif args.format == "csv":
        print(export_csv(ledger))
    else:
        print(export_markdown(ledger))
        print()
        summary = ledger_summary(ledger)
        print(
            f"路由结论：{summary['verdict']}｜字段 {summary['field_count']} 个"
            f"｜待人工 {summary['review_queue_size']} 格"
            f"｜核验成本 {summary['budget']['total_cost']:.2f} rcu"
            f"｜耗时 {report.elapsed:.2f}s"
        )

    if args.out:
        Path(args.out).write_text(ledger.to_json(), encoding="utf-8")
        print(f"账本已保存：{args.out}", file=sys.stderr)
    return 0


def cmd_demo(args) -> int:
    session = ReplaySession.load(args.session)
    for doc_id in session.doc_ids():
        document = session.document(doc_id) or {}
        schema = document.get("schema", "invoice")
        loop = _build_loop(args, doc_id, schema)
        report = loop.run(b"", doc_id, document.get("source", ""))
        ledger = report.ledger
        summary = ledger_summary(ledger)
        print("=" * 72)
        print(f"文档 {doc_id}（{document.get('label', '')}）→ {summary['verdict']}")
        print("=" * 72)
        print(export_markdown(ledger))
        print()
    return 0


def cmd_case(args) -> int:
    """独立应用案例：一模型一案例，各自可独立运行。"""

    from .cases import run_jev_case, run_ocr_case, run_vlm_case

    image = _read_bytes(args.image)
    doc_id = args.doc_id or Path(args.image).stem
    parser, scorer, escalator = _build_components(args, doc_id)

    if args.case == "ocr":
        result = run_ocr_case(parser, image)
    elif args.case == "jev":
        result = run_jev_case(scorer, parser, image, threshold=args.threshold)
    else:
        result = run_vlm_case(escalator, image, question=args.question)

    if args.format == "json":
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"案例：{result.case}    使用模型：{'、'.join(result.models)}")
        print("-" * 72)
        for key, value in result.summary.items():
            print(f"  {key}: {value}")
        if result.regions:
            print()
            for region in result.regions:
                score = region.get("score")
                score_text = f"{score:.3f}" if isinstance(score, float) else str(region.get("verdict"))
                print(f"  [{region['id']}] {score_text}  {region['text'][:48]}")
        if result.markdown:
            print()
            print(result.markdown)
        if result.notes:
            print()
            for note in result.notes:
                print(f"  · {note}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .web.app import create_app

    app = create_app(provider=args.provider)
    print(f"TraceCell 已启动：http://{args.host}:{args.port}  （provider={args.provider}）")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_mcp(args) -> int:
    from .mcp.server import main as mcp_main

    return mcp_main(["--session", args.session])


def cmd_calibrate(args) -> int:
    samples = load_samples(args.samples)
    result = calibrate(samples, target_accuracy=args.target_accuracy)
    print(f"阈值 {result.threshold:.3f}")
    print(f"免检率 {result.accept_rate:.2%}（{result.accepted}/{result.total}）")
    print(f"自动通过准确率 {result.accept_accuracy:.2%}，漏放 {result.false_accepts} 处")
    print(f"AUROC {result.auroc:.4f}")
    if result.note:
        print(f"提示：{result.note}")
    stats = per_category_stats(samples)
    if stats:
        print("\n分类别统计：")
        for category, row in sorted(stats.items()):
            print(
                f"  {category:8s} 样本 {row['count']:3d}  平均分 {row['mean_score']:.3f}"
                f"  AUROC {row['auroc']:.4f}"
            )
    return 0


def cmd_reconcile(args) -> int:
    from .types import DocumentLedger

    ledgers = [DocumentLedger.from_json(Path(path).read_text(encoding="utf-8")) for path in args.ledgers]
    result = reconcile(ledgers)
    print(f"比对字段 {result.compared_fields} 个，一致 {result.matched} 个，矛盾 {len(result.mismatches)} 处")
    for mismatch in result.mismatches:
        print(f"\n[{mismatch.kind}] {mismatch.field_name}：{mismatch.note}")
        for value in mismatch.values:
            print(
                f"   {value['doc_id']}: {value['value']}（归一化 {value['normalized']}，"
                f"置信度 {value['confidence']}）"
            )
    if args.out:
        merged = merge_ledgers(ledgers)
        Path(args.out).write_text(merged.to_json(), encoding="utf-8")
        print(f"\n合并账本已保存：{args.out}", file=sys.stderr)
    return 1 if result.mismatches else 0


def cmd_doctor(args) -> int:
    """检查三个模型服务是否就绪。"""

    ok = True

    print("MiniCPM-OCR 1.0")
    try:
        response = __import__("requests").get(f"{args.ocr_url}/v1/models", timeout=8)
        print(f"  ✓ {args.ocr_url} 响应 {response.status_code}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  ✗ {args.ocr_url} 连不上：{exc}")

    print("MiniCPM-OCR Jev")
    try:
        health = HttpJevScorer(base_url=args.jev_url).health()
        print(f"  ✓ {args.jev_url} status={health.get('status')}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  ✗ {args.jev_url} 连不上：{exc}")

    print("MiniCPM-V 4.7")
    try:
        response = __import__("requests").get(f"{args.vlm_url}/v1/models", timeout=8)
        print(f"  ✓ {args.vlm_url} 响应 {response.status_code}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  ✗ {args.vlm_url} 连不上：{exc}")

    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tracecell", description="TraceCell：可追责的文档解析 Agent")
    parser.add_argument("--version", action="version", version=f"tracecell {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_provider_args(sp) -> None:
        sp.add_argument("--provider", choices=["replay", "minicpm"], default="replay")
        sp.add_argument("--session", default=str(DEFAULT_SESSION), help="回放会话路径")
        sp.add_argument("--ocr-url", default="http://127.0.0.1:8000")
        sp.add_argument("--ocr-model", default="openbmb/MiniCPM-OCR-1.0")
        sp.add_argument("--jev-url", default="http://127.0.0.1:8030")
        sp.add_argument("--vlm-url", default="http://127.0.0.1:8001")
        sp.add_argument("--vlm-model", default="openbmb/MiniCPM-V-4.7-35B-A3B")

    p_parse = sub.add_parser("parse", help="解析并核验一份文档")
    p_parse.add_argument("image")
    p_parse.add_argument("--doc-id", default=None)
    p_parse.add_argument("--schema", default="invoice", choices=["invoice", "contract", "expense"])
    p_parse.add_argument("--format", default="markdown", choices=["markdown", "json", "csv"])
    p_parse.add_argument("--out", default=None, help="把账本 JSON 存到这里")
    add_provider_args(p_parse)
    p_parse.set_defaults(func=cmd_parse)

    p_demo = sub.add_parser("demo", help="跑内置样例")
    add_provider_args(p_demo)
    p_demo.set_defaults(func=cmd_demo)

    p_case = sub.add_parser("case", help="跑单个模型的独立应用案例")
    p_case.add_argument("case", choices=["ocr", "jev", "vlm"], help="案例对应的模型")
    p_case.add_argument("image")
    p_case.add_argument("--doc-id", default=None)
    p_case.add_argument("--threshold", type=float, default=0.9, help="jev 案例的放行阈值")
    p_case.add_argument("--question", default="", help="vlm 案例要问的问题")
    p_case.add_argument("--format", default="text", choices=["text", "json"])
    add_provider_args(p_case)
    p_case.set_defaults(func=cmd_case)

    p_serve = sub.add_parser("serve", help="启动 Web 界面")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    add_provider_args(p_serve)
    p_serve.set_defaults(func=cmd_serve)

    p_mcp = sub.add_parser("mcp", help="以 MCP 服务方式运行")
    p_mcp.add_argument("--session", default=str(DEFAULT_SESSION))
    p_mcp.set_defaults(func=cmd_mcp)

    p_cal = sub.add_parser("calibrate", help="标定自动放行阈值")
    p_cal.add_argument("samples")
    p_cal.add_argument("--target-accuracy", type=float, default=0.995)
    p_cal.set_defaults(func=cmd_calibrate)

    p_rec = sub.add_parser("reconcile", help="跨文档对账")
    p_rec.add_argument("ledgers", nargs="+")
    p_rec.add_argument("--out", default=None)
    p_rec.set_defaults(func=cmd_reconcile)

    p_doc = sub.add_parser("doctor", help="检查模型服务状态")
    p_doc.add_argument("--ocr-url", default="http://127.0.0.1:8000")
    p_doc.add_argument("--jev-url", default="http://127.0.0.1:8030")
    p_doc.add_argument("--vlm-url", default="http://127.0.0.1:8001")
    p_doc.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
