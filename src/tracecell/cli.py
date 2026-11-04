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
from .audit import AuditLog
from .baseline import compare as compare_baselines
from .batch import discover, run_batch, write_summary
from .degrade import DegradationLog, ResilientEscalator, ResilientParser, ResilientScorer
from .export import export_bundle, export_csv, export_excel, export_json, export_markdown
from .feedback import apply_all, load_feedback, save_feedback, to_labeled_samples
from .agent.loop import VerificationLoop
from .calibration import calibrate, load_samples, per_category_stats
from .extractors import build_extractor
from .scenarios import list_scenarios
from .ledger import export_csv, export_markdown, ledger_summary
from .policy import BudgetPolicy, RiskPolicy
from .providers.jev import HttpJevScorer
from .providers.ocr import OpenAICompatOcr
from .providers.replay import ReplayEscalator, ReplayParser, ReplayScorer, ReplaySession
from .providers.vlm import OpenAICompatVlm
from .reconcile import merge_ledgers, reconcile

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION = ROOT / "samples" / "sessions"


def _read_bytes(path: str) -> bytes:
    data = Path(path).read_bytes()
    if not data:
        raise SystemExit(f"文件是空的：{path}")
    return data


def _build_components(args, doc_id: str):
    """按 provider 组装三个模型的客户端，同时确定这份文档的场景。"""

    if args.provider == "minicpm":
        parser = OpenAICompatOcr(base_url=args.ocr_url, model=args.ocr_model)
        scorer = HttpJevScorer(base_url=args.jev_url)
        escalator = OpenAICompatVlm(base_url=args.vlm_url, model=args.vlm_model)
        return parser, scorer, escalator, args.scenario

    session = _load_session(args.session)
    if doc_id not in session.doc_ids():
        found = session.find_by_source(doc_id)
        if found is None:
            raise SystemExit(
                f"回放会话里没有文档 {doc_id}，可选：{', '.join(session.doc_ids())}"
            )
        doc_id = found
    document = session.document(doc_id) or {}
    scenario = document.get("schema") or args.scenario
    parser = ReplayParser(session, doc_id)
    scorer = ReplayScorer(session, doc_id)
    escalator = ReplayEscalator(session, doc_id)
    return parser, scorer, escalator, scenario


def _build_loop(args, doc_id: str, scenario: str | None = None) -> VerificationLoop:
    parser, scorer, escalator, detected = _build_components(args, doc_id)
    scenario = scenario or detected
    # 三个模型都套上降级包装：任何一个挂了都不至于整份核验崩掉，
    # 具体降级行为写进账本的 degradation 字段。
    degradation = DegradationLog()
    policy = BudgetPolicy()
    if getattr(args, "policy", None):
        policy = BudgetPolicy.from_dict(json.loads(Path(args.policy).read_text(encoding="utf-8")))
    if getattr(args, "calibrate", None):
        policy = BudgetPolicy.from_calibration(args.calibrate, base=policy)
        print(
            f"阈值已按标注样本标定为 {policy.accept_threshold:.3f}"
            f"（免检率 {policy.calibration.get('accept_rate', 0):.2%}）",
            file=sys.stderr,
        )
    return VerificationLoop(
        parser=ResilientParser(parser, degradation),
        scorer=ResilientScorer(scorer, degradation),
        escalator=ResilientEscalator(escalator, degradation),
        policy=policy,
        risk_policy=RiskPolicy(),
        extractor=build_extractor(scenario),
        scenario=scenario,
        audit=AuditLog(),
        degradation=degradation,
    )


def _load_session(path: str) -> ReplaySession:
    """加载会话。给目录就合并目录下的所有会话文件。"""

    target = Path(path)
    if target.is_dir():
        return ReplaySession.load_dir(target)
    return ReplaySession.load(target)


def cmd_parse(args) -> int:
    image = _read_bytes(args.image)
    doc_id = args.doc_id or Path(args.image).stem
    loop = _build_loop(args, doc_id, args.scenario)
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
        _write_output(ledger, args.out, args.format)
        print(f"已保存：{args.out}", file=sys.stderr)
    if args.audit_out:
        Path(args.audit_out).write_text(report.audit.to_json(), encoding="utf-8")
        print(f"轨迹已保存：{args.audit_out}", file=sys.stderr)
    return 0


def _write_output(ledger, out_path: str, fmt: str) -> None:
    """按格式写出账本。"""

    target = Path(out_path)
    if fmt == "csv":
        target.write_text(export_csv(ledger), encoding="utf-8")
    elif fmt == "xlsx":
        target.write_bytes(export_excel(ledger))
    elif fmt == "json":
        target.write_text(export_json(ledger), encoding="utf-8")
    elif fmt == "bundle":
        target.mkdir(parents=True, exist_ok=True)
        for name, content in export_bundle(ledger).items():
            (target / name).write_bytes(content)
    else:
        target.write_text(export_markdown(ledger), encoding="utf-8")


def cmd_demo(args) -> int:
    session = _load_session(args.session)
    for doc_id in session.doc_ids():
        document = session.document(doc_id) or {}
        loop = _build_loop(args, doc_id)
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
    parser, scorer, escalator, _scenario = _build_components(args, doc_id)

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


def cmd_batch(args) -> int:
    """批量核验一个目录里的文档。"""

    files = discover(args.paths)
    if not files:
        print("没有找到可处理的文档（支持 png/jpg/webp/pdf）", file=sys.stderr)
        return 1

    def process(path: Path, scenario: str):
        loop = _build_loop(args, path.stem, scenario)
        report = loop.run(str(path), path.stem, str(path))
        return report.ledger

    summary = run_batch(files, process, default_scenario=args.scenario, jobs=args.jobs)
    print(f"共 {len(summary.items)} 份，成功 {len(summary.ok)}，失败 {len(summary.failed)}")
    for item in summary.items:
        if item.ledger is None:
            print(f"  ✗ {item.source}：{item.error}")
        else:
            print(
                f"  {'✓' if item.ledger.verdict != 'full_review' else '!'} {item.source}"
                f"  {item.ledger.verdict}  字段 {len(item.ledger.fields)}"
                f"  问题 {len(item.ledger.findings)}  待人工 {len(item.ledger.review_queue)}"
            )

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in summary.ok:
            target = out_dir / f"{item.doc_id}.json"
            target.write_text(export_json(item.ledger), encoding="utf-8")
        write_summary(summary, out_dir / "summary.json")
        print(f"\n账本与汇总已写入 {out_dir}", file=sys.stderr)
    if args.summary_out:
        write_summary(summary, args.summary_out)
    return 0


def cmd_feedback(args) -> int:
    """把人工复核结论回灌到账本，并导出标定样本。"""

    from .types import DocumentLedger

    ledger = DocumentLedger.from_json(Path(args.ledger).read_text(encoding="utf-8"))
    items = load_feedback(args.feedback)
    applied = [item for item in items if item.doc_id == ledger.doc_id]
    if not applied:
        print(f"没有针对 {ledger.doc_id} 的复核记录", file=sys.stderr)
        return 1

    apply_all(ledger, applied)
    if args.out:
        Path(args.out).write_text(export_json(ledger), encoding="utf-8")
        print(f"已回灌并保存：{args.out}", file=sys.stderr)

    if args.labels_out:
        samples = to_labeled_samples(applied, [ledger])
        Path(args.labels_out).write_text(
            json.dumps([s.to_dict() for s in samples], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"已导出 {len(samples)} 条标定样本：{args.labels_out}", file=sys.stderr
        )
    return 0


def cmd_fit_gains(args) -> int:
    """从录制数据里回归"换一种策略能修好这一格的概率"。"""

    import json as _json

    from .budget import fit_repair_gains

    sessions = []
    for path in args.sessions:
        sessions.append(_json.loads(Path(path).read_text(encoding="utf-8")))
    gains = fit_repair_gains(sessions)
    print("回归出的期望增益：")
    for name, value in sorted(gains.items()):
        print(f"  {name:16s} {value:.4f}")

    policy = BudgetPolicy()
    policy.repair_gains = gains
    if args.out:
        Path(args.out).write_text(
            _json.dumps(policy.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"策略配置已保存：{args.out}", file=sys.stderr)
    return 0


def cmd_baseline(args) -> int:
    """三条路线的基线对比：人工要花多少力气，能拦下多少错误。"""

    from .baseline import compare as compare_baselines
    from .reconcile import check_sum_consistency, reconcile
    from .scenarios import get_scenario

    session = _load_session(args.session)
    ledgers = []
    truths: dict[tuple[str, str], str] = {}

    for doc_id in session.doc_ids():
        document = session.document(doc_id) or {}
        scenario = document.get("schema", "invoice")
        loop = _build_loop(args, doc_id, scenario)
        report = loop.run(b"", doc_id, document.get("source", ""))
        ledgers.append(report.ledger)
        for region in document.get("regions", []):
            truth = region.get("truth") or region.get("text")
            if truth:
                truths[(doc_id, region["id"])] = truth

    # 单据自身的矛盾：内部勾稽 + 跨文档对账
    document_errors = 0
    for ledger in ledgers:
        rules = get_scenario(str(ledger.meta.get("scenario", "invoice"))).rules
        document_errors += len(check_sum_consistency(ledger.fields, rules))
    document_errors += len(reconcile(ledgers).mismatches)

    result = compare_baselines(ledgers, truths, document_errors=document_errors)
    print(result.render())
    if args.out:
        Path(args.out).write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n已保存：{args.out}", file=sys.stderr)
    return 0


def cmd_scenarios(args) -> int:
    """列出支持的场景及其勾稽关系。"""

    for item in list_scenarios():
        print(f"{item['name']:20s} {item['label']}")
        print(f"{'':20s} {item['description']}")
        print(f"{'':20s} 字段 {item['field_count']} 个，勾稽关系 {item['rule_count']} 条")
        print()
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
        sp.add_argument("--policy", default=None, help="策略配置 JSON 路径")
        sp.add_argument(
            "--calibrate",
            default=None,
            help="用这份标注样本标定放行阈值，覆盖默认阈值",
        )
        sp.add_argument(
            "--scenario",
            default="invoice",
            choices=[item["name"] for item in list_scenarios()],
            help="核验场景，决定抽哪些字段、查哪些勾稽关系",
        )

    p_parse = sub.add_parser("parse", help="解析并核验一份文档")
    p_parse.add_argument("image")
    p_parse.add_argument("--doc-id", default=None)
    p_parse.add_argument(
        "--format",
        default="markdown",
        choices=["markdown", "json", "csv", "xlsx", "bundle"],
    )
    p_parse.add_argument("--out", default=None, help="输出路径（bundle 格式时是目录）")
    p_parse.add_argument("--audit-out", default=None, help="把执行轨迹存到这里")
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

    p_batch = sub.add_parser("batch", help="批量核验一个目录里的文档")
    p_batch.add_argument("paths", nargs="+", help="文件或目录")
    p_batch.add_argument("--out", default=None, help="账本与汇总的输出目录")
    p_batch.add_argument("--summary-out", default=None, help="只导出汇总 JSON")
    p_batch.add_argument("--jobs", type=int, default=1, help="并发数，模型调用是网络等待，可以开大点")
    add_provider_args(p_batch)
    p_batch.set_defaults(func=cmd_batch)

    p_fb = sub.add_parser("feedback", help="把人工复核结论回灌到账本")
    p_fb.add_argument("ledger", help="账本 JSON 路径")
    p_fb.add_argument("--feedback", required=True, help="复核记录 JSON 路径")
    p_fb.add_argument("--out", default=None, help="回灌后的账本输出路径")
    p_fb.add_argument("--labels-out", default=None, help="导出标定样本的路径")
    p_fb.set_defaults(func=cmd_feedback)

    p_base = sub.add_parser("baseline", help="三条路线的基线对比")
    p_base.add_argument("--out", default=None, help="结果 JSON 输出路径")
    add_provider_args(p_base)
    p_base.set_defaults(func=cmd_baseline)

    p_fit = sub.add_parser("fit-gains", help="从录制数据回归修复增益系数")
    p_fit.add_argument("sessions", nargs="+", help="会话 JSON 路径")
    p_fit.add_argument("--out", default=None, help="策略配置输出路径")
    p_fit.set_defaults(func=cmd_fit_gains)

    p_scen = sub.add_parser("scenarios", help="列出支持的核验场景")
    p_scen.set_defaults(func=cmd_scenarios)

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
