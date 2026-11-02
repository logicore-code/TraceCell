"""TraceCell 的 MCP 服务。

这个进程通过 stdio 说 JSON-RPC，把 TraceCell 的能力暴露成 MCP 工具，
任何支持 MCP 的宿主（Claude Code、Cursor、各种 Agent 框架）都能直接挂上：

    tracecell mcp --session samples/sessions/demo.json

之所以手写而不是装官方 SDK：MCP 的 stdio 传输本身就是"一行一个 JSON"，
几百行就够；不引依赖意味着在任何一台装了 Python 的机器上都能起来，
不用管版本冲突。协议细节遵循 MCP 规范（initialize / tools/list / tools/call）。

暴露的工具刻意做得很细——不是"一个万能接口"，而是让宿主 Agent 能自己
决定什么时候解析一份文档、什么时候追问某个区域、什么时候做跨文档对账。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

from .. import __version__
from ..agent.loop import VerificationLoop
from ..calibration import calibrate, load_samples
from ..extractors import RegexFieldExtractor, schema_by_name
from ..ledger import export_csv, export_markdown, ledger_summary
from ..providers.replay import ReplayEscalator, ReplayParser, ReplayScorer, ReplaySession
from ..reconcile import findings_from_reconciliation, reconcile
from ..routing import render_review_checklist
from ..types import DocumentLedger

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")


class ToolError(RuntimeError):
    """工具执行失败，会以 isError 的形式回给调用方。"""


class TraceCellService:
    """工具实现。进程内缓存已经跑过的账本。"""

    def __init__(self, session_path: str | Path | None = None) -> None:
        self.session_path = Path(session_path) if session_path else None
        self.session: ReplaySession | None = None
        if self.session_path and self.session_path.exists():
            self.session = ReplaySession.load(self.session_path)
        self.ledgers: dict[str, DocumentLedger] = {}

    # -- 工具实现 ---------------------------------------------------------
    def verify_document(self, arguments: dict[str, Any]) -> dict[str, Any]:
        doc_id = arguments.get("doc_id")
        if not doc_id:
            raise ToolError("缺少参数 doc_id")
        ledger = self.ledgers.get(doc_id) or self._load_ledger(doc_id)
        summary = ledger_summary(ledger)
        return {
            "doc_id": ledger.doc_id,
            "verdict": ledger.verdict,
            "summary": summary,
            "fields": [
                {
                    "name": f.name,
                    "value": f.value,
                    "confidence": round(f.confidence, 4),
                    "severity": f.severity.value,
                    "status": f.status.value,
                    "region_ids": f.region_ids,
                }
                for f in ledger.fields
            ],
            "review_queue": ledger.review_queue,
            "findings": [f.to_dict() for f in ledger.findings],
            "budget": ledger.budget.to_dict(),
        }

    def inspect_region(self, arguments: dict[str, Any]) -> dict[str, Any]:
        doc_id = arguments.get("doc_id")
        region_id = arguments.get("region_id")
        if not doc_id or not region_id:
            raise ToolError("需要 doc_id 和 region_id")
        ledger = self.ledgers.get(doc_id) or self._load_ledger(doc_id)
        region = ledger.region(region_id)
        if region is None:
            raise ToolError(f"账本里没有区域 {region_id}")
        related = [f for f in ledger.fields if region.id in f.region_ids]
        return {
            "region_id": region.id,
            "page": region.page,
            "bbox": {
                "x0": region.bbox.x0,
                "y0": region.bbox.y0,
                "x1": region.bbox.x1,
                "y1": region.bbox.y1,
            },
            "category": region.category.value,
            "text": region.text,
            "status": region.status.value,
            "note": region.note,
            "candidates": [c.to_dict() for c in region.candidates],
            "fields": [f.to_dict() for f in related],
        }

    def review_checklist(self, arguments: dict[str, Any]) -> dict[str, Any]:
        doc_id = arguments.get("doc_id")
        if not doc_id:
            raise ToolError("缺少参数 doc_id")
        ledger = self.ledgers.get(doc_id) or self._load_ledger(doc_id)
        region_map = {r.id: r for r in ledger.regions}
        return {
            "doc_id": doc_id,
            "verdict": ledger.verdict,
            "items": render_review_checklist(region_map, ledger.fields, ledger.review_queue),
        }

    def export_ledger(self, arguments: dict[str, Any]) -> dict[str, Any]:
        doc_id = arguments.get("doc_id")
        fmt = arguments.get("format", "markdown")
        if not doc_id:
            raise ToolError("缺少参数 doc_id")
        ledger = self.ledgers.get(doc_id) or self._load_ledger(doc_id)
        if fmt == "csv":
            return {"doc_id": doc_id, "format": "csv", "content": export_csv(ledger)}
        if fmt == "json":
            return {"doc_id": doc_id, "format": "json", "content": ledger.to_json()}
        return {"doc_id": doc_id, "format": "markdown", "content": export_markdown(ledger)}

    def reconcile_documents(self, arguments: dict[str, Any]) -> dict[str, Any]:
        doc_ids = arguments.get("doc_ids") or []
        if len(doc_ids) < 2:
            raise ToolError("至少需要两个 doc_id")
        ledgers = [self.ledgers.get(doc) or self._load_ledger(doc) for doc in doc_ids]
        result = reconcile(ledgers)
        return {
            "compared_fields": result.compared_fields,
            "matched": result.matched,
            "mismatches": [m.to_dict() for m in result.mismatches],
            "single_source": result.single_source,
            "findings": [f.to_dict() for f in findings_from_reconciliation(result)],
        }

    def calibrate_threshold(self, arguments: dict[str, Any]) -> dict[str, Any]:
        path = arguments.get("samples_path")
        target = float(arguments.get("target_accuracy", 0.995))
        if not path:
            raise ToolError("缺少参数 samples_path")
        samples_path = Path(path)
        if not samples_path.exists():
            raise ToolError(f"找不到标定样本文件：{path}")
        result = calibrate(load_samples(samples_path), target_accuracy=target)
        return result.to_dict()

    # -- 内部 -------------------------------------------------------------
    def _load_ledger(self, doc_id: str) -> DocumentLedger:
        if self.session is None:
            raise ToolError("没有可用的回放会话，请用 --session 指定")
        document = self.session.document(doc_id)
        if document is None:
            raise ToolError(f"回放会话里没有文档 {doc_id}")
        schema = document.get("schema", "invoice")
        loop = VerificationLoop(
            parser=ReplayParser(self.session, doc_id),
            scorer=ReplayScorer(self.session, doc_id),
            escalator=ReplayEscalator(self.session, doc_id),
            extractor=RegexFieldExtractor(schema_by_name(schema)),
        )
        report = loop.run(b"", doc_id, document.get("source", ""))
        self.ledgers[doc_id] = report.ledger
        return report.ledger


TOOLS: list[dict[str, Any]] = [
    {
        "name": "tracecell_verify_document",
        "description": (
            "核验一份文档，产出证据账本摘要：每个字段的值、置信度、风险等级、"
            "需要人工复核的区域，以及核验成本。返回的是可追责的结论，不是裸文本。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string", "description": "文档标识"}},
            "required": ["doc_id"],
        },
    },
    {
        "name": "tracecell_inspect_region",
        "description": (
            "查看某个区域的全部候选、坐标、决策理由。用来回答"
            "「机器在这里读到了什么、它有多确定、为什么这么判断」。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
                "region_id": {"type": "string"},
            },
            "required": ["doc_id", "region_id"],
        },
    },
    {
        "name": "tracecell_review_checklist",
        "description": (
            "生成人工复核清单：需要人看的区域、为什么需要看、每个区域的候选结果。"
            "清单可以直接当工单用。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
    },
    {
        "name": "tracecell_export_ledger",
        "description": "导出账本，支持 markdown（复核报告）、csv（台账）、json（完整数据）。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
                "format": {"type": "string", "enum": ["markdown", "csv", "json"]},
            },
            "required": ["doc_id"],
        },
    },
    {
        "name": "tracecell_reconcile_documents",
        "description": (
            "对多份文档做字段级对账，检出金额、日期、主体名称上的矛盾。"
            "两侧的证据坐标都会返回。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "至少两个文档标识",
                }
            },
            "required": ["doc_ids"],
        },
    },
    {
        "name": "tracecell_calibrate_threshold",
        "description": (
            "用人工标注样本标定自动放行阈值：在目标自动通过准确率下，"
            "求出最大的免检比例。返回阈值、免检率、AUROC。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "samples_path": {"type": "string", "description": "标注样本 JSON 路径"},
                "target_accuracy": {"type": "number", "description": "目标自动通过准确率"},
            },
            "required": ["samples_path"],
        },
    },
]


class McpServer:
    """极简 MCP stdio 服务端。"""

    def __init__(self, service: TraceCellService) -> None:
        self.service = service
        self.handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "tracecell_verify_document": service.verify_document,
            "tracecell_inspect_region": service.inspect_region,
            "tracecell_review_checklist": service.review_checklist,
            "tracecell_export_ledger": service.export_ledger,
            "tracecell_reconcile_documents": service.reconcile_documents,
            "tracecell_calibrate_threshold": service.calibrate_threshold,
        }

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        message_id = message.get("id")

        if method == "initialize":
            requested = (message.get("params") or {}).get("protocolVersion")
            version = requested if requested in SUPPORTED_PROTOCOLS else PROTOCOL_VERSION
            return self._result(
                message_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "tracecell", "version": __version__},
                },
            )

        if method in ("notifications/initialized", "initialized"):
            return None

        if method == "ping":
            return self._result(message_id, {})

        if method == "tools/list":
            return self._result(message_id, {"tools": TOOLS})

        if method == "tools/call":
            params = message.get("params") or {}
            name = params.get("name", "")
            arguments = params.get("arguments") or {}
            handler = self.handlers.get(name)
            if handler is None:
                return self._result(
                    message_id,
                    {"content": [{"type": "text", "text": f"没有这个工具：{name}"}], "isError": True},
                )
            try:
                payload = handler(arguments)
            except ToolError as exc:
                return self._result(
                    message_id,
                    {"content": [{"type": "text", "text": str(exc)}], "isError": True},
                )
            except Exception as exc:  # noqa: BLE001 - 不能让工具异常打死服务
                return self._result(
                    message_id,
                    {"content": [{"type": "text", "text": f"工具执行失败：{exc}"}], "isError": True},
                )
            return self._result(
                message_id,
                {
                    "content": [
                        {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}
                    ],
                    "isError": False,
                },
            )

        if method == "resources/list":
            return self._result(
                message_id,
                {
                    "resources": [
                        {
                            "uri": f"tracecell://ledger/{doc_id}",
                            "name": f"证据账本 {doc_id}",
                            "mimeType": "application/json",
                        }
                        for doc_id in (self.service.session.doc_ids() if self.service.session else [])
                    ]
                },
            )

        if method == "resources/read":
            params = message.get("params") or {}
            uri = params.get("uri", "")
            doc_id = uri.rsplit("/", 1)[-1]
            try:
                ledger = self.service.ledgers.get(doc_id) or self.service._load_ledger(doc_id)
            except ToolError as exc:
                return self._error(message_id, -32000, str(exc))
            return self._result(
                message_id,
                {"contents": [{"uri": uri, "mimeType": "application/json", "text": ledger.to_json()}]},
            )

        if method == "prompts/list":
            return self._result(message_id, {"prompts": []})

        if message_id is None:
            return None
        return self._error(message_id, -32601, f"不支持的方法：{method}")

    @staticmethod
    def _result(message_id: Any, result: dict[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "result": result}

    @staticmethod
    def _error(message_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": message_id, "error": {"code": code, "message": message}}

    def serve(self, stdin=None, stdout=None) -> None:
        stdin = stdin or sys.stdin
        stdout = stdout or sys.stdout
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = self.handle(message)
            if response is not None:
                stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TraceCell MCP 服务")
    parser.add_argument(
        "--session",
        default="samples/sessions/demo.json",
        help="回放会话路径（离线演示用）",
    )
    args = parser.parse_args(argv)
    server = McpServer(TraceCellService(args.session))
    server.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
