"""MCP 服务端的协议测试。"""

from __future__ import annotations

import io
import json

import pytest

from tracecell.mcp.server import McpServer, TraceCellService


@pytest.fixture()
def server(tmp_path):
    session = {
        "documents": [
            {
                "doc_id": "doc-1",
                "source": "samples/doc-1.png",
                "label": "测试单据",
                "schema": "invoice",
                "regions": [
                    {
                        "id": "r000",
                        "page": 1,
                        "bbox": {"x0": 0.1, "y0": 0.1, "x1": 0.6, "y1": 0.2},
                        "category": "text",
                        "text": "价税合计（小写）¥145,318.00",
                        "candidates": [
                            {
                                "text": "价税合计（小写）¥145,318.00",
                                "origin": "primary",
                                "score": 0.97,
                                "verdict": "scored",
                            }
                        ],
                    }
                ],
                "fields": [
                    {"name": "价税合计", "value": "145,318.00", "region_ids": ["r000"]}
                ],
            }
        ]
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(session, ensure_ascii=False), encoding="utf-8")
    return McpServer(TraceCellService(path))


def call(server, method, params=None, message_id=1):
    message = {"jsonrpc": "2.0", "id": message_id, "method": method}
    if params is not None:
        message["params"] = params
    return server.handle(message)


class TestMcpProtocol:
    def test_initialize_handshake(self, server):
        response = call(server, "initialize", {"protocolVersion": "2025-06-18"})
        result = response["result"]
        assert result["protocolVersion"] == "2025-06-18"
        assert result["serverInfo"]["name"] == "tracecell"
        assert "tools" in result["capabilities"]

    def test_unknown_protocol_version_falls_back(self, server):
        response = call(server, "initialize", {"protocolVersion": "1999-01-01"})
        assert response["result"]["protocolVersion"] == "2025-06-18"

    def test_tools_list_declares_schemas(self, server):
        tools = call(server, "tools/list")["result"]["tools"]
        assert tools
        for tool in tools:
            assert tool["name"].startswith("tracecell_")
            assert tool["description"]
            assert tool["inputSchema"]["type"] == "object"

    def test_tools_call_returns_text_content(self, server):
        response = call(
            server,
            "tools/call",
            {"name": "tracecell_verify_document", "arguments": {"doc_id": "doc-1"}},
        )
        result = response["result"]
        assert result["isError"] is False
        payload = json.loads(result["content"][0]["text"])
        assert payload["doc_id"] == "doc-1"
        assert payload["fields"]

    def test_unknown_tool_is_reported(self, server):
        response = call(server, "tools/call", {"name": "tracecell_nope", "arguments": {}})
        assert response["result"]["isError"] is True

    def test_missing_argument_is_reported(self, server):
        response = call(
            server, "tools/call", {"name": "tracecell_verify_document", "arguments": {}}
        )
        assert response["result"]["isError"] is True
        assert "doc_id" in response["result"]["content"][0]["text"]

    def test_inspect_region_returns_candidates(self, server):
        response = call(
            server,
            "tools/call",
            {"name": "tracecell_inspect_region", "arguments": {"doc_id": "doc-1", "region_id": "r000"}},
        )
        payload = json.loads(response["result"]["content"][0]["text"])
        assert payload["region_id"] == "r000"
        assert payload["candidates"][0]["score"] == 0.97
        assert payload["bbox"]["x0"] == 0.1

    def test_export_ledger_supports_csv(self, server):
        response = call(
            server,
            "tools/call",
            {"name": "tracecell_export_ledger", "arguments": {"doc_id": "doc-1", "format": "csv"}},
        )
        payload = json.loads(response["result"]["content"][0]["text"])
        assert payload["format"] == "csv"
        assert "价税合计" in payload["content"]

    def test_notifications_do_not_produce_a_response(self, server):
        assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    def test_stdio_loop_handles_multiple_lines(self, server):
        lines = "\n".join(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            ]
        )
        output = io.StringIO()
        server.serve(io.StringIO(lines + "\n"), output)
        responses = [json.loads(line) for line in output.getvalue().splitlines() if line.strip()]
        assert len(responses) == 2
        assert responses[0]["id"] == 1
        assert responses[1]["id"] == 2
