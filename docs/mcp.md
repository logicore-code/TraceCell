# MCP 接口

TraceCell 可以作为一个 MCP 服务挂进任意 Agent 宿主。实现是手写的 stdio
JSON-RPC，不依赖官方 SDK——这样在任何一台装了 Python 的机器上都能起来，
不用管版本冲突。

## 启动

```bash
tracecell mcp --session samples/sessions/demo.json
```

宿主配置示例：

```json
{
  "mcpServers": {
    "tracecell": {
      "command": "tracecell",
      "args": ["mcp", "--session", "samples/sessions/demo.json"]
    }
  }
}
```

协议细节：MCP `2025-06-18`，同时兼容 `2025-03-26` 与 `2024-11-05`。
握手流程是标准的 `initialize` → `notifications/initialized` → `tools/list` /
`tools/call`。

## 工具

### `tracecell_verify_document`

核验一份文档，返回完整的账本摘要。

输入：`{"doc_id": "invoice-001"}`

返回：

```json
{
  "doc_id": "invoice-001",
  "verdict": "spot_check",
  "summary": {
    "field_count": 9,
    "region_count": 18,
    "expected_loss": 0.031,
    "band": "low",
    "review_queue_size": 2,
    "mean_region_confidence": 0.94
  },
  "fields": [
    {"name": "价税合计", "value": "145,318.00", "confidence": 0.5312,
     "severity": "critical", "status": "pending", "region_ids": ["r012"]}
  ],
  "review_queue": ["r012"],
  "findings": [...],
  "budget": {"total_cost": 5.4, "base_cost": 3.6, "extra_cost": 1.8, ...}
}
```

### `tracecell_inspect_region`

查看单个区域的全部候选。

输入：`{"doc_id": "invoice-001", "region_id": "r012"}`

返回坐标、原文、状态、决策理由，以及每个候选的来源与分数：

```json
{
  "region_id": "r012",
  "bbox": {"x0": 0.62, "y0": 0.71, "x1": 0.88, "y1": 0.74},
  "text": "价税合计（小写）¥145,318.00",
  "candidates": [
    {"text": "价税合计（小写）¥145,318.00", "origin": "primary", "score": 0.53, "verdict": "scored"},
    {"text": "价税合计（小写）¥145,818.00", "origin": "reparse-16x", "score": 0.19, "verdict": "scored"}
  ]
}
```

### `tracecell_review_checklist`

生成人工复核清单。每条包含"看哪儿、看什么、为什么看、候选有哪些"。

### `tracecell_export_ledger`

输入 `{"doc_id": "...", "format": "markdown| csv |json"}`，导出复核报告、
台账 CSV 或完整账本。

### `tracecell_reconcile_documents`

输入 `{"doc_ids": ["contract-001", "invoice-001"]}`，返回跨文档矛盾的字段、
各文档的取值与归一化结果、两侧证据坐标。

### `tracecell_calibrate_threshold`

输入 `{"samples_path": "...", "target_accuracy": 0.995}`，返回阈值、免检率、
自动通过准确率与 AUROC。

## 资源

除了工具，服务还暴露 `resources/list` 与 `resources/read`，可以直接把账本
当作资源读取：

```
tracecell://ledger/invoice-001
```

## 错误处理

- 工具执行失败返回 `isError: true` 的结果，内容是中文错误说明，宿主可以
  直接展示给用户。
- 未知工具名同样返回 `isError: true`，而不是抛 JSON-RPC 错误，这样宿主
  的错误处理只需要一条路径。
- 通知类消息（`notifications/*`）不产生响应。

## 在 Claude Code 里试一下

```bash
tracecell mcp --session samples/sessions/demo.json &
```

然后在 Claude Code 里问：

> 看一下 invoice-001 这张发票，把需要人工复核的地方列出来，并且告诉我
> 「价税合计」这一格的机器读数到底有几种说法。
