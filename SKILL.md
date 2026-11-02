---
name: tracecell
description: 把 OCR 结果变成可追责的证据账本。用 MiniCPM-OCR 1.0 解析文档、MiniCPM-OCR Jev 给每个区域打分、TraceCell 的预算决策器决定哪些区域要重解析/升级复核/转人工，最后产出带坐标和置信度的字段台账。当用户需要"解析并核验单据/合同/发票"、"检查 OCR 结果是否可信"、"把纸质文档变成可导出的台账"、"把文档核验接进 Agent/MCP"时使用。
---

# TraceCell

## 这个工具解决什么

OCR 工具给你一段 Markdown，TraceCell 给你一份**可以签字的结论**：每个字段
的值、从原图哪一格读出来的、有多可信、还需不需要人再看一眼。

三个 MiniCPM 模型串成一条会自我纠错的链：

- **MiniCPM-OCR 1.0**（1.3B）解析文档，给出带坐标的文本块
- **MiniCPM-OCR Jev**（2B）给每个块打 0~1 的正确性分
- **MiniCPM-V 4.7**（35B-A3B）在前两者都搞不定时做最后裁决

TraceCell 自己负责中间的决策：什么时候再查一次、什么时候转人工、这份文档
能不能免检。

## 快速开始（不需要 GPU）

```bash
pip install -e .
tracecell demo                      # 跑内置样例，看完整链路
tracecell serve                     # 起 Web 界面 http://127.0.0.1:8000
```

回放模式跑的是同一套策略代码，只是把模型调用换成了录制数据。

## 接真实模型

```bash
# 1. MiniCPM-OCR 1.0（解析）
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model openbmb/MiniCPM-OCR-1.0 --served-model-name openbmb/MiniCPM-OCR-1.0 \
  --port 8000 --trust-remote-code --gpu-memory-utilization=0.28

# 2. MiniCPM-OCR Jev（评分）
#    见 docs/deployment.md，镜像来自 OpenBMB/MiniCPM-OCR 的 scripts/minicpm-ocr-jev
docker run -d --gpus all -p 127.0.0.1:8030:8030 \
  -v /path/to/jev-weights:/weights:ro minicpm-ocr-jev:latest 8030 /weights

# 3. 跑起来
tracecell doctor                    # 确认三个服务都活着
tracecell parse 票据.png --provider minicpm --schema invoice
```

## 在代码里用

```python
from tracecell.agent.loop import VerificationLoop
from tracecell.extractors import RegexFieldExtractor, schema_by_name
from tracecell.providers.jev import HttpJevScorer
from tracecell.providers.ocr import OpenAICompatOcr
from tracecell.providers.vlm import OpenAICompatVlm

loop = VerificationLoop(
    parser=OpenAICompatOcr(base_url="http://127.0.0.1:8000"),
    scorer=HttpJevScorer(base_url="http://127.0.0.1:8030"),
    escalator=OpenAICompatVlm(base_url="http://127.0.0.1:8001"),
    extractor=RegexFieldExtractor(schema_by_name("invoice")),
)

report = loop.run(open("票据.png", "rb").read(), doc_id="inv-001")
ledger = report.ledger

print(ledger.verdict)               # auto_pass / spot_check / full_review
for f in ledger.fields:
    print(f.name, f.value, f.confidence, f.region_ids)
```

## 作为 MCP 服务挂到 Agent 上

```bash
tracecell mcp --session samples/sessions/demo.json
```

提供的工具：

| 工具 | 用途 |
| --- | --- |
| `tracecell_verify_document` | 核验一份文档，返回字段、置信度、待复核清单、成本 |
| `tracecell_inspect_region` | 看某个区域的全部候选、坐标、决策理由 |
| `tracecell_review_checklist` | 生成可直接当工单用的人工复核清单 |
| `tracecell_export_ledger` | 导出 markdown / csv / json |
| `tracecell_reconcile_documents` | 多份文档之间找金额、日期、主体名称的矛盾 |
| `tracecell_calibrate_threshold` | 用标注样本标定自动放行阈值 |

## 关键约定（改代码前务必看）

1. **Jev 的 `verdict` 必须检查。** 不是 `scored` 时返回的分数是失败态固定值，
   不能当成"模型认为识别错了"。`providers/jev.py` 已经处理好，接入别的
   服务时也要照做。
2. **字段置信度取支撑区域的最小值，不取平均。** 任何一处认错，这个字段
   就是错的。
3. **抽不到就是抽不到。** 字段抽取失败时留空并标成"无证据支撑"，不要填
   占位值。
4. **坐标一律归一化到 0~1**，因为版面检测、浏览器渲染、打印输出的像素
   尺寸都不一样。

## 文档

- `docs/method.md` —— 四个机制和公式
- `docs/calibration.md` —— 阈值怎么标
- `docs/deployment.md` —— 三种部署形态
- `docs/mcp.md` —— MCP 接口细节
