# 部署

TraceCell 的三个模型都可以本地跑，数据不出机器。下面按"从最重到最轻"给出三种部署形态。

| 形态 | 硬件 | 能力 | 适用 |
| --- | --- | --- | --- |
| A 全离线单卡 | 1×24GB 显卡 | 解析 + 评分 + 升级复核 | 生产环境、数据不能出内网 |
| B 端侧 CPU | 笔记本，纯 CPU | 解析 + 评分 | 现场作业、离线演示 |
| C 回放模式 | 无需 GPU | 全流程跑通 | 评审、CI、开发 |

---

## A. 全离线单卡

三个模型按职责分工，串在一条流水线上：

```
     图像
      │
      ▼
┌──────────────────┐   区域 + 文本
│ MiniCPM-OCR 1.0  │──────────────┐
│   1.3B  解析     │              │
└──────────────────┘              ▼
                          ┌──────────────────┐
                          │ MiniCPM-OCR Jev  │  0~1 正确性分
                          │   2B   评分      │──────────────┐
                          └──────────────────┘              │
                                                            ▼
                                                  ┌──────────────────┐
                                                  │  TraceCell 决策  │
                                                  │  预算 / 路由     │
                                                  └────────┬─────────┘
                                     ┌───────────────────┼──────────────────┐
                                     ▼                   ▼                  ▼
                              放行(免检)          转人工复核        ┌──────────────────┐
                                                                       │ MiniCPM-V 4.7   │
                                                                       │ 35B-A3B 升级复核 │
                                                                       └──────────────────┘
```

### 1. MiniCPM-OCR 1.0（解析）

用 vLLM 起 OpenAI 兼容服务：

```bash
docker run --gpus all -p 8000:8000 \
  -v hf-cache:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model openbmb/MiniCPM-OCR-1.0 \
  --served-model-name openbmb/MiniCPM-OCR-1.0 \
  --port 8000 --trust-remote-code \
  --limit-mm-per-prompt.image=8 \
  --gpu-memory-utilization=0.28
```

国内网络建议加 `HF_ENDPOINT=https://hf-mirror.com`。

也可以用官方 pip SDK（自带版面检测与 Markdown 拼装）：

```bash
pip install minicpm-ocr
```

### 2. MiniCPM-OCR Jev（评分）

Jev 官方提供 Docker 镜像，接口带渲染器，调用方只要传原图、候选文本和区域类别：

```bash
git clone https://github.com/OpenBMB/MiniCPM-OCR.git
cd MiniCPM-OCR/scripts/minicpm-ocr-jev
docker build -f serve/Dockerfile -t minicpm-ocr-jev:latest .

# 权重
hf download openbmb/MiniCPM-OCR-Jev --local-dir ./jev-weights

docker run -d --name minicpm-ocr-jev --gpus all \
  -p 127.0.0.1:8030:8030 \
  -v $(pwd)/jev-weights:/weights:ro \
  minicpm-ocr-jev:latest 8030 /weights

curl http://127.0.0.1:8030/healthz     # {"status":"ok"}
```

> **务必检查 `verdict` 字段。** 服务在图像解码失败、渲染失败时会返回固定分数，那是"没跑成"而不是"跑出来说不对"。TraceCell 的客户端会把这两种情况分开处理，见 `providers/jev.py`。

### 3. MiniCPM-V 4.7（升级复核）

```bash
pip install "transformers[serving]>=5.18.0"
transformers serve openbmb/MiniCPM-V-4.7-35B-A3B --port 8001 --host 0.0.0.0
```

或者用 Ollama（端侧最省事）：

```bash
ollama run openbmb/minicpm-v4.7
```

显存紧张时换成 1B 版本：`openbmb/MiniCPM-V-4.7-1B`。

### 4. 起 TraceCell

```bash
docker compose -f deploy/docker-compose.yml up -d
# 打开 http://127.0.0.1:8080
```

或本地开发模式：

```bash
pip install -e ".[web]"
tracecell serve --provider minicpm --port 8080
```

先确认三个服务都活着：

```bash
tracecell doctor
```

---

## B. 端侧 CPU（llama.cpp）

没有显卡时用 GGUF 量化权重，CPU 也能跑。

```bash
# MiniCPM-OCR 1.0：Q4_K_M 约 530MB，4x 版 mmproj 约 900MB
git lfs install
git clone https://huggingface.co/openbmb/MiniCPM-OCR-1.0-gguf

# Jev 的 GGUF 版本
git clone https://huggingface.co/openbmb/MiniCPM-OCR-Jev-gguf
```

调用（llama.cpp 的多模态 CLI）：

```bash
llama-mtmd-cli \
  -m MiniCPM-OCR-1_0-Q4_K_M.gguf \
  --mmproj mmproj-model-f16-4x.gguf \
  --image samples/invoice_001.png \
  --prompt "请解析这张图片，按阅读顺序输出内容。"
```

TraceCell 通过 `LlamaCppOcr` 走这条路径：

```python
from tracecell.providers.ocr import LlamaCppOcr

ocr = LlamaCppOcr(
    binary="llama-mtmd-cli",
    model_path="MiniCPM-OCR-1_0-Q4_K_M.gguf",
    mmproj_path="mmproj-model-f16-4x.gguf",
)
```

**视觉压缩档位**直接影响 OCR 质量与速度：`4x` 保留更多视觉 token，适合密集文字与表格；`16x` 更快，适合实时交互。TraceCell 的对抗式核验正是靠"同一块内容用两档参数各读一次"来制造互相独立的候选。

---

## C. 回放模式（无需 GPU）

仓库自带一份录制的会话，样例图 + 解析结果 + 评分都在里面。评审和 CI 用它就够了：

```bash
tracecell demo
tracecell serve            # http://127.0.0.1:8000
```

回放模式跑的是**同一套策略代码**，只是把三个模型的调用换成了录制数据。换成真实模型只需要：

```bash
tracecell serve --provider minicpm
```

---

## 显存预算

24GB 卡的推荐分配：

| 服务 | 权重 | `--gpu-memory-utilization` | 说明 |
| --- | --- | --- | --- |
| MiniCPM-OCR 1.0 | 1.3B | 0.28 | 视觉编码器占得多 |
| MiniCPM-OCR Jev | 2B | 0.25 | 二分类头，激活少 |
| MiniCPM-V 4.7 | 35B-A3B | 0.36 | MoE，权重分片但激活只有 3B |

不够就按这个顺序降级：V 4.7 → 1B 版本 → GGUF 端侧 → 关掉升级复核（转人工）。

---

## 常见问题

**Q：Jev 的分数能不能直接当概率用？**
不能。AUROC 93.6 是"区分对错的能力"，不是"正确率"。阈值必须在自己业务的数据上标，见 `docs/calibration.md`。

**Q：`verdict` 不是 `scored` 时分数是 0，是不是说明识别错了？**
不是。那是失败态的固定值，说明这一步根本没跑成。TraceCell 会把它记成"未评分"，不参与置信度计算。

**Q：多模态升级复核为什么不用整页重读？**
窄问题可判定。一句"这块是 145,318.00 还是 145,818.00"的答案可以直接变成新候选再交给 Jev 复评；一段散文不行。

**Q：能用其它 OCR 模型吗？**
可以。Jev 官方在 GLM-OCR、PaddleOCR-VL-1.6、Xiaomi-OCR-0 的解析结果上测过，AUROC 保持在 91.9~93.4。接入方式是实现 `providers/base.py` 里的 `Parser` 协议。
