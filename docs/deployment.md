# 部署

TraceCell 的三个模型都可以本地跑，数据不出机器。下面按"从最重到最轻"给出三种部署形态。

| 形态 | 硬件 | 能力 | 适用 |
| --- | --- | --- | --- |
| A 全离线单卡 | 1×24GB 显卡，V 4.7 用 1B 版 | 解析 + 评分 + 升级复核 | 生产环境、数据不能出内网 |
| A' 全离线单卡 | 1×48GB 显卡，V 4.7 用 35B-A3B AWQ | 同上，疑难区域判得更准 | 对复核质量有要求的场景 |
| B 端侧 CPU | 笔记本，纯 CPU | 解析 + 评分 | 现场作业、离线演示 |
| C 回放模式 | 无需 GPU | 全流程跑通 | 评审、CI、开发 |

显存预算见文末的表。V 4.7 的 35B-A3B 版 BF16 权重是 65.6 GiB，单卡跑不下，
要么换 AWQ，要么换 1B 版。

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

> `verdict` 字段务必检查。服务在图像解码失败、渲染失败时会返回固定分数，
> 含义是"没跑成"，不是"跑出来说不对"。TraceCell 的客户端把这两种情况分开
> 处理，见 `providers/jev.py`。

### 3. MiniCPM-V 4.7（升级复核）

BF16 权重 65.6 GiB，24GB 卡放不下。按显存选版本：

```bash
# 24GB 卡：1B 版（BF16 约 3.1GB）
pip install "transformers[serving]>=5.18.0"
transformers serve openbmb/MiniCPM-V-4.7-1B --port 8001 --host 0.0.0.0

# 48GB 卡：35B-A3B 的 AWQ 版（约 22.7GB）
transformers serve openbmb/MiniCPM-V-4.7-35B-A3B-AWQ --port 8001 --host 0.0.0.0
```

或者用 Ollama（端侧最省事，官方已经收录）：

```bash
ollama run openbmb/minicpm-v4.7
```

docker-compose 里的默认模型是 35B-A3B 的 AWQ 版，24GB 卡的环境把
`VLM_MODEL` 换成 `openbmb/MiniCPM-V-4.7-1B` 即可。

### 4. 版面检测（可选但建议）

TraceCell 自带一个投影法版面切分器，不依赖任何模型，复杂表格和多栏排版会偏。
真要处理扫描件，换成官方的 PP-DocLayoutV3——这也是 MiniCPM-OCR 官方流水线
用的那一个：

```bash
pip install onnxruntime
python3 scripts/fetch_layout_model.py          # 约 136MB，ONNX 版
export TRACECELL_LAYOUT_MODEL=$(python3 scripts/fetch_layout_model.py --path-only)
```

模型文件不进仓库，用环境变量指路。没设这个变量就继续用自带的投影法，流程
照跑不误。

### 5. 起 TraceCell

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

视觉压缩档位直接影响 OCR 质量与速度：`4x` 保留更多视觉 token，适合密集文字
与表格；`16x` 更快，适合实时交互。TraceCell 的对抗式核验靠"同一块内容用两档
参数各读一次"来制造互相独立的候选。

## C. 回放模式（无需 GPU）

仓库自带一份录制的会话，样例图 + 解析结果 + 评分都在里面。评审和 CI 用它就够了：

```bash
tracecell demo
tracecell serve            # http://127.0.0.1:8000
```

回放模式跑的是同一套策略代码，把三个模型的调用换成了录制数据。换成真实模型只需要：

```bash
tracecell serve --provider minicpm
```

## 显存预算

权重实际占用的显存（不含 KV cache 与激活）：

| 模型 | 精度 | 权重体积 |
| --- | --- | --- |
| MiniCPM-OCR 1.0 | BF16 | 2.6 GB |
| MiniCPM-OCR 1.0（mmproj 4x） | F16 | 0.9 GB |
| MiniCPM-OCR Jev | BF16 | 4.2 GB |
| MiniCPM-V 4.7 35B-A3B | BF16 | 65.6 GB |
| MiniCPM-V 4.7 35B-A3B | AWQ | 22.7 GB |
| MiniCPM-V 4.7 35B-A3B | FP8 | 38.0 GB |
| MiniCPM-V 4.7 1B | BF16 | 3.1 GB |

24GB 卡（三个模型都在同一张卡上）：OCR 1.0 用 BF16、Jev 用 BF16、V 4.7 用
1B 版，权重合计约 10.8GB，剩下的留给 KV cache 和视觉编码的中间激活。

48GB 卡：把 V 4.7 换成 35B-A3B 的 AWQ 版，权重合计约 30.4GB，还剩十几 GB
做 KV cache。想要 BF16 的 35B-A3B，显存得往 80GB 走，或者用官方那套把专家
权重分片、按需从 SSD 读的端侧方案（见 OpenBMB/MiniCPM-V-Apps）。

再不够就按这个顺序降级：V 4.7 换 1B → 关掉升级复核（转人工）→ OCR 和 Jev
也换 GGUF 走 CPU。

## 常见问题

**Q：Jev 的分数能不能直接当概率用？**
不能。AUROC 93.6 是"区分对错的能力"，不是"正确率"。阈值必须在自己业务的数据上标，
`tracecell calibrate` 就是干这个的。

**Q：`verdict` 不是 `scored` 时分数是 0，是不是说明识别错了？**
不是。那是失败态的固定值，说明这一步根本没跑成。TraceCell 会把它记成"未评分"，不参与置信度计算。

**Q：多模态升级复核为什么不用整页重读？**
窄问题可判定。一句"这块是 145,318.00 还是 145,818.00"的答案可以直接变成新候选再交给 Jev 复评；一段散文不行。

**Q：能用其它 OCR 模型吗？**
可以。Jev 官方在 GLM-OCR、PaddleOCR-VL-1.6、Xiaomi-OCR-0 的解析结果上测过，AUROC 保持在 91.9~93.4。接入方式是实现 `providers/base.py` 里的 `Parser` 协议。
