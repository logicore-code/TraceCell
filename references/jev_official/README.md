# Jev 官方样例

这三组样例来自 Jev 的官方仓库，用来验证打分链路跟官方一致：

```
https://github.com/OpenBMB/MiniCPM-OCR/tree/main/scripts/minicpm-ocr-jev/examples
```

Apache-2.0。每一组里有：

| 文件 | 是什么 |
| --- | --- |
| `block.png` | 原图块——单据上切下来的那一小块 |
| `reference_render.png` | **官方自己渲染的**候选文本图 |
| `case.json` | 候选文本、真值，以及官方跑出来的概率（`observed_prob_yes`） |

用法是拿 `reference_render.png` 当渲染图去打分，和 `case.json` 里的官方概率
对。用官方自己的渲染图，是为了绕开"我渲染得和官方不一样"这个变量，专测请求
格式、图像顺序、概率取值。

    python3 scripts/verify_against_official.py

实测三个案例的差都在 0.005 以内，表格那组差 0.00001。`verification.json` 是
本机跑出来的结果。

三个案例也各自说明了 Jev 的脾气：`text_putnam_2009` 是文本重排（文字几乎都
对，但顺序毁了）；`formula_putnam_1999` 是公式；`table_gb_standard` 是表格，
**正确转写只拿到 0.04 分**——绝对分不可信，要比的是候选之间的排序。
