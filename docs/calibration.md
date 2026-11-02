# 阈值标定

MiniCPM-OCR Jev 输出 0~1 的分数。**这个分数不是"识别正确的概率"。**

官方模型卡写得很清楚：Jev 在 JevOCRBench 上的 decision AUROC 是 93.6。
AUROC 衡量的是"能不能把对的和错的分开"，不是"自动通过的部分有多少是对的"。
直接拿 0.9 当阈值、然后宣称"90% 准确"，是把两个不同的指标混为一谈。

要用在生产上，就得拿自己业务里的数据标一次。这件事没有捷径，但有套路。

---

## 一、准备标注样本

样本 = 一条候选识别结果 + 一个人工判断（对/错）。

```json
[
  {"score": 0.97, "correct": true,  "category": "text"},
  {"score": 0.94, "correct": true,  "category": "text"},
  {"score": 0.61, "correct": false, "category": "table"},
  {"score": 0.22, "correct": false, "category": "table"}
]
```

**怎么采：**

1. 从真实业务里捞一批文档，走一遍 TraceCell，把所有候选和它们的分数导出。
2. 人眼看原图，逐条标注"这个候选对不对"。
3. **按类别分层**：text / formula / table 三类的错误形态差别很大（错字、
   上下标、行列错位），混在一起标会得到一个谁都不适用的阈值。
4. 样本量：每个类别至少 100 条，其中**要包含低分样本**。全采高分样本的话，
   阈值会标得虚高。

导出候选的代码：

```python
from tracecell.providers.replay import ReplaySession
from tracecell.types import Verdict

session = ReplaySession.load("samples/sessions/demo.json")
for doc_id in session.doc_ids():
    for region in session.document(doc_id)["regions"]:
        for candidate in region["candidates"]:
            print({
                "doc_id": doc_id,
                "region_id": region["id"],
                "text": candidate["text"],
                "score": candidate.get("score"),
                "category": region["category"],
            })
```

---

## 二、跑标定

```bash
tracecell calibrate samples/labelled.json --target-accuracy 0.995
```

输出：

```
阈值 0.920
免检率 72.34%（340/470）
自动通过准确率 99.71%，漏放 1 处
AUROC 0.9381

分类别统计：
  table     样本 150  平均分 0.842  AUROC 0.9102
  text      样本 320  平均分 0.913  AUROC 0.9447
```

三个数要一起看：

- **免检率**：省掉的人工。70% 意味着七成的区域不用人看。
- **自动通过准确率**：放行的那些里面有多少真的对。这是安全线。
- **AUROC**：分数本身的区分能力。如果你的数据上 AUROC 只有 0.75，
  说明这个场景不适合用 Jev 做闸门，再怎么调阈值也救不回来。

---

## 三、按成本选阈值

纯准确率不是唯一目标。把人工成本摊进来：

```python
from tracecell.calibration import recommend_threshold

best = recommend_threshold(
    samples,
    human_review_cost=8.0,   # 一次人工复核的相对成本
    error_cost=1.0,          # 漏放一个错误的代价
)
```

总期望成本：

```
E(t) = [ 漏放数(t) × error_cost + 转人工数(t) × human_review_cost ] / 总数
```

取 E 最小的阈值。人工贵的时候阈值应该放宽，人工便宜的时候应该收紧。

TraceCell 的默认成本模型（`policy.CostModel`）里人工复核是 8.0 rcu，多模态
升级复核是 3.0 rcu。这几个数字请按自己的实际情况改。

---

## 四、概率校准（可选）

阈值标完之后，你手上是一个"二元闸门"。如果还想要"这个区域有 92% 的概率
是对的"这样的数字，就再做一次保序回归：

```python
from tracecell.calibration import isotonic_regression, apply_isotonic

xs, ys = isotonic_regression(samples)
print(apply_isotonic(xs, ys, 0.8))   # 0.8 分实际对应的正确比例
```

保序回归保证校准后的概率随分数单调不减——这是它相对逻辑回归的好处：
不会凭空造出非单调的曲线。

---

## 五、注意事项

**1. 阈值是文档类型相关的。** 官方模型卡给的"法律文书免检率 85%、财务票据
56.2%、复杂文档 35.1%"是内部统计，不能直接套到你的业务上。

**2. 解析器换了，阈值要重标。** Jev 在 MiniCPM-OCR 1.0、GLM-OCR、
PaddleOCR-VL-1.6、Xiaomi-OCR-0 上的 AUROC 都在 91.9~93.4，但那是同一批
200 条样本上的结果。换解析器、换文档分布，都要重新标。

**3. 渲染配置要统一。** Jev 会把候选文本重新渲染成图像再比对。不同字体、
不同浏览器渲染出来的图不一样，分数会有轻微漂移。生产环境固定一套渲染配置。

**4. 阈值是会被监控的。** 上线后持续看两个指标：

- 自动通过率（人工工作量）
- 自动通过准确率（真实风险）

任一个偏离预期，回到这一步重新标。

---

## 六、把标定结果接回策略

```python
from tracecell.policy import BudgetPolicy

policy = BudgetPolicy()
policy.accept_threshold = 0.92      # 用标定出来的阈值
policy.residual_tolerance = 0.03    # 更保守
```

或者写成配置文件：

```json
{
  "accept_threshold": 0.92,
  "escalate_threshold": 0.55,
  "residual_tolerance": 0.03,
  "max_rounds": 3,
  "total_budget": 12.0,
  "cost_model": {
    "primary_parse": 1.0,
    "score": 0.3,
    "reparse": 1.2,
    "best_of_n": 0.4,
    "vlm_escalation": 3.0,
    "human_review": 8.0
  }
}
```

```python
policy = BudgetPolicy.from_dict(json.loads(Path("policy.json").read_text()))
```
