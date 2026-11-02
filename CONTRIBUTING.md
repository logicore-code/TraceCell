# 参与 TraceCell

## 开发环境

```bash
git clone https://gitcode.com/gcw_EkTYFH3T/TraceCell.git
cd TraceCell
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[web,dev]"
```

## 跑测试

```bash
pytest              # 全部
pytest -k budget    # 按名字筛
```

测试不依赖 GPU：模型调用全部用回放数据替换，测的是策略逻辑。

## 代码风格

- 行宽 100，`ruff check` 过一遍。
- 类型标注写全，公开函数要有 docstring，**解释为什么这么写**，不是复述
  代码在干什么。
- 中文注释和英文标识符混用是刻意的：注释给读代码的人看，标识符给工具看。
- 不要在注释里写"这个函数返回结果"这种废话。

## 提 PR

1. 从 `main` 切分支：`git checkout -b feature/xxx`
2. 改动尽量小而完整，一个 PR 做一件事。
3. 提交信息用 `type(scope): 摘要`，例如 `fix(budget): 修正预算耗尽后的降级路径`
4. 跑一遍 `pytest` 和 `ruff check`
5. 提 PR 到 `main`，描述里写清楚"改了什么"和"为什么"

## 改模型相关代码前先看

- `providers/jev.py` —— `verdict` 必须检查，这是官方文档反复强调的
- `risk.py` —— 字段置信度取支撑区域的最小值，不要改成平均值
- `extractors.py` —— 抽不到就留空，不要填占位值

这三条改错了不会立刻报错，但会让账本悄悄变错。

## 提 Issue

请附上：

- TraceCell 版本和模型版本
- 复现用的文档或图片
- 实际输出与期望输出
- `tracecell doctor` 的输出（如果涉及真实模型）
