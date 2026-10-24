# stock-agent

以 A 股金融分析为场景的 **ReAct（Reasoning + Acting）Agent** 项目，基于 LLM 实现 Thought → Action → Observation 循环，逐步调用工具完成财务对比、年报检索、股价分析等任务。

提供两种工程实现，可直接对比学习：

- **手写 Prompt 解析版**：`Thought / Action / Action Input` 格式约束 + 正则解析，推理过程完全可见
- **Function Calling 版**：通过 OpenAI 风格的 `tool_calls` 接口原生注册工具，格式由 API 保证

> 仅供学习研究使用，不构成任何投资建议。

## 功能特性

- ReAct 循环驱动（默认最多 10 步），支持 CLI 与 Web UI（SSE 流式展示每步推理）
- 5 个异构工具：
  | 工具 | 数据来源 | 用途 |
  |------|---------|------|
  | `company_lookup` | 静态字典 | 公司名 → 股票代码，防幻觉 |
  | `rag_search` | FAISS + DashScope Embedding | 年报原文语义检索（5 家公司 2021–2023 年报，10353 条向量） |
  | `financial_indicator` | AkShare | 近 3 年结构化财务指标 |
  | `stock_price` | AkShare | 历史股价与区间涨跌幅 |
  | `calculator` | 受限 Python eval | 四则运算、百分比、CAGR |
- 同一 Agent、同一问题，工具组合路径因模型自主决策而不同，便于观察推理行为

## 架构

```
用户问题
    │
    ▼
┌─────────────────────────────────────────────────────┐
│               ReAct 循环（最多 10 步）               │
│                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐   │
│  │  Thought  │──▶│  Action  │──▶│ Observation  │   │
│  │ LLM 推理  │   │ 工具调用  │   │  工具返回结果  │   │
│  └──────────┘   └──────────┘   └──────┬───────┘   │
│       ▲                                │           │
│       └────────────────────────────────┘           │
│                      循环                           │
└─────────────────────────────────────────────────────┘
    │ Final Answer
    ▼
Web UI 展示 / CLI 打印
```

## 目录结构

```
stock-agent/
├── src/
│   ├── tools.py                  # 5 个工具实现
│   ├── react_manual.py           # 手写 Prompt 解析版 ReAct
│   ├── react_function_calling.py # Function Calling 版 ReAct
│   ├── agent.py                  # 统一 CLI 入口（--mode manual/fc）
│   ├── evaluate.py               # 两种实现对比评估
│   └── serve.py                  # FastAPI 服务
├── vectorstore/
│   ├── faiss_index.bin           # FAISS 索引（1024 维，10353 条）
│   └── faiss_meta.json           # 向量 chunk 元数据
├── models/
│   └── bge-small-zh-v1.5/        # 本地 embedding 模型（备用）
├── evaluation/
│   └── compare_result.json       # 对比评估结果示例
├── index.html                    # Web UI（流式展示每步循环）
├── requirements.txt
├── .gitignore
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
export DASHSCOPE_API_KEY="sk-xxx"       # 必填，用于 LLM 推理和 RAG Embedding
export AGENT_MODEL="qwen-max"           # 可选，默认 qwen-max，可换 deepseek-v3 等
```

Windows PowerShell：

```powershell
$env:DASHSCOPE_API_KEY = "sk-xxx"
$env:AGENT_MODEL = "qwen-max"
```

### 3. 命令行运行

```bash
cd stock-agent/src

# 手写 Prompt 解析版（默认）
python agent.py --mode manual --question "茅台和五粮液2023年毛利率差多少？"

# Function Calling 版
python agent.py --mode fc --question "宁德时代2023年营业收入是多少？"

# 调整最大步数（默认 10）
python agent.py --mode manual --question "..." --max_steps 8
```

预期输出（手写版）：

```
[Step 1]
🧠 Thought: 需要先获取两家公司的股票代码...
🔧 Action:  company_lookup
   Input:   {"name": "贵州茅台"}
👁  Obs:     贵州茅台 的股票代码为 600519
...
✅ Final Answer:
茅台2023年毛利率91.96%，高于五粮液的75.79%，差16.17个百分点。
```

### 4. 启动 Web 服务

```bash
cd stock-agent/src
uvicorn serve:app --host 0.0.0.0 --port 8000
```

访问 http://localhost:8000 。

接口列表：

- `GET  /health` — 健康检查
- `POST /query/manual` — 手写版，流式 SSE 返回每步
- `POST /query/fc` — Function Calling 版，流式 SSE 返回每步

SSE 事件格式：

```json
{"type": "start",  "question": "...", "mode": "manual"}
{"type": "action", "step": 1, "thought": "...", "action": "company_lookup", "action_input": {...}, "observation": "..."}
{"type": "final",  "step": 6, "thought": "...", "answer": "..."}
{"type": "done"}
```

## 两种实现对比

| 维度 | 手写 Prompt 解析 | Function Calling API |
|------|----------------|----------------------|
| Thought 可见性 | 完全可见，正则解析 | 模型内部，不可见 |
| 格式稳定性 | 依赖 Prompt 工程，偶有漂移 | 原生结构化，格式稳定 |
| 代码量 | ~150 行核心逻辑 | ~80 行核心逻辑 |
| 可控性 | 高，可定制停止词和格式 | 低，依赖模型实现 |

停止词机制（手写版）：`stop=["Observation:"]`，让模型在调用工具前停止，由 Python 执行工具后再追加 Observation 继续对话。

### 对比评估

```bash
python evaluate.py --output ../evaluation/compare_result.json
```

评估问题集（5 题）：跨公司财务对比、跨年度趋势 + CAGR、单次年报检索、股价查询 + 涨跌幅计算、边界拒绝（预测 A 股走势应拒绝）。

## 作为模块调用

```python
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
sys.path.insert(0, "stock-agent/src")

from react_manual import run as manual_run

for step in manual_run("茅台2023年毛利率是多少？", max_steps=10):
    if step["type"] == "action":
        print(f"Step {step['step']}: {step['action']}({step['action_input']})")
    elif step["type"] == "final":
        print(f"Final: {step['answer']}")

# Function Calling 版接口完全一致
from react_function_calling import run as fc_run
```

工具单独调用：

```python
from tools import TOOLS_MAP

print(TOOLS_MAP["company_lookup"](name="贵州茅台"))
print(TOOLS_MAP["rag_search"](query="茅台2023年毛利率", top_k=3))
print(TOOLS_MAP["financial_indicator"](symbol="600519"))
print(TOOLS_MAP["stock_price"](symbol="600519", start_date="20230101", end_date="20231231"))
```

## FAQ

**Q：`rag_search` 报错 `assert d == self.d`**
A：Embedding 维度不匹配。索引由 DashScope `text-embedding-v3`（1024 维）构建，确认 `DASHSCOPE_API_KEY` 已设置，查询向量同样来自该 API。

**Q：`financial_indicator` 很慢（20 秒以上）**
A：正常，AkShare `stock_financial_abstract` 需拉取历史财务数据。可用类似接力缓存的思路优化同一公司多次调用。

**Q：手写版某步 Thought 为空**
A：正常现象，部分模型在中间步骤会省略 Thought 直接输出 Action，解析器已容忍，不影响循环执行。

**Q：想换 DeepSeek-V3**
A：修改环境变量 `AGENT_MODEL="deepseek-v3"`；若走 DeepSeek 官方接口，将 `tools.py` 与 `react_manual.py` 中的 `base_url` 改为 `https://api.deepseek.com/v1`。

**Q：Web UI 显示"请求失败"**
A：检查 `uvicorn` 是否正常启动，访问 `http://localhost:8000/health` 确认服务状态。

## 免责声明

本项目用于学习 Agent 技术范式与工程实践，数据来源于 AkShare 公开接口及公开年报，不构成投资建议。

## License

MIT
