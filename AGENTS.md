# AGENTS.md

A股 ReAct Agent 工程（刻意不用 LangChain/AutoGen）：手写 Prompt 解析版与 Function Calling 版两种实现共用同一套工具与评估脚本。README.md 是主要文档，但以下坑点需注意。

## 运行环境

- Python 3.10，venv 在仓库根 `.venv/`（Windows）；`pip install -r requirements.txt` 后用 `.venv\Scripts\python.exe`。
- **文档语言/代码注释/研报输出均为中文**，沿用此约定。
- `DASHSCOPE_API_KEY` 必填（LLM 推理 + RAG embedding）；换服务商用 `AGENT_API_KEY` / `AGENT_BASE_URL`（OpenAI 兼容）+ `AGENT_MODEL`（默认 `qwen3.8-max`）。
- **本地 LiteLLM 网关（`gateway/`，推荐路径）**：业务侧设 `AGENT_BASE_URL=http://localhost:4000/v1` + `AGENT_API_KEY=<master key>` + `AGENT_EMBEDDING_BASE_URL=http://localhost:4000/v1` 即可（零代码改动）；`DASHSCOPE_API_KEY` 只给网关进程。配置在 `gateway/config.yaml`，`qwen3.8-max`/`text-embedding-v3`/`vl-model`（→`dashscope/qwen3-vl-plus`，供 image_analysis 工具）均为 alias，embedding 一旦换了模型/维度，`rag_search` 立即报 `assert d == self.d`。启动与验证见 `gateway/README.md`。Admin UI（`/ui`）登录**必须连接 PostgreSQL**，无 DB 时正确密码也报 `Not connected to DB!`——不是密码错。
- **网关启动统一走 `powershell -ExecutionPolicy Bypass -File gateway\start-gateway.ps1`**：裸 `litellm` 不在 PATH（在 `.venv\Scripts\`，不激活 venv 敲了没反应）；脚本里 master key 持久化到 `gateway/.master_key`（已 git-ignore）；脚本运行时设 `LITELLM_LOCAL_MODEL_COST_MAP=True`（本机出 raw.githubusercontent.com 被 RST，不设会卡启动）与 `NO_PROXY=dashscope.aliyuncs.com`。`gateway/` 下可执行文件类文件（config.yaml、start-gateway.ps1）必须纯 ASCII：LiteLLM 吃 GBK 读 YAML、PowerShell 5.1 用 GBK 解析无 BOM 的 ps1，非 ASCII 直接崩/解析错。
- 所有命令在 `src/` 目录下运行，不能从仓库根运行：模块间是平铺导入（`from tools import ...`），`agent.py` 自己不补 sys.path（只有 `serve.py`/`evaluate.py` 有 insert）。Uvicorn 同理：`cd src; uvicorn serve:app --port 8000`。
- faiss + numpy 在 Windows 上会触发 `KMP_DUPLICATE_LIB_OK` 冲突，各入口模块已 `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`，新入口也要加。

## 验证方式（没有 tests / lint / typecheck）

改动的行为验证只能靠真实运行（需 API key + 网络走 DashScope/AkShare）：

```powershell
cd src
python agent.py --mode manual --question "茅台2023年毛利率"   # mode: manual|fc
python agent.py --mode fc --question "解读这张图" --image D:\tmp\kline.png   # 附图分析（视觉工具）
python agent.py --interactive                                # 多轮对话
python evaluate.py --output ../evaluation/compare_result.json # 双实现对比（5 题，强制冷缓存）
```

- 单工具冒烟：`from tools import TOOLS_MAP; TOOLS_MAP["company_lookup"](name="贵州茅台")`。
- `financial_indicator` 20s+ 慢是正常现象（AkShare 拉历史财务），不用修。

## 缓存陷阱（最高频坑）

- 8 个网络型工具带进程内 TTL 缓存（最长 1h）。**改了工具代码结果不变** ≠ 改错：用 `TOOL_CACHE_DISABLE=1` 环境变量 / CLI `--no-cache` / `POST /cache/clear` 旁路。`evaluate.py` 自动禁用缓存。
- 错误结果一律不缓存；缓存故障一律降级直连——不要把"缓存返回旧数据"当 bug 修。

## 工具层约定（tools.py）

- 新增/修改联网工具必须保持三层结构：`@_with_cache` → `@_with_retry` → 原函数。retry 装饰器靠 `functools.wraps` 保留 `__name__` 匹配 TTL 表（tools.py:83 `_CACHE_TTL`）与缓存 key，删掉 wraps 会静默破坏缓存。
- 工具返回统一字符串；网络异常返回错误描述文本而不是抛异常（LLM 直接消费结果）。

## RAG / 模型

- `rag_search` 的 FAISS 索引是 1024 维、DashScope `text-embedding-v3` 建的（`vectorstore/`）。报 `assert d == self.d` = embedding 提供商不匹配，查 `DASHSCOPE_API_KEY` / `AGENT_EMBEDDING_BASE_URL`。
- `models/bge-small-zh-v1.5/` 只是备用，NOT 当前使用的 embedding 来源。

## 多模态（v2.4）

- `image_analysis(image, question)`：独立 VL 客户端，默认走网关 alias `vl-model`（→`dashscope/qwen3-vl-plus`）；`AGENT_VL_BASE_URL`/`AGENT_VL_MODEL`/`AGENT_VL_API_KEY` 可覆写直连 DashScope。**网关无 DB 模式只认 master key**：带其他 key 的请求报 `400 No connected db.`——serve/agent 进程务必设 `AGENT_API_KEY=<gateway/.master_key>`（实测踩坑：uvicorn 不带此 env 时视觉调用全部失败）。
- 本地图片自动转 base64 data URI（≤10MB，png/jpg/jpeg/bmp/webp/gif）；http(s) URL 直接透传给 VL 模型。
- `price_chart(symbol, start_date, end_date)`：matplotlib Agg 后端出 PNG 到 `reports/charts/`，中文字体依赖 Windows 自带的 Microsoft YaHei/SimHei；observation 返回的相对路径会被 `report.py` 解析为研报「图表」章节（无图时编号保持三/四）。
- 图片入口：CLI `--image`（可多次）；Web 走 `POST /upload`（multipart → `uploads/`，uuid 命名，后缀+10MB 校验）→ 前端把路径放进 QueryRequest.images → serve.py 注入问题文本，两个引擎零改动。
- 主推理模型仍为纯文本 qwen3.8-max，多模态走工具路线；"主模型直看图"（content 数组注入）仅在文档记为进阶实验，未实装。

## 产物

- `reports/`、`uploads/` 在 .gitignore 中，是运行时自动生成的产物，不要提交。
- 调试 Web 服务：`/health`（含缓存状态）、`/cache/stats`、`POST /cache/clear`、`POST /upload`（图片上传）；根路径托管 `../index.html`。
