"""
研报生成器：把一次 ReAct 分析的完整过程导出为结构化 Markdown 研报

供两个场景调用：
  1. serve.py  — 每次问答结束后自动生成，并提供 /report/ 下载接口
  2. agent.py  — --save-report 时 CLI 保存到 reports/ 目录

设计原则：
  - 纯 Python 拼装，不额外调用 LLM（结论直接来自 Final Answer）
  - 报告包含全部推理步骤，便于复盘 Agent 的工具调用路径
"""

import re
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
REPORT_DIR = BASE_DIR / "reports"

_DISCLAIMER = (
    "> 免责声明：本报告由 AI Agent 自动生成，数据来自 AkShare 公开接口及公开年报，"
    "可能存在延迟或错误，仅供学习研究使用，不构成任何投资建议。"
)


def _merge_tool_sources(steps: list[dict]) -> list[str]:
    """从全部 Observation 中提取用到的数据来源"""
    sources: dict[str, str] = {}
    for s in steps:
        if s.get("type") != "action":
            continue
        action, obs = s.get("action", ""), s.get("observation", "")
        if not isinstance(obs, str):
            continue
        if action in ("rag_search",):
            for m in re.findall(r"来源：(\d{6}) (\d{4})年报 第(\d+)页", obs):
                sources.setdefault(f"年报 {m[0]}-{m[1]}", "FAISS 语义检索（DashScope embedding）")
        elif action in ("financial_indicator",):
            sources.setdefault("财务指标", "AkShare stock_financial_abstract")
        elif action in ("stock_price",):
            sources.setdefault("历史行情", "AkShare stock_zh_a_hist")
        elif action in ("technical_indicator",):
            sources.setdefault("技术指标", "AkShare stock_zh_a_hist + pandas 计算（MA/RSI/MACD）")
        elif action in ("news_search",):
            sources.setdefault("新闻资讯", "AkShare stock_news_em（东方财富）")
        elif action in ("industry_peers",):
            sources.setdefault("行业成分", "AkShare stock_board_industry_cons_em（东方财富）")
        elif action in ("image_analysis",):
            sources.setdefault("图像理解", "DashScope qwen3-vl-plus 视觉模型（用户提供的图片）")
        elif action in ("price_chart",):
            sources.setdefault("走势图表", "AkShare stock_zh_a_hist + matplotlib（收盘价/MA20）")
    return [f"- **{k}**：{v}" for k, v in sources.items()] or ["- 本轮分析未调用外部数据工具"]


_BASE_LIMITATIONS = [
    "财务与行情数据存在披露延迟，可能与最新公告不一致",
    "年报语义检索仅覆盖 5 家公司 2021-2023 年报，范围外的公司无法溯源",
    "技术指标基于历史价格计算，对未来走势无预测能力",
    "结论由 LLM 归纳生成，可能存在理解偏差，请交叉验证关键数字",
]


def _collect_chart_paths(steps: list[dict]) -> list[str]:
    """从 price_chart 工具的 Observation 中提取图片相对路径（去重保序）"""
    paths: list[str] = []
    for s in steps:
        if s.get("type") != "action" or s.get("action") != "price_chart":
            continue
        m = re.search(r"走势图已生成: (\S+\.png)", str(s.get("observation", "")))
        if m and m.group(1) not in paths:
            paths.append(m.group(1))
    return paths


def build_report_md(question: str, mode: str, model: str,
                    final_answer: str, steps: list[dict],
                    elapsed_s: float | None = None) -> str:
    """把一次 ReAct 分析拼装成 Markdown 研报"""
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M")
    mode_text = "手写 Prompt 解析" if mode == "manual" else "Function Calling"

    rows = []
    for s in steps:
        if s.get("type") != "action":
            continue
        obs = str(s.get("observation", "")).replace("\n", " ")[:160]
        args = str(s.get("action_input", {})).replace("\n", " ")
        thought = str(s.get("thought", ""))[:60]
        dur = s.get("duration_s")
        rows.append(f"| {s.get('step')} | {thought} | {s.get('action')} | {args} | {dur}s | {obs} |")

    process_table = (
        "| Step | Thought | Action | Input | 耗时 | Observation（截断） |\n"
        "|------|---------|--------|-------|------|---------------------|\n" + "\n".join(rows)
    ) if rows else "（本轮未能完成工具调用）"

    finish_line = (
        f"- 分析完成，用时 **{elapsed_s:.1f} 秒**"
        if elapsed_s is not None
        else f"- 分析完成，共 **{len(rows)}** 步工具调用"
    )
    sources = "\n".join(_merge_tool_sources(steps))

    # 图表附录：本轮调用过 price_chart 时新增「三、图表」，后续章节顺延编号
    chart_paths = _collect_chart_paths(steps)
    limitations = list(_BASE_LIMITATIONS)
    if chart_paths:
        charts_md = "\n\n".join(f"![走势图 {i + 1}]({p})" for i, p in enumerate(chart_paths))
        chart_block = f"## 三、图表\n\n{charts_md}\n\n---\n\n"
        source_title, limit_title = "## 四、数据来源", "## 五、局限性说明"
        limitations.insert(1, "走势图基于前复权收盘价与MA20均线，仅作可视化参考")
    else:
        chart_block = ""
        source_title, limit_title = "## 三、数据来源", "## 四、局限性说明"
    limit_md = "\n".join(f"- {item}" for item in limitations)

    return f"""# A股分析研报

**研究问题**：{question}

**生成信息**：{ts} ｜ 模型：{model} ｜ 实现：{mode_text} ｜ {finish_line}

---

## 一、结论

{final_answer}

---

## 二、分析过程（Agent 推理链）

{process_table}

---

{chart_block}{source_title}

{sources}

---

{limit_title}

{limit_md}

---

{_DISCLAIMER}
"""


def save_report(md: str, prefix: str = "report") -> str:
    """保存研报到 reports/ 目录，返回相对项目根的文件路径字符串"""
    REPORT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = REPORT_DIR / f"{prefix}_{ts}.md"
    path.write_text(md, encoding="utf-8")
    logger_name = __name__
    import logging
    logging.getLogger(logger_name).info(f"研报已保存: {path}")
    return f"{path.relative_to(BASE_DIR)}"
