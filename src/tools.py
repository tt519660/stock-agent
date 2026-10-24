"""
8个工具的统一实现，供 ReAct Agent 调用

教学重点：
  1. 工具异构性：语义检索 / 结构化数据 / 实时行情 / 技术指标 / 新闻 / 行业成分 / 计算 / 静态映射
  2. 每个工具返回统一格式的字符串，便于 LLM 消费
  3. 所有网络请求做异常保护，降级返回错误描述而非抛出

使用方式：
  from tools import TOOLS_MAP
  result = TOOLS_MAP["rag_search"](query="茅台2023年营收")

注意：
  rag_search 使用 DashScope text-embedding-v3（1024维）与建索引时保持一致，
  需要设置 DASHSCOPE_API_KEY 环境变量。
"""

import os
import json
import math
import time
import logging
from pathlib import Path
from typing import Any

import numpy as np
import faiss
from openai import OpenAI

logger = logging.getLogger(__name__)

# ── 网络重试保护 ──────────────────────────────────────────────────────────────
# AkShare 底层走 requests，会读取 Windows 系统代理。代理偶发抖动时自动重试，
# 且在第 2 次起临时绕过代理直连（NO_PROXY="*"），结束后恢复原环境。

_RETRY_KEYWORDS = ("Timeout", "Connection", "Remote", "ProxyError", "Retrying", "Max retries")

def _with_retry(times: int = 3):
    def deco(fn):
        def wrapper(*args, **kwargs):
            original_no_proxy = os.environ.get("NO_PROXY")   # 可能为 None
            last = None
            for attempt in range(1, times + 1):
                # 之后轮次绕过系统代理直连（应对代理半死不活的场景）
                direct = attempt >= 2
                if direct and original_no_proxy != "*":
                    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "*"
                    logger.warning(f"{fn.__name__} 尝试绕过系统代理直连...")
                try:
                    result = str(fn(*args, **kwargs))
                except Exception as e:
                    result = f"{fn.__name__} 执行出错: {e}"
                hit = any(k in result for k in _RETRY_KEYWORDS)
                if not hit:
                    return result
                last = result
                time.sleep(min(2 ** attempt, 4))
            # 恢复代理环境
            if original_no_proxy is None:
                os.environ.pop("NO_PROXY", None)
                os.environ.pop("no_proxy", None)
            else:
                os.environ["NO_PROXY"] = os.environ["no_proxy"] = original_no_proxy
            # 面向 LLM 的简洁错误（超长 URL 会污染上下文）
            return f"{fn.__name__} 网络请求失败：系统代理和直连均不可用，可能是网络波动。请告知用户稍后重试。"
        return wrapper
    return deco

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# ── Embedding 客户端（与建索引时相同的模型）────────────────────────────────────
_embed_client = OpenAI(
    api_key=os.getenv("AGENT_API_KEY") or os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv(
        "AGENT_EMBEDDING_BASE_URL",
        os.getenv("AGENT_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    ),
)
EMBED_MODEL = "text-embedding-v3"

# ── 公司映射（名称 → 股票代码） ────────────────────────────────────────────────
COMPANY_MAP = {
    "贵州茅台": "600519",
    "茅台":     "600519",
    "五粮液":   "000858",
    "宁德时代": "300750",
    "中国平安": "601318",
    "平安":     "601318",
    "海康威视": "002415",
    "海康":     "002415",
}

# ── RAG 工具（懒加载，首次调用时初始化索引） ─────────────────────────────────
_faiss_index = None
_faiss_meta  = None

def _load_rag():
    global _faiss_index, _faiss_meta
    if _faiss_index is not None:
        return
    logger.info("加载 FAISS 索引...")
    _faiss_index = faiss.read_index(str(VECTORSTORE_DIR / "faiss_index.bin"))
    with open(VECTORSTORE_DIR / "faiss_meta.json", encoding="utf-8") as f:
        _faiss_meta = json.load(f)
    logger.info(f"FAISS 就绪，共 {_faiss_index.ntotal} 条向量")


def _embed_query(text: str) -> np.ndarray:
    """调用 DashScope text-embedding-v3 对查询文本编码（与建索引时保持一致）"""
    resp = _embed_client.embeddings.create(model=EMBED_MODEL, input=[text])
    vec  = np.array(resp.data[0].embedding, dtype="float32")
    vec  = vec / np.linalg.norm(vec)   # 归一化，与建索引时一致
    return vec.reshape(1, -1)


def tool_rag_search(query: str, top_k: int = 5) -> str:
    """在年报 FAISS 索引中语义检索，返回最相关的文本段落"""
    try:
        _load_rag()
        vec = _embed_query(query)
        scores, indices = _faiss_index.search(vec, top_k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < 0:
                continue
            meta = _faiss_meta[idx]
            results.append(
                f"[{rank}] 来源：{meta.get('stock_code','')} {meta.get('year','')}年报 "
                f"第{meta.get('page_num','')}页 (相关度:{score:.3f})\n{meta['content']}"
            )
        return "\n\n".join(results) if results else "未检索到相关内容"
    except Exception as e:
        return f"rag_search 执行出错: {e}"


# ── 公司映射工具 ──────────────────────────────────────────────────────────────

def tool_company_lookup(name: str) -> str:
    """将公司中文名转换为 A 股股票代码"""
    code = COMPANY_MAP.get(name.strip())
    if code:
        return f"{name} 的股票代码为 {code}"
    candidates = [k for k in COMPANY_MAP if name in k]
    if candidates:
        return "未精确匹配，相似公司：" + "、".join(f"{k}({COMPANY_MAP[k]})" for k in candidates)
    supported = "、".join(COMPANY_MAP.keys())
    return f"未找到 '{name}'，当前支持：{supported}"


# ── 计算器工具 ────────────────────────────────────────────────────────────────

_SAFE_NAMES = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
_SAFE_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})

def tool_calculator(expr: str) -> str:
    """安全计算数学表达式，支持四则运算和 math 模块函数"""
    try:
        result = eval(expr, {"__builtins__": {}}, _SAFE_NAMES)  # noqa: S307
        return str(round(float(result), 6))
    except Exception as e:
        return f"计算出错: {e}，表达式: {expr}"


# ── AkShare：实时财务指标 ─────────────────────────────────────────────────────

@_with_retry()
def tool_financial_indicator(symbol: str) -> str:
    """获取 A 股近3年关键财务指标：营收/净利润/毛利率/ROE/资产负债率等"""
    try:
        import akshare as ak
        df = ak.stock_financial_abstract(symbol=symbol)
        if df is None or df.empty:
            return f"未获取到 {symbol} 的财务指标数据"

        # 取最近3个年报期（列名格式 YYYYMMDD，筛选 1231 结尾的年报）
        date_cols = [c for c in df.columns if str(c).endswith("1231")][:3]
        if not date_cols:
            date_cols = df.columns[2:5].tolist()

        # 感兴趣的指标行名
        target_rows = ["归母净利润", "营业总收入", "毛利率", "净利率",
                       "净资产收益率", "资产负债率", "每股收益"]
        lines = [f"股票代码: {symbol}，数据截至最近三个年报"]
        for _, row in df.iterrows():
            label = str(row.get("指标", ""))
            if any(t in label for t in target_rows):
                vals = []
                for col in date_cols:
                    v = row.get(col)
                    try:
                        v = f"{float(v):.4g}"
                    except (TypeError, ValueError):
                        v = str(v)
                    vals.append(f"{col[:4]}年: {v}")
                lines.append(f"  {label}: " + " | ".join(vals))

        return "\n".join(lines) if len(lines) > 1 else f"{symbol} 未找到关键财务指标行"
    except Exception as e:
        return f"financial_indicator 执行出错: {e}"


# ── AkShare：历史股价 ─────────────────────────────────────────────────────────

@_with_retry()
def tool_stock_price(symbol: str, start_date: str, end_date: str) -> str:
    """
    获取 A 股历史股价及区间涨跌幅
    日期格式：YYYYMMDD，例如 20230101
    """
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if df is None or df.empty:
            return f"未获取到 {symbol} 在 {start_date}~{end_date} 的行情数据"

        first_close = float(df.iloc[0]["收盘"])
        last_close  = float(df.iloc[-1]["收盘"])
        high        = float(df["最高"].max())
        low         = float(df["最低"].min())
        change_pct  = (last_close - first_close) / first_close * 100

        return (
            f"股票代码: {symbol}，区间: {start_date}~{end_date}\n"
            f"  区间起始收盘价: {first_close:.2f} 元\n"
            f"  区间末尾收盘价: {last_close:.2f} 元\n"
            f"  区间最高价: {high:.2f} 元\n"
            f"  区间最低价: {low:.2f} 元\n"
            f"  区间涨跌幅: {change_pct:+.2f}%"
        )
    except Exception as e:
        return f"stock_price 执行出错: {e}"


# ── AkShare：技术指标（MACD / RSI / 均线，纯 pandas 计算） ────────────────────

@_with_retry()
def tool_technical_indicator(symbol: str) -> str:
    """
    获取 A 股最近 120 个交易日行情并计算技术指标：
    MA5/MA10/MA20/MA60、RSI14、MACD(DIF/DEA/HIST)，并给出金叉/死叉提示
    """
    try:
        from datetime import datetime, timedelta
        import akshare as ak

        end   = datetime.now()
        start = end - timedelta(days=300)   # 留足回看天数，保证 MA60 与 MACD 收敛
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
        if df is None or df.empty:
            return f"未获取到 {symbol} 的行情数据"

        close = df["收盘"].astype(float).reset_index(drop=True)

        # 均线
        ma5  = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean()

        # RSI14（Wilder 平滑，对涨跌幅全为0的情况做除零保护）
        delta = close.diff()
        gain  = delta.ewm(alpha=1/14, adjust=False).mean().clip(lower=0)
        loss  = (-delta).ewm(alpha=1/14, adjust=False).mean().clip(lower=0)
        denom = gain + loss
        rsi14 = np.where(denom > 0, 100 * gain / denom, 50.0)

        # MACD（EMA12-EMA26 → DIF；DEA=DIF 的 EMA9；BAR=2*(DIF-DEA)）
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        hist = (dif - dea) * 2

        last = len(close) - 1
        last_date = str(df.iloc[last]["日期"])
        last_close = float(close.iloc[last])

        # 判断 DIF 上穿/下穿 DEA 的位置关系
        cross = ""
        if dif.iloc[last] > dea.iloc[last] and dif.iloc[last-1] <= dea.iloc[last-1]:
            cross = " ▲ 金叉"
        elif dif.iloc[last] < dea.iloc[last] and dif.iloc[last-1] >= dea.iloc[last-1]:
            cross = " ▼ 死叉"

        rsi_val = float(rsi14.iloc[last])
        rsi_hint = "超买" if rsi_val >= 80 else "超卖" if rsi_val <= 20 else "中性"

        return (
            f"股票代码: {symbol}，数据截至 {last_date}（收盘价 {last_close:.2f} 元）\n"
            f"  MA5:  {ma5.iloc[last]:.2f}\n"
            f"  MA10: {ma10.iloc[last]:.2f}\n"
            f"  MA20: {ma20.iloc[last]:.2f}\n"
            f"  MA60: {ma60.iloc[last]:.2f}\n"
            f"  RSI14: {rsi_val:.1f}（{rsi_hint}，>80超买/<20超卖）\n"
            f"  MACD DIF: {dif.iloc[last]:.3f} | DEA: {dea.iloc[last]:.3f} | BAR: {hist.iloc[last]:.3f}{cross}\n"
            f"  DIF 与 DEA 关系: {'DIF>DEA 多头' if dif.iloc[last] > dea.iloc[last] else 'DIF<DEA 空头'}"
        )
    except Exception as e:
        return f"technical_indicator 执行出错: {e}"


# ── AkShare：个股新闻 ─────────────────────────────────────────────────────────

@_with_retry()
def tool_news_search(symbol: str, top_k: int = 5) -> str:
    """获取个股最近的新闻资讯（来源：东方财富），返回标题/时间/摘要"""
    try:
        import akshare as ak
        df = ak.stock_news_em(symbol=symbol)
        if df is None or df.empty:
            return f"未获取到 {symbol} 的新闻"
        lines = [f"{symbol} 最近新闻（来源：东方财富）:"]
        for i, (_, row) in enumerate(df.head(top_k).iterrows(), 1):
            title  = str(row.get("新闻标题", "")).strip()
            pub    = str(row.get("发布时间", "")).strip()
            src    = str(row.get("文章来源", "")).strip() or str(row.get("新闻来源", "")).strip()
            content = str(row.get("新闻内容", "")).strip()[:120]
            lines.append(f"[{i}] {title}（{pub}，{src}）\n    {content}")
        return "\n".join(lines)
    except Exception as e:
        return f"news_search 执行出错: {e}"


# ── AkShare：同行业可比公司 ───────────────────────────────────────────────────

@_with_retry()
def tool_industry_peers(symbol: str, top_k: int = 10) -> str:
    """查询个股所属行业及行业内主要成分股（按总市值排序），支持后续横向对比"""
    try:
        import akshare as ak
        info = ak.stock_individual_info_em(symbol=symbol)
        if info is None or info.empty:
            return f"未获取到 {symbol} 的公司信息"
        industry = ""
        name = ""
        for _, row in info.iterrows():
            item, value = str(row.get("item", "")), str(row.get("value", ""))
            if item == "行业":
                industry = value
            elif item == "股票简称":
                name = value
        if not industry:
            return f"{symbol} 未返回行业字段"

        df = ak.stock_board_industry_cons_em(symbol=industry)
        if df is None or df.empty:
            return f"{name}({symbol}) 所属板块「{industry}」，但未获取到成分股列表"

        lines = [f"{name}({symbol}) 所属板块「{industry}」，主流成分股（按总市值排序,{top_k}个）:"]
        cap_key = next((k for k in df.columns if "总市值" in k), None)
        df_use = df.sort_values(cap_key, ascending=False) if cap_key else df
        for i, (_, row) in enumerate(df_use.head(top_k).iterrows(), 1):
            peer_code = str(row.get("代码", ""))
            peer_name = str(row.get("名称", ""))
            extra = ""
            if cap_key:
                try:
                    extra = f"（市值 {float(row[cap_key])/1e8:.0f} 亿元）"
                except (TypeError, ValueError):
                    pass
            lines.append(f"[{i}] {peer_name}({peer_code}){extra}")
        return "\n".join(lines)
    except Exception as e:
        return f"industry_peers 执行出错: {e}"


# ── 统一工具注册表 ─────────────────────────────────────────────────────────────

TOOLS_MAP: dict[str, Any] = {
    "rag_search":           tool_rag_search,
    "company_lookup":       tool_company_lookup,
    "calculator":           tool_calculator,
    "financial_indicator":  tool_financial_indicator,
    "stock_price":          tool_stock_price,
    "technical_indicator":  tool_technical_indicator,
    "news_search":          tool_news_search,
    "industry_peers":       tool_industry_peers,
}

# Function Calling 版所需的 JSON Schema 描述
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "在5家A股公司（贵州茅台/五粮液/宁德时代/中国平安/海康威视）2021-2023年年报中语义检索，适合查询定性描述、战略规划、风险因素、管理层讨论等文本内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题，尽量具体，如'茅台2023年毛利率'"},
                    "top_k": {"type": "integer", "description": "返回结果数量，默认5", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "company_lookup",
            "description": "将公司中文名称转换为A股股票代码，在调用 financial_indicator 或 stock_price 前必须先用此工具获取代码",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "公司中文名，如'贵州茅台'"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "安全计算数学表达式，支持加减乘除、幂运算、math模块函数（sqrt/log/pow等），用于财务计算如增长率、PE、差值等",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {"type": "string", "description": "数学表达式，如 '(747 - 524) / 524 * 100'"},
                },
                "required": ["expr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "financial_indicator",
            "description": "获取A股近3年关键财务指标（营收/净利润/毛利率/ROE/资产负债率等），适合做跨年对比或与年报数据交叉验证",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，如'600519'"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stock_price",
            "description": "获取A股历史股价及区间涨跌幅，日期格式为YYYYMMDD",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol":     {"type": "string", "description": "股票代码，如'600519'"},
                    "start_date": {"type": "string", "description": "起始日期，格式YYYYMMDD，如'20230101'"},
                    "end_date":   {"type": "string", "description": "结束日期，格式YYYYMMDD，如'20231231'"},
                },
                "required": ["symbol", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "technical_indicator",
            "description": "获取A股技术指标：MA5/MA10/MA20/MA60均线、RSI14、MACD(DIF/DEA/BAR)最新值，并提示金叉/死叉信号，适合回答技术面/趋势/买卖信号类问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，如'600519'"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "news_search",
            "description": "获取某只A股最近的新闻资讯（东方财富来源），适合查询公司近况、舆情、事件驱动等问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，如'600519'"},
                    "top_k":  {"type": "integer", "description": "返回新闻条数，默认5"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "industry_peers",
            "description": "查询A股个股所属行业及行业内主流成分股（按总市值排序），用于横向对比、行业分析场景",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "股票代码，如'600519'"},
                    "top_k":  {"type": "integer", "description": "返回成分股数量，默认10"},
                },
                "required": ["symbol"],
            },
        },
    },
]
