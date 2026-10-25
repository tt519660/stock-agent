"""
FastAPI HTTP 服务，提供流式 SSE 接口给 Web UI

接口：
  POST /query/manual  - 手写版 ReAct，流式返回每步
  POST /query/fc      - Function Calling 版，流式返回每步
  GET  /health        - 健康检查

使用方式：
  uvicorn serve:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import json
import time
import uuid
import logging
import asyncio
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 短期记忆：会话管理 ─────────────────────────────────────────────────────────
_sessions: dict[str, list] = {}
_sessions_lock = threading.Lock()

# ── 打断支持：query_id → cancel_event ─────────────────────────────────────────
_cancel_flags: dict[str, threading.Event] = {}
_cancel_lock = threading.Lock()


def _register_cancel_flag(query_id: str) -> threading.Event:
    ev = threading.Event()
    with _cancel_lock:
        _cancel_flags[query_id] = ev
    return ev


def _release_cancel_flag(query_id: str) -> None:
    with _cancel_lock:
        _cancel_flags.pop(query_id, None)


# ── 预加载 FAISS（启动时执行一次）────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("预加载 FAISS 索引和 Embedding 模型...")
    from tools import _load_rag
    await asyncio.to_thread(_load_rag)
    logger.info("预加载完成，服务就绪")
    yield


app = FastAPI(title="Financial Agent", lifespan=lifespan)


# ── 请求/响应模型 ─────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question:   str
    max_steps:  int = 10
    session_id: str | None = None  # 不传则开启新会话


# ── SSE 流式生成器 ────────────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _friendly_llm_error(e: Exception) -> str:
    """把 OpenAI/DashScope 常见报错翻译为用户能看懂的话"""
    text = str(e)
    if "Arrearage" in text:
        return "DashScope（阿里云百炼）账户欠费，LLM 无法调用。请到百炼控制台充值，或通过 AGENT_BASE_URL/AGENT_API_KEY 环境变量切换到其他服务商（如 DeepSeek）。"
    if "Access denied" in text or "InvalidApiKey" in text:
        return "API Key 无效或被拒绝访问，请检查 DASHSCOPE_API_KEY（或 AGENT_API_KEY）是否正确、账户状态是否正常。"
    if "Throttling" in text or "RateLimit" in text:
        return "请求被限流（调用过快或同时并发过高），请稍后再试。"
    if "insufficient" in text.lower() and "quota" in text.lower():
        return "调用配额耗尽，请前往对应模型平台检查额度。"
    if "timed out" in text.lower() or "connection" in text.lower():
        return "连接 LLM 服务超时，请检查网络或代理后重试。"
    return text[:300]


async def _stream_react(question: str, max_steps: int, mode: str,
                         session_id: str | None = None):
    """
    同步生成器（react_run）在独立线程中逐步执行，
    每产出一步通过 asyncio.Queue 传递给异步 SSE 生成器，
    实现真正的边思考边推送。

    支持 session_id：同一会话的多轮对话累积短期记忆。
    """
    if mode == "manual":
        from react_manual import run as react_run
    else:
        from react_function_calling import run as react_run

    # ── 会话管理 ────────────────────────────────────────────────────────
    is_new = False
    if session_id is None:
        session_id = uuid.uuid4().hex[:12]
        is_new = True

    with _sessions_lock:
        if session_id not in _sessions:
            _sessions[session_id] = []
            is_new = True
        history = _sessions[session_id]

    # 新会话用 reset=True 初始化 system prompt，后续对话追加
    reset = is_new

    # ── 本次查询的取消标志 ──────────────────────────────────────────────
    query_id = uuid.uuid4().hex[:8]
    cancel_event = _register_cancel_flag(query_id)

    item_queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _worker():
        try:
            for step_data in react_run(question, max_steps=max_steps,
                                       reset=reset, history=history,
                                       cancel_event=cancel_event):
                item_queue.put_nowait(step_data)
        except Exception as e:
            # LLM 调用等异常（欠费/网络/限流）不能让 SSE 静默挂掉，转为 error 事件
            logger.exception("ReAct 执行异常")
            item_queue.put_nowait({
                "type": "error",
                "answer": f"Agent 执行失败: {_friendly_llm_error(e)}",
            })
        finally:
            item_queue.put_nowait(_SENTINEL)

    try:
        yield _sse({
            "type": "start", "question": question, "mode": mode,
            "session_id": session_id, "query_id": query_id,
        })

        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, _worker)

        steps: list = []
        final_answer: str | None = None
        start_ts = time.time()

        while True:
            step_data = await item_queue.get()
            if step_data is _SENTINEL:
                break
            steps.append(step_data)
            if step_data.get("type") == "final":
                final_answer = step_data.get("answer", "")
            yield _sse(step_data)

        # ── 分析结束后自动生成 Markdown 研报 ──────────────────────────────
        if final_answer:
            try:
                from report import build_report_md, save_report
                elapsed = time.time() - start_ts
                md = build_report_md(
                    question=question,
                    mode=mode,
                    model=os.getenv("AGENT_MODEL", "qwen3.8-max"),
                    final_answer=final_answer,
                    steps=steps,
                    elapsed_s=elapsed,
                )
                report_file = await asyncio.to_thread(save_report, md)
                yield _sse({"type": "report", "file": report_file, "session_id": session_id})
            except Exception as e:   # 研报生成失败不影响主流程
                logger.warning(f"研报生成失败: {e}")
                yield _sse({"type": "report", "file": None, "error": str(e)})

        yield _sse({"type": "done", "session_id": session_id})

    except (asyncio.CancelledError, GeneratorExit):
        # 客户端断开（页面刷新/前端 abort）：级联触发打断，避免后台白跑
        logger.info(f"客户端断开, 级联取消 {query_id}")
        raise
    finally:
        cancel_event.set()           # 正常结束与否都置位；worker 已结束时无副作用
        _release_cancel_flag(query_id)


# ── 路由 ──────────────────────────────────────────────────────────────────────
@app.post("/query/manual")
async def query_manual(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "manual",
                       session_id=req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/query/fc")
async def query_fc(req: QueryRequest):
    return StreamingResponse(
        _stream_react(req.question, req.max_steps, "fc",
                       session_id=req.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    from tools import cache_stats as _stats
    return {
        "status": "ok",
        "model": os.getenv("AGENT_MODEL", "qwen3.8-max"),
        "cache": _stats(),
    }


@app.post("/cancel/{query_id}")
async def cancel_query(query_id: str):
    """打断一次正在进行的分析（协作式：当前子步骤跑完后立即停止）"""
    with _cancel_lock:
        ev = _cancel_flags.get(query_id)
    if ev is not None and not ev.is_set():
        ev.set()
        logger.info(f"收到打断请求 {query_id}")
        return {"cancelled": True}
    return {"cancelled": False}


@app.get("/cache/stats")
async def cache_stats():
    """工具缓存运行统计（命中数/命中率/条目数/是否启用）"""
    from tools import cache_stats as _stats
    return _stats()


@app.post("/cache/clear")
async def cache_clear():
    """手动清空工具缓存（调试：改了工具逻辑后立即生效）"""
    from tools import clear_cache as _clear
    _clear()
    return {"cleared": True}


# ── 研报下载接口 ──────────────────────────────────────────────────────────────
REPORT_DIR = Path(__file__).parent.parent / "reports"

@app.get("/report/latest")
async def report_latest():
    files = sorted(REPORT_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return HTMLResponse("暂无研报，请先发起一次分析", status_code=404)
    return _serve_report_file(files[0])


@app.get("/report/{filename}")
async def report_file(filename: str):
    safe = Path(filename).name          # 防目录穿越
    target = REPORT_DIR / safe
    if not target.is_file():
        return HTMLResponse("研报不存在", status_code=404)
    return _serve_report_file(target)


def _serve_report_file(path: Path):
    content = path.read_text(encoding="utf-8")
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


# ── 托管 index.html ──────────────────────────────────────────────────────────
HTML_PATH = Path(__file__).parent.parent / "index.html"

@app.get("/")
async def root():
    if HTML_PATH.exists():
        return HTMLResponse(HTML_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>index.html not found</h2>")
