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

    item_queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()

    def _worker():
        try:
            for step_data in react_run(question, max_steps=max_steps,
                                       reset=reset, history=history):
                item_queue.put_nowait(step_data)
        finally:
            item_queue.put_nowait(_SENTINEL)

    yield _sse({
        "type": "start", "question": question, "mode": mode,
        "session_id": session_id,
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
                model=os.getenv("AGENT_MODEL", "qwen-max"),
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
    return {"status": "ok", "model": os.getenv("AGENT_MODEL", "qwen-max")}


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
