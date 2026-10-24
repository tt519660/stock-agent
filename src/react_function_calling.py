"""
Function Calling API 版 ReAct Agent

教学重点：
  1. 与手写版对比：框架帮你处理格式解析，但 Thought 过程在内部不可见
  2. tool_choice="auto" 让模型自己决定调用哪个工具或直接回答
  3. finish_reason 判断：tool_calls 表示继续调用，stop 表示给出最终答案
  4. 相同工具集，相同问题，对比两种实现的稳定性和步骤数

使用方式：
  python react_function_calling.py
  python react_function_calling.py --question "茅台近一年股价涨跌幅如何？"
  python react_function_calling.py --question "..." --max_steps 8

依赖：
  pip install openai faiss-cpu sentence-transformers akshare
  export DASHSCOPE_API_KEY="sk-xxx"
"""

import os
import json
import time
import logging
import argparse
from typing import Generator

from openai import OpenAI

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── 短期记忆 ──────────────────────────────────────────────────────────────────
_conversation_history: list | None = None

def reset_conversation():
    """清空短期记忆，开始全新对话"""
    global _conversation_history
    _conversation_history = None


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL = os.getenv("AGENT_MODEL", "qwen3.8-max")
# client = OpenAI(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com",
# )
# MODEL = os.getenv("AGENT_MODEL", "deepseek-v4-flash")

FC_SYSTEM_PROMPT = """你是一个专业的A股金融分析助手。
规则：
- 调用 financial_indicator 或 stock_price 之前，必须先用 company_lookup 获取股票代码
- 数字计算必须使用 calculator 工具，不能心算
- Final Answer 必须引用具体数据来源
- 如果没有合适工具能回答，直接说明原因
"""


def run(question: str, max_steps: int = 10, reset: bool = True,
        history: list | None = None) -> Generator[dict, None, None]:
    """
    执行 Function Calling 版 ReAct 循环，yield 每一步结构化结果

    支持短期记忆：同一进程内多次调用会自动追加到对话历史。
    调用 reset_conversation() 或传入 reset=True 可清空记忆，开始全新对话。
    传入 history=[] 可使用外部管理的对话历史（serve.py 多 session 场景），
    此时由调用方负责 history 的生命周期，模块级 _conversation_history 不受影响。

    格式与 react_manual.run() 保持一致，便于 evaluate.py 统一对比
    """
    from tools import TOOLS_MAP, TOOLS_SCHEMA

    global _conversation_history

    if history is not None:
        if reset:
            history.clear()
            history.append({"role": "system", "content": FC_SYSTEM_PROMPT})
        history.append({"role": "user", "content": question})
        messages = history
        _conversation_history = history   # 让后续 _conversation_history 引用也指向外部列表
    else:
        if reset or _conversation_history is None:
            _conversation_history = [
                {"role": "system", "content": FC_SYSTEM_PROMPT},
            ]
        _conversation_history.append({"role": "user", "content": question})
        messages = _conversation_history

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0,
        )
        msg    = response.choices[0].message
        reason = response.choices[0].finish_reason

        # 模型决定直接回答（无工具调用）
        if reason == "stop" or not msg.tool_calls:
            messages.append(msg)  # 存入短期记忆
            yield {
                "step":   step,
                "type":   "final",
                "thought": "",
                "answer": msg.content or "（模型返回空内容）",
            }
            return

        # 模型请求调用工具
        messages.append(msg)

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_args = {}

            tool_fn = TOOLS_MAP.get(tool_name)
            if tool_fn is None:
                observation = f"未知工具 '{tool_name}'"
            else:
                try:
                    observation = tool_fn(**tool_args)
                except TypeError as e:
                    observation = f"工具参数错误: {e}"

            step_result = {
                "step":         step,
                "type":         "action",
                "thought":      "",   # Function Calling 版 Thought 在模型内部，不可见
                "action":       tool_name,
                "action_input": tool_args,
                "observation":  str(observation),
            }
            yield step_result

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      str(observation),
            })

    # 超出最大步数
    _conversation_history.append({
        "role": "assistant",
        "content": f"已达最大步数 {max_steps}，未能得出最终答案",
    })
    yield {
        "step":   max_steps + 1,
        "type":   "max_steps",
        "answer": f"已达最大步数 {max_steps}，未能得出最终答案",
    }


# ── CLI 打印（复用 react_manual 的彩色输出） ───────────────────────────────────

COLORS = {
    "thought": "\033[36m",
    "action":  "\033[33m",
    "obs":     "\033[32m",
    "final":   "\033[35m",
    "error":   "\033[31m",
    "reset":   "\033[0m",
}

def _c(color: str, text: str) -> str:
    return f"{COLORS[color]}{text}{COLORS['reset']}"


def run_and_print(question: str, max_steps: int = 10, reset: bool = True):
    print(f"\n{'='*60}")
    print(f"问题: {question}")
    print(f"模型: {MODEL}  实现: Function Calling")
    print('='*60)

    start = time.time()

    for step_data in run(question, max_steps=max_steps, reset=reset):
        stype = step_data["type"]

        if stype == "action":
            print(f"\n[Step {step_data['step']}]")
            # Thought 在 FC 版不可见，显示提示
            print(_c("thought", "🧠 Thought: （模型内部推理，Function Calling 版不可见）"))
            print(_c("action",  f"🔧 Action:  {step_data['action']}"))
            print(_c("action",  f"   Input:   {json.dumps(step_data['action_input'], ensure_ascii=False)}"))
            print(_c("obs",     f"👁  Obs:     {step_data['observation'][:300]}"))

        elif stype == "final":
            elapsed = time.time() - start
            print(f"\n{'─'*60}")
            print(_c("final", f"\n✅ Final Answer:\n{step_data['answer']}"))
            print(f"\n共 {step_data['step']} 步，耗时 {elapsed:.1f}s")

        elif stype in ("error", "max_steps"):
            print(_c("error", f"\n⚠️  {step_data.get('answer', '')}"))


def chat(max_steps: int = 10):
    """交互式多轮对话循环，每次输入一个问题，自动累积短期记忆"""
    reset_conversation()  # 每次进入 chat 都是全新对话
    print("\nReAct Financial Agent — 多轮对话模式")
    print("输入问题开始对话，/reset 清空记忆，/exit 退出\n")
    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q == "/exit":
            break
        if q == "/reset":
            reset_conversation()
            print("记忆已清空")
            continue
        if not q:
            continue
        run_and_print(q, max_steps, reset=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question",    default="贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？")
    parser.add_argument("--max_steps",   type=int, default=10)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    if args.interactive:
        chat(max_steps=args.max_steps)
    else:
        run_and_print(args.question, args.max_steps)
