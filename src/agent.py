"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent

使用方式：
  python agent.py
  python agent.py --mode manual   --question "茅台2023年毛利率是多少？"
  python agent.py --mode fc       --question "五粮液近一年股价涨跌幅？"
  python agent.py --mode manual   --question "..." --max_steps 8
  python agent.py --mode manual   --question "..." --save-report   # 同时导出 Markdown 研报

环境变量：
  DASHSCOPE_API_KEY  必填
  AGENT_MODEL        默认 qwen3.8-max，可换 deepseek-v3 等
"""

import os
import time
import argparse

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

DEFAULT_QUESTION = "贵州茅台和五粮液2023年的毛利率哪家更高？差多少个百分点？"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReAct Financial Agent")
    parser.add_argument(
        "--mode", choices=["manual", "fc"], default="manual",
        help="manual=手写Prompt解析版  fc=Function Calling版",
    )
    parser.add_argument("--question",    default=DEFAULT_QUESTION)
    parser.add_argument("--max_steps",   type=int, default=10)
    parser.add_argument("--interactive", action="store_true",
                        help="多轮对话模式（短期记忆）")
    parser.add_argument("--save-report", action="store_true",
                        help="分析结束后导出 Markdown 研报到 reports/ 目录")
    args = parser.parse_args()

    if args.interactive:
        if args.mode == "manual":
            from react_manual import chat
        else:
            from react_function_calling import chat
        chat(max_steps=args.max_steps)
    elif args.save_report:
        model = os.getenv("AGENT_MODEL", "qwen3.8-max")
        if args.mode == "manual":
            from react_manual import run
        else:
            from react_function_calling import run

        steps, start = [], time.time()
        final_answer = ""
        for step_data in run(args.question, max_steps=args.max_steps):
            if step_data["type"] == "action":
                print(f"\n[Step {step_data['step']}]")
                print(f"🔧 {step_data['action']}({step_data['action_input']})")
                print(f"👁  Obs: {step_data['observation'][:300]}")
            elif step_data["type"] == "final":
                final_answer = step_data["answer"]
                print(f"\n✅ Final Answer:\n{final_answer}")
            steps.append(step_data)

        from report import build_report_md, save_report
        md = build_report_md(args.question, args.mode, model, final_answer,
                             steps, time.time() - start)
        output = save_report(md)
        print(f"\n📄 研报已保存: {output}")
    else:
        if args.mode == "manual":
            from react_manual import run_and_print
        else:
            from react_function_calling import run_and_print
        run_and_print(args.question, args.max_steps)
