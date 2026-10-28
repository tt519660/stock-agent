"""
统一入口：切换手写版 / Function Calling 版 ReAct Agent

使用方式：
  python agent.py
  python agent.py --mode manual   --question "茅台2023年毛利率是多少？"
  python agent.py --mode fc       --question "五粮液近一年股价涨跌幅？"
  python agent.py --mode manual   --question "..." --max_steps 8
  python agent.py --mode manual   --question "..." --save-report   # 同时导出 Markdown 研报
  python agent.py --mode fc       --question "解读这张图" --image D:\kline.png  # 附图分析（视觉工具）

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
    parser.add_argument("--no-cache", action="store_true",
                        help="禁用工具结果缓存（全部走真实网络请求)")
    parser.add_argument("--image", action="append", default=None,
                        help="附带图片（本地路径或URL，可多次），Agent 会先用视觉工具 image_analysis 理解图片")
    args = parser.parse_args()

    if args.no_cache:
        os.environ["TOOL_CACHE_DISABLE"] = "1"

    # 附图注入：把图片路径写进问题文本，由 Agent 自主调用 image_analysis（引擎零改动）
    full_question = args.question
    if args.image:
        if args.interactive:
            print("⚠ --interactive 模式暂不支持 --image，本次附图将被忽略")
        else:
            lines = [f"[用户附带图片: {img}]" for img in args.image]
            full_question = (f"{args.question}\n\n"
                             f"用户本次附带图片，请先用 image_analysis 工具查看后再分析：\n"
                             + "\n".join(lines))

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
        for step_data in run(full_question, max_steps=args.max_steps):
            if step_data["type"] == "action":
                print(f"\n[Step {step_data['step']}]")
                print(f"🔧 {step_data['action']}({step_data['action_input']})")
                print(f"👁  Obs: {step_data['observation'][:300]}")
            elif step_data["type"] == "final":
                final_answer = step_data["answer"]
                print(f"\n✅ Final Answer:\n{final_answer}")
            elif step_data["type"] == "stopped":
                final_answer = "已手动停止"
                print("\n⏹ 已手动停止")
            steps.append(step_data)

        from report import build_report_md, save_report
        md = build_report_md(full_question, args.mode, model, final_answer,
                             steps, time.time() - start)
        output = save_report(md)
        print(f"\n📄 研报已保存: {output}")
    else:
        if args.mode == "manual":
            from react_manual import run_and_print
        else:
            from react_function_calling import run_and_print
        run_and_print(full_question, args.max_steps)
