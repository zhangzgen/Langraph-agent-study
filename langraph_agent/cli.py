from __future__ import annotations

import argparse

from dotenv import load_dotenv

from langraph_agent.skills.registry import discover_skills, format_skill_catalog


def main() -> None:
    # 加载项目根目录下的 .env，便于长期在本项目环境中使用。
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run a minimal LangGraph ReAct agent.")
    parser.add_argument("question", nargs="?", default="北京现在几点？再计算 23 * 47。")
    parser.add_argument("--chat", action="store_true", help="Start multi-turn chat mode.")
    parser.add_argument("--debug", action="store_true", help="Print graph updates.")
    parser.add_argument(
        "--thread-id",
        default="default",
        help="Conversation id used by LangGraph checkpointer in chat mode.",
    )
    parser.add_argument("--list-skills", action="store_true", help="Print discovered skills.")
    args = parser.parse_args()

    if args.list_skills:
        print(format_skill_catalog(discover_skills()))
        return

    if args.chat:
        from langraph_agent.graph import chat

        chat(thread_id=args.thread_id, debug=args.debug)
        return

    from langraph_agent.graph import run

    final_message = run(args.question, debug=args.debug)
    if args.debug:
        return

    print("\nFinal answer:")
    print(final_message.content)
