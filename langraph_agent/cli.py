from __future__ import annotations

import argparse

from langraph_agent.skills.registry import discover_skills, format_skill_catalog


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal LangGraph ReAct agent.")
    parser.add_argument("question", nargs="?", default="北京现在几点？再计算 23 * 47。")
    parser.add_argument("--chat", action="store_true", help="Start multi-turn chat mode.")
    parser.add_argument("--debug", action="store_true", help="Print graph updates.")
    parser.add_argument(
        "--thread-id",
        default="default",
        help="Conversation id used by LangGraph checkpointer in chat mode.",
    )
    parser.add_argument(
        "--checkpoint-db",
        default=None,
        help=(
            "SQLite checkpoint database path. Defaults to data/checkpoints.sqlite "
            "or LANGRAPH_CHECKPOINT_DB_PATH."
        ),
    )
    parser.add_argument("--list-skills", action="store_true", help="Print discovered skills.")
    args = parser.parse_args()

    if args.list_skills:
        print(format_skill_catalog(discover_skills()))
        return

    if args.chat:
        from langraph_agent.graph import chat

        chat(
            thread_id=args.thread_id,
            debug=args.debug,
            checkpoint_db_path=args.checkpoint_db,
        )
        return

    from langraph_agent.graph import run

    final_message = run(
        args.question,
        debug=args.debug,
        checkpoint_db_path=args.checkpoint_db,
    )
    if args.debug:
        return

    print("\nFinal answer:")
    print(final_message.content)
