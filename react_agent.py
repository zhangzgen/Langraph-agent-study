from __future__ import annotations

import argparse
import ast
import operator
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


XIAOMI_DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
XIAOMI_DEFAULT_MODEL = "mimo-v2.5-pro"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SKILLS_DIR = PROJECT_ROOT / "skills"


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    path: Path


@tool
def calculator(expression: str) -> str:
    """安全计算基础算术表达式，例如 '23 * 47'。"""
    # @tool 会把函数签名和 docstring 暴露给模型。
    # 当模型判断需要计算时，会按 OpenAI tool calling 格式请求调用这个工具。
    try:
        result = _safe_eval(expression)
    except Exception as exc:
        return f"计算失败: {exc}"
    return str(result)


@tool
def current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取指定 IANA 时区的当前时间，例如 Asia/Shanghai 或 America/New_York。"""
    # 这是一个实时信息工具。模型本身不知道当前时间，应该通过工具获得。
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception as exc:
        return f"无法识别时区 {timezone!r}: {exc}"
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


# 基础工具列表会同时传给两处：
# 1. llm.bind_tools(BASE_TOOLS): 告诉模型“你可以调用这些普通工具”。
# 2. ToolNode(TOOLS): 真正执行模型请求的所有工具调用，包括 Skill 工具。
BASE_TOOLS = [calculator, current_time]


@tool
def list_skills() -> str:
    """列出当前项目 skills 目录中可用 Skill 的名称和描述。"""
    skills = discover_skills()
    if not skills:
        return "当前没有发现可用 Skill。"
    return _format_skill_catalog(skills)


@tool
def load_skill(skill_name: str) -> str:
    """按 Skill 名称加载完整 SKILL.md 说明正文。"""
    # 这是 Skill 机制的“按需加载”入口。
    # 模型启动时只看到 YAML 元数据；判断需要某个 Skill 后，再调用这个工具加载正文。
    skill = find_skill(skill_name)
    if skill is None:
        available = ", ".join(item.name for item in discover_skills()) or "无"
        return f"没有找到 Skill: {skill_name}。可用 Skill: {available}"

    content = skill.path.read_text(encoding="utf-8")
    metadata, body = split_skill_file(content)
    description = metadata.get("description", skill.description)
    return (
        f"Skill 名称: {skill.name}\n"
        f"Skill 描述: {description}\n\n"
        f"完整说明:\n{body.strip()}"
    )


SKILL_TOOLS = [list_skills, load_skill]
TOOLS = [*BASE_TOOLS, *SKILL_TOOLS]


def build_llm() -> ChatOpenAI:
    # 真实 AK 从 .env 或 shell 环境变量读取，避免写入代码仓库。
    api_key = os.getenv("XIAOMI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "缺少 XIAOMI_API_KEY。请复制 .env.example 为 .env，并填入你的真实 AK。"
        )

    # 小米的接口兼容 OpenAI Chat Completions，因此可以直接使用 ChatOpenAI。
    # bind_tools 会把工具 schema 附到请求里，模型如果需要工具，会返回 tool_calls。
    return ChatOpenAI(
        model=os.getenv("XIAOMI_MODEL", XIAOMI_DEFAULT_MODEL),
        api_key=api_key,
        base_url=os.getenv("XIAOMI_BASE_URL", XIAOMI_DEFAULT_BASE_URL),
        temperature=0,
    ).bind_tools(TOOLS)


def build_graph(with_memory: bool = False):
    llm = build_llm()
    skill_catalog = _format_skill_catalog(discover_skills())

    def call_llm(state: MessagesState) -> dict[str, list[BaseMessage]]:
        # MessagesState 是 LangGraph 内置 State，结构类似：
        # {"messages": [HumanMessage, AIMessage, ToolMessage, ...]}
        # 每个 node 只需要返回要追加到 state 的消息。
        system_message = {
            "role": "system",
            "content": (
                "你是一个会使用工具的 ReAct agent。"
                "需要实时信息或计算时先调用工具，拿到工具结果后再给最终回答。"
                "\n\n你还具备动态 Skill 能力。启动时你只会看到每个 Skill 的 YAML 元数据。"
                "如果用户请求匹配某个 Skill 的 description，必须先调用 load_skill(skill_name) "
                "读取完整技能说明，再按照该 Skill 回答。"
                "如果不确定有哪些 Skill，可以调用 list_skills。"
                "Skill 是行为说明，不替代工具；需要计算或实时信息时仍然继续调用工具。"
                f"\n\n当前可用 Skill 元数据:\n{skill_catalog}"
            ),
        }
        # 这里是 ReAct 中的“Reason/Act 决策”阶段：
        # 模型读取用户问题和历史消息，决定直接回答，还是返回 tool_calls。
        response = llm.invoke([system_message, *state["messages"]])
        return {"messages": [response]}

    # StateGraph 定义“状态如何在节点之间流动”。
    # 这个示例只手写两个节点：llm 和 tools。
    builder = StateGraph(MessagesState)

    # llm 节点：负责调用模型，让模型判断下一步。
    builder.add_node("llm", call_llm)

    # tools 节点：LangGraph 预置节点，负责执行 AIMessage.tool_calls。
    # 执行结果会变成 ToolMessage 追加回 messages。
    builder.add_node("tools", ToolNode(TOOLS))

    # 图从 START 进入 llm。
    builder.add_edge(START, "llm")

    # 条件边是 ReAct 循环的核心：
    # - 如果上一条 AIMessage 有 tool_calls，则路由到 tools。
    # - 如果没有 tool_calls，则路由到 END，表示模型已经给出最终回答。
    builder.add_conditional_edges("llm", tools_condition)

    # 工具执行完以后回到 llm。
    # 模型会看到 ToolMessage，再决定继续调用工具还是输出最终答案。
    builder.add_edge("tools", "llm")

    if not with_memory:
        return builder.compile()

    # checkpointer 是 LangGraph 的“会话记忆”入口。
    # 同一个 thread_id 下，每次 invoke/stream 只需要传入新增消息；
    # LangGraph 会从 checkpointer 取出旧 State，再把新消息追加进去。
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


def run(question: str, debug: bool = False) -> AIMessage:
    graph = build_graph()

    # 输入状态只需要包含用户消息。
    # 后续 AIMessage 和 ToolMessage 都由图中的节点自动追加。
    inputs = {"messages": [{"role": "user", "content": question}]}
    return _invoke_graph(graph, inputs, config=None, debug=debug)


def chat(thread_id: str = "default", debug: bool = False) -> None:
    # 多轮对话和单次调用的主要区别：
    # 1. graph 编译时启用 checkpointer。
    # 2. 每一轮调用都传同一个 thread_id。
    # 3. 输入只传本轮用户新消息，历史消息由 LangGraph 自动恢复。
    graph = build_graph(with_memory=True)
    config = {"configurable": {"thread_id": thread_id}}

    print("进入多轮对话模式。输入 exit、quit 或 q 结束。")
    print(f"thread_id: {thread_id}")

    while True:
        question = input("\n你: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("已结束多轮对话。")
            return
        if not question:
            continue

        inputs = {"messages": [{"role": "user", "content": question}]}
        final_message = _invoke_graph(graph, inputs, config=config, debug=debug)
        print(f"\n助手: {final_message.content}")


def _invoke_graph(graph, inputs: dict, config: dict | None, debug: bool) -> AIMessage:
    # 单次问答和多轮问答都可以复用这个执行函数。
    # debug=True 时打印每个 node 的增量输出，方便观察 ReAct 路由。
    if not debug:
        # invoke 会一次性跑完整张图，直到 END。
        result = graph.invoke(inputs, config=config)
        return result["messages"][-1]

    # debug 模式用 stream 查看每个节点的增量输出。
    # 这对学习 ReAct 很有用：可以看到 llm -> tools -> llm 的实际跳转。
    final_message: AIMessage | None = None
    for event in graph.stream(inputs, config=config, stream_mode="updates"):
        for node_name, update in event.items():
            print(f"\n[{node_name}]")
            for message in update.get("messages", []):
                print(_format_message(message))
                if isinstance(message, AIMessage):
                    final_message = message

    if final_message is None:
        raise RuntimeError("图执行结束，但没有得到 AIMessage。")
    return final_message


def _format_message(message: BaseMessage) -> str:
    # debug 打印时，如果模型请求工具调用，优先展示 tool_calls。
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        return f"{message.type}: tool_calls={tool_calls}"
    return f"{message.type}: {message.content}"


def get_skills_dir() -> Path:
    # 可通过 AGENT_SKILLS_DIR 覆盖默认目录，便于后续实验不同 Skill 集合。
    return Path(os.getenv("AGENT_SKILLS_DIR", DEFAULT_SKILLS_DIR)).expanduser()


def discover_skills() -> list[SkillMetadata]:
    # 只读取 SKILL.md 的 YAML frontmatter，不加载正文。
    # 这对应 Skill 的第一层 progressive disclosure：name + description。
    skills_dir = get_skills_dir()
    if not skills_dir.exists():
        return []

    skills: list[SkillMetadata] = []
    for skill_file in sorted(skills_dir.glob("*/SKILL.md")):
        try:
            metadata, _body = split_skill_file(skill_file.read_text(encoding="utf-8"))
        except ValueError:
            continue

        name = metadata.get("name")
        description = metadata.get("description")
        if not isinstance(name, str) or not isinstance(description, str):
            continue
        skills.append(
            SkillMetadata(
                name=name.strip(),
                description=description.strip(),
                path=skill_file,
            )
        )
    return skills


def find_skill(skill_name: str) -> SkillMetadata | None:
    normalized = skill_name.strip().lower()
    for skill in discover_skills():
        if skill.name.lower() == normalized:
            return skill
    return None


def split_skill_file(content: str) -> tuple[dict, str]:
    if not content.startswith("---\n"):
        raise ValueError("SKILL.md 必须以 YAML frontmatter 开始。")

    marker = "\n---\n"
    end = content.find(marker, 4)
    if end == -1:
        raise ValueError("SKILL.md 缺少 YAML frontmatter 结束标记。")

    frontmatter = content[4:end]
    body = content[end + len(marker) :]
    metadata = yaml.safe_load(frontmatter) or {}
    if not isinstance(metadata, dict):
        raise ValueError("SKILL.md frontmatter 必须是 YAML mapping。")
    return metadata, body


def _format_skill_catalog(skills: list[SkillMetadata]) -> str:
    if not skills:
        return "无"
    return "\n".join(
        f"- name: {skill.name}\n  description: {skill.description}"
        for skill in skills
    )


def _safe_eval(expression: str) -> int | float:
    # 不使用 eval，改用 ast 白名单，只允许基础算术节点。
    # 这是示例工具的安全边界，避免执行任意 Python 代码。
    operators: dict[type[ast.operator | ast.unaryop], Callable] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def eval_node(node: ast.AST) -> int | float:
        # 递归解释 AST，只处理数字、二元运算和正负号。
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](eval_node(node.left), eval_node(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](eval_node(node.operand))
        raise ValueError(f"不支持的表达式: {ast.dump(node, include_attributes=False)}")

    parsed = ast.parse(expression, mode="eval")
    return eval_node(parsed)


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
        print(_format_skill_catalog(discover_skills()))
        return

    if args.chat:
        chat(thread_id=args.thread_id, debug=args.debug)
        return

    final_message = run(args.question, debug=args.debug)
    print("\nFinal answer:")
    print(final_message.content)


if __name__ == "__main__":
    main()
