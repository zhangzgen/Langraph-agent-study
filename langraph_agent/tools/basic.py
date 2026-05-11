from __future__ import annotations

import ast
import operator
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from langchain_core.tools import tool


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
