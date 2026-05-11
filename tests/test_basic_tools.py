from __future__ import annotations

import pytest

from langraph_agent.tools.basic import _safe_eval, calculator


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("23 * 47", 1081),
        ("2 + 3 * 4", 14),
        ("(10 - 3) / 7", 1),
        ("2 ** 3", 8),
        ("-5 + 2", -3),
    ],
)
def test_safe_eval_allows_basic_arithmetic(expression: str, expected: int | float) -> None:
    assert _safe_eval(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "abs(-1)",
        "__import__('os').system('echo unsafe')",
        "'hello'",
    ],
)
def test_safe_eval_rejects_non_arithmetic_nodes(expression: str) -> None:
    with pytest.raises(ValueError):
        _safe_eval(expression)


def test_calculator_tool_returns_result() -> None:
    assert calculator.invoke({"expression": "23 * 47"}) == "1081"


def test_calculator_tool_reports_invalid_expression() -> None:
    result = calculator.invoke({"expression": "abs(-1)"})

    assert result.startswith("计算失败:")
