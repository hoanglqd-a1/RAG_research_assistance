"""A small arithmetic evaluator that never executes Python code."""

import ast
import math
import operator
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from app.agent.registry import Tool


class CalculatorInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression: str = Field(min_length=1, max_length=200)


_BINARY_OPERATORS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_arithmetic(expression: str) -> int | float:
    """Evaluate only numeric literals and explicitly allowed arithmetic nodes."""

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Expression is not valid arithmetic") from exc
    if sum(1 for _ in ast.walk(tree)) > 50:
        raise ValueError("Expression is too complex")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed")
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](float(evaluate(node.operand)))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(float(right)) > 100:
                raise ValueError("Exponent is too large")
            value = _BINARY_OPERATORS[type(node.op)](float(left), float(right))
            if not math.isfinite(value) or abs(value) > 1e100:
                raise ValueError("Result is outside the allowed numeric range")
            return int(value) if value.is_integer() else value
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    return evaluate(tree)


def create_calculator_tool() -> Tool[CalculatorInput]:
    def calculator(arguments: CalculatorInput) -> dict[str, int | float | str]:
        return {
            "expression": arguments.expression,
            "result": evaluate_arithmetic(arguments.expression),
        }

    return Tool(
        name="calculator",
        description=(
            "Safely calculate a numerical arithmetic expression using addition, "
            "subtraction, multiplication, division, parentheses, or exponentiation."
        ),
        input_model=CalculatorInput,
        handler=calculator,
    )
