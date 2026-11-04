import ast
import math
import operator

from pydantic import BaseModel, ConfigDict, Field


class CalculatorArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expression: str = Field(min_length=1, max_length=300)


_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "sin": math.sin,
    "cos": math.cos,
    "log": math.log,
}


def calculate(expression: str) -> float | int:
    if len(expression) > 300:
        raise ValueError("Expression is too long")
    tree = ast.parse(expression, mode="eval")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            if isinstance(node.op, ast.Pow) and abs(visit(node.right)) > 100:
                raise ValueError("Exponent too large")
            return _BIN[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](visit(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS
            and not node.keywords
        ):
            return _FUNCS[node.func.id](*(visit(arg) for arg in node.args))
        raise ValueError("Unsupported expression")

    value = visit(tree)
    if not math.isfinite(value):
        raise ValueError("Result is not finite")
    return value
