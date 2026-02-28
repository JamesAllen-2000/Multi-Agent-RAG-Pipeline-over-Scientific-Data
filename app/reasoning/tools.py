"""
Single tool for the reasoning agent beyond retrieval: a safe calculator.
No code execution — only a restricted set of operations (numbers, +, -, *, /, **, parentheses).
This satisfies "at least one tool beyond retrieval" without allowing arbitrary code.
"""
import re


# Allowed: digits, spaces, + - * / ** ( )
_CALC_RE = re.compile(r"^[\d\s+\-*/().]+$")


def calculator(expression: str) -> str:
    """
    Evaluate a single mathematical expression. No exec, no eval of arbitrary code.
    Only digits and operators + - * / ** ( ) allowed. Returns result or error message.
    """
    expr = expression.strip()
    if not expr:
        return "Error: empty expression"
    if not _CALC_RE.match(expr):
        return "Error: only numbers and + - * / ** ( ) are allowed"
    try:
        # Safe: we only allow digits and arithmetic operators; no names or builtins
        result = eval(expr)
        if isinstance(result, (int, float)):
            return str(result)
        return "Error: result must be a number"
    except Exception as e:
        return f"Error: {e}"
