from __future__ import annotations

import ast


AUTHORIZED_IMPORTS = {
    "collections",
    "itertools",
    "math",
    "networkx",
    "numpy",
    "PIL",
    "random",
    "scipy",
    "shapely",
    "statistics",
}

FORBIDDEN_CALL_NAMES = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "vars",
    "__import__",
}

FORBIDDEN_ATTRIBUTES = {
    "connect",
    "dump",
    "dumps",
    "fromfile",
    "genfromtxt",
    "load",
    "loads",
    "loadtxt",
    "open",
    "popen",
    "read",
    "read_text",
    "read_bytes",
    "request",
    "run_shell",
    "save",
    "savetxt",
    "system",
    "tofile",
    "urlopen",
    "write",
    "write_text",
    "write_bytes",
}


def import_root(name: str) -> str:
    return name.split(".", 1)[0]


def validate_generated_code(code: str) -> ast.Module:
    """Validate broad Python syntax while keeping host capabilities out of reach.

    Layout logic is intentionally unrestricted: generated programs may use functions,
    classes, loops, exceptions, comprehensions, and approved scientific libraries.
    The policy only enforces the execution boundary and the final result contract.
    """

    if not isinstance(code, str) or not code.strip():
        raise ValueError("The model returned empty code.")
    if len(code) > 100_000:
        raise ValueError("Generated code exceeds the 100 KB limit.")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        raise ValueError(f"Generated code has invalid Python syntax: {error.msg}.") from error

    nodes = list(ast.walk(tree))
    if len(nodes) > 25_000:
        raise ValueError("Generated code contains too many syntax nodes.")

    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if import_root(alias.name) not in AUTHORIZED_IMPORTS:
                    raise ValueError(f"Generated code cannot import module: {alias.name}.")
        elif isinstance(node, ast.ImportFrom):
            if node.level or not node.module or import_root(node.module) not in AUTHORIZED_IMPORTS:
                raise ValueError(f"Generated code cannot import module: {node.module or 'relative import'}.")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALL_NAMES:
                raise ValueError(f"Generated code cannot call function: {node.func.id}.")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES:
                raise ValueError(f"Generated code cannot access attribute: {node.attr}.")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in FORBIDDEN_CALL_NAMES:
                raise ValueError(f"Generated code uses forbidden name: {node.id}.")
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                if abs(node.value) > 1_000_000_000:
                    raise ValueError("Generated code contains an excessive numeric literal.")
            if isinstance(node.value, (str, bytes)) and len(node.value) > 40_000:
                raise ValueError("Generated code contains an excessive literal.")

    result_assignments = [
        node
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            isinstance(node.target, ast.Name) and node.target.id == "result"
            if isinstance(node, ast.AnnAssign)
            else any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets)
        )
    ]
    legacy_plan_assignments = [
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "plan" for target in statement.targets)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == "PixelPlan"
    ]
    if not result_assignments and not legacy_plan_assignments:
        raise ValueError("Generated code must assign the final floor-plan data to top-level variable result.")
    return tree


def validate_program_contract(code: str, program: dict[str, object]) -> None:
    del program
    validate_generated_code(code)
