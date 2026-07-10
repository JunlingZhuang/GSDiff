from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from code_policy import validate_generated_code

# Mirror of runtime.py for the isolated ICU room pipeline. The scaffolding is duplicated on
# purpose: the room flow must not share mutable surface with the floor runtime (isolation rule).
RUNNER = Path(__file__).with_name("room_sandbox_runner.py")


def sanitize_code(raw_code: str) -> str:
    code = str(raw_code or "").strip()
    if code.startswith("```"):
        lines = code.splitlines()
        lines = lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:]
        code = "\n".join(lines).strip()
    validate_generated_code(code)
    return code


def execute_room_code(raw_code: str) -> tuple[str, dict[str, Any]]:
    code = sanitize_code(raw_code)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-B", str(RUNNER)],
            input=code,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=12.0,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError("Generated code exceeded the execution time limit.") from error
    if len(completed.stdout) > 8_000_000:
        raise ValueError("Generated room exceeded the output size limit.")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ValueError(f"Sandbox returned invalid output: {completed.stderr[:300]}") from error
    if not payload.get("ok"):
        raise ValueError(str(payload.get("error", "Generated code failed.")))
    return code, payload["result"]
