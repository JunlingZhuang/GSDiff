from __future__ import annotations

import json
import sys

from service import dispatch_job


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read() or "{}")
        if not isinstance(payload, dict):
            raise ValueError("Generation payload must be a JSON object.")
        result = dispatch_job(payload)
        print(json.dumps({"ok": True, "result": result}, separators=(",", ":"), ensure_ascii=False))
        return 0
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}, separators=(",", ":"), ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
