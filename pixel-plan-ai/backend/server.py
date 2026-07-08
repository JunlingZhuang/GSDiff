from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
PUBLIC = ROOT / "public"
DATA = ROOT / "data" / "samples.json"
sys.path.insert(0, str(Path(__file__).resolve().parent))

JOB_RUNNER = Path(__file__).resolve().parent / "job_runner.py"
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,64}$")
PROGRESS_ROOT = Path(tempfile.gettempdir()) / "pixel-plan-ai-progress"
ACTIVE_JOBS: dict[str, subprocess.Popen[str]] = {}
CANCELLED_JOBS: set[str] = set()
JOB_LOCK = threading.Lock()


class GenerationCancelled(Exception):
    pass


def progress_path_for(request_id: str) -> Path:
    return PROGRESS_ROOT / f"{request_id}.json"


def write_initial_progress(request_id: str) -> Path:
    path = progress_path_for(request_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": "starting",
                "phase": "starting",
                "message": "Starting the isolated generation worker.",
                "iterations": [],
                "preview": None,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return path


def read_progress_events(progress_file: Path) -> list[dict[str, Any]]:
    """Last 200 parsed events from the JSONL timeline beside the snapshot (docs #6)."""
    events_path = Path(str(progress_file) + ".events.jsonl")
    if not events_path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            events.append(json.loads(stripped))
        except json.JSONDecodeError:
            continue  # tolerate a torn last line from a concurrent append
    return events[-200:]


def read_generation_progress(request_id: str) -> dict[str, Any]:
    path = progress_path_for(request_id)
    with JOB_LOCK:
        active = request_id in ACTIVE_JOBS and ACTIVE_JOBS[request_id].poll() is None
    if not path.is_file():
        return {
            "status": "starting" if active else "missing",
            "phase": "starting" if active else "complete",
            "message": "Waiting for the first agent checkpoint." if active else "No active generation job was found.",
            "iterations": [],
            "preview": None,
            "events": read_progress_events(path),
            "active": active,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {
            "status": "starting",
            "phase": "starting",
            "message": "Waiting for the next complete agent checkpoint.",
            "iterations": [],
            "preview": None,
        }
    payload["events"] = read_progress_events(path)
    payload["active"] = active
    if not active and payload.get("status") == "running":
        payload["status"] = "complete"
    return payload


def run_generation_job(payload: dict[str, Any], request_id: str) -> dict[str, Any]:
    if not JOB_ID_PATTERN.fullmatch(request_id):
        raise ValueError("request_id must contain only letters, numbers, and hyphens.")
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    worker_environment = os.environ.copy()
    worker_environment["PYTHONIOENCODING"] = "utf-8"
    worker_environment["PYTHONUTF8"] = "1"
    progress_path = write_initial_progress(request_id)
    worker_environment["PIXEL_PLAN_PROGRESS_FILE"] = str(progress_path)
    process = subprocess.Popen(
        [sys.executable, "-B", str(JOB_RUNNER)],
        cwd=ROOT,
        env=worker_environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        creationflags=creation_flags,
    )
    with JOB_LOCK:
        if request_id in ACTIVE_JOBS:
            process.terminate()
            raise ValueError("A generation job with this request_id is already active.")
        ACTIVE_JOBS[request_id] = process
    job_timeout_seconds = max(60.0, float(os.environ.get("GENERATION_TIMEOUT_SECONDS", "330")))
    try:
        try:
            stdout, stderr = process.communicate(input=json.dumps(payload), timeout=job_timeout_seconds)
        except subprocess.TimeoutExpired as error:
            process.kill()
            process.communicate()
            raise ValueError(f"Generation exceeded the {job_timeout_seconds:g} second server limit.") from error
        with JOB_LOCK:
            was_cancelled = request_id in CANCELLED_JOBS
            CANCELLED_JOBS.discard(request_id)
        if was_cancelled:
            raise GenerationCancelled("Generation was stopped by the user.")
        if len(stdout) > 12_000_000:
            raise ValueError("Generation output exceeded the 12 MB limit.")
        try:
            response = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise ValueError(f"Generation worker returned invalid output: {stderr[:300]}") from error
        if not response.get("ok"):
            raise ValueError(str(response.get("error", "Generation worker failed.")))
        return response["result"]
    finally:
        with JOB_LOCK:
            if ACTIVE_JOBS.get(request_id) is process:
                ACTIVE_JOBS.pop(request_id, None)


def cancel_generation_job(request_id: str) -> bool:
    with JOB_LOCK:
        process = ACTIVE_JOBS.get(request_id)
        if process is None or process.poll() is not None:
            return False
        CANCELLED_JOBS.add(request_id)
    process.terminate()
    try:
        process.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        process.kill()
    return True


def load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()
SAMPLES = json.loads(DATA.read_text(encoding="utf-8"))
mimetypes.add_type("text/javascript", ".js")


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "PixelPlanAI/0.1"

    def log_message(self, format_string: str, *args: object) -> None:
        print(f"[{self.log_date_time_string()}] {format_string % args}")

    def send_json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        # image mode carries a base64 reference drawing in the generate payload
        if length > 10_000_000:
            raise ValueError("Request body exceeds 10 MB.")
        try:
            value = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as error:
            raise ValueError("Request body is not valid JSON.") from error
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object.")
        return value

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/health":
            self.send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "gemini_configured": bool(os.environ.get("GEMINI_API_KEY")),
                    "model": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
                    "fast_model": os.environ.get("GEMINI_FAST_MODEL", "gemini-3.1-flash-lite"),
                    "quality_model": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
                    "complex_model": os.environ.get("GEMINI_COMPLEX_MODEL", "gemini-3.1-pro-preview"),
                },
            )
            return
        if path == "/api/samples":
            self.send_json(HTTPStatus.OK, SAMPLES)
            return
        progress_prefix = "/api/generate/"
        if path.startswith(progress_prefix):
            request_id = unquote(path[len(progress_prefix) :])
            if not JOB_ID_PATTERN.fullmatch(request_id):
                self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid request id"})
                return
            self.send_json(HTTPStatus.OK, read_generation_progress(request_id))
            return
        self.serve_static(path)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        job_kinds = {
            "/api/generate": "agent",
            "/api/execute": "execute",
            "/api/images": "images",
            "/api/refine": "refine",
        }
        if path not in job_kinds:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            payload = self.read_json()
            payload["job_kind"] = job_kinds[path]
            request_id = str(payload.get("request_id") or uuid.uuid4())
            result = run_generation_job(payload, request_id)
            result["generated_at"] = datetime.now(timezone.utc).isoformat()
            self.send_json(HTTPStatus.OK, result)
        except GenerationCancelled as error:
            self.send_json(409, {"cancelled": True, "error": str(error)})
        except Exception as error:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        prefix = "/api/generate/"
        if not path.startswith(prefix):
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        request_id = unquote(path[len(prefix) :])
        if not JOB_ID_PATTERN.fullmatch(request_id):
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid request id"})
            return
        cancelled = cancel_generation_job(request_id)
        self.send_json(HTTPStatus.OK, {"cancelled": cancelled, "request_id": request_id})

    def serve_static(self, request_path: str) -> None:
        relative = "index.html" if request_path == "/" else unquote(request_path.lstrip("/"))
        target = (PUBLIC / relative).resolve()
        try:
            target.relative_to(PUBLIC.resolve())
        except ValueError:
            self.send_json(HTTPStatus.FORBIDDEN, {"error": "Forbidden"})
            return
        if not target.is_file():
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            content_type += "; charset=utf-8"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    requested_port = int(os.environ.get("PORT", "8765"))
    candidates = list(dict.fromkeys((requested_port, 8765, 8787)))
    server: ThreadingHTTPServer | None = None
    selected_port = requested_port
    last_error: OSError | None = None
    for candidate in candidates:
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), RequestHandler)
            selected_port = candidate
            break
        except OSError as error:
            last_error = error
    if server is None:
        raise RuntimeError(f"Could not bind a local server port: {last_error}")
    if selected_port != requested_port:
        print(f"Port {requested_port} was unavailable; using {selected_port} instead.")
    print(f"Pixel Plan AI running at http://127.0.0.1:{selected_port}")
    print("Gemini configured." if os.environ.get("GEMINI_API_KEY") else "Gemini key is required before generation.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
