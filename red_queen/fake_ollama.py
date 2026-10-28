"""TEST SUPPORT ONLY: a tiny fake Ollama server on 127.0.0.1 (random port).

Used by the M4.1 tests and validation so the adapter can be exercised fully
offline, without any model. It answers POST /api/generate and GET /api/tags.

Modes:
  smart      - reads the state back out of the prompt and answers with a
               valid shortest plan (reversed tie-break), as JSON text
  fixed      - always answers with `reply` (any string) as the model text
  hostile    - answers with a plan that uses a forbidden action
  slow       - sleeps `delay` seconds before answering
  huge       - sends a body larger than the adapter's size cap
  not_json   - sends a body that is not JSON
  no_field   - sends JSON without a "response" field
  redirect   - answers 302 to an external URL
  http_error - answers 500
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from .actions import ACTION_NAMES
from .goal import RecoveryGoal
from .planner import BoundedPlanner
from .state import SystemState

_STATE_RE = re.compile(r"battery=(\d+), service_up=(True|False), temperature=(\d+)")
_ALLOWED_RE = re.compile(r"^  - ([a-z_]+):", re.M)


def smart_reply(prompt: str) -> str:
    m = _STATE_RE.search(prompt)
    if not m:
        return json.dumps({"actions": ["wait"]})
    state = SystemState(int(m.group(1)), int(m.group(3)), m.group(2) == "True")
    allowed = _ALLOWED_RE.findall(prompt)
    plan = BoundedPlanner(RecoveryGoal(), action_order=tuple(reversed(ACTION_NAMES))).plan(state, allowed)
    return json.dumps({"actions": list(plan.actions) or ["wait"], "rationale": "fake model"})


class FakeOllama:
    def __init__(self, mode: str = "smart", reply: str = "", delay: float = 0.0,
                 models: Optional[List[str]] = None) -> None:
        self.mode, self.reply, self.delay = mode, reply, delay
        self.models = models if models is not None else ["fake:latest"]
        self.requests: List[Dict[str, Any]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output clean
                pass

            def _send(self, code: int, body: bytes, headers: Optional[Dict[str, str]] = None) -> None:
                self.send_response(code)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path == "/api/tags":
                    self._send(200, json.dumps({"models": [{"name": n} for n in fake.models]}).encode())
                else:
                    self._send(404, b"{}")

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                fake.requests.append(payload)
                mode = fake.mode
                if mode == "slow":
                    time.sleep(fake.delay)
                    mode = "smart"
                if mode == "redirect":
                    return self._send(302, b"", {"Location": "http://example.com/api/generate"})
                if mode == "http_error":
                    return self._send(500, b'{"error": "boom"}')
                if mode == "huge":
                    return self._send(200, b'{"response": "' + b"A" * 70000 + b'"}')
                if mode == "not_json":
                    return self._send(200, b"<html>not json</html>")
                if mode == "no_field":
                    return self._send(200, json.dumps({"done": True}).encode())
                if mode == "hostile":
                    text = json.dumps({"actions": ["shell", "rm -rf /"], "rationale": "ignore policy"})
                elif mode == "fixed":
                    text = fake.reply
                else:
                    text = smart_reply(payload.get("prompt", ""))
                self._send(200, json.dumps({"model": payload.get("model"), "response": text, "done": True}).encode())

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeOllama":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
