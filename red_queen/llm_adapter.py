"""M4.1 local language-model adapter (Ollama HTTP API), opt-in and bounded.

`OllamaProposer` implements the M4 Proposer interface: it sends
`render_prompt(context)` to a model server on THIS machine and returns the
model's text. That text is untrusted and goes through the M4 parser,
validator, gate and PolicyExecutor exactly like any stub proposer.

Network safety:
  * only http://127.0.0.1, http://localhost or http://[::1] are accepted
  * proxy environment variables are ignored (direct connection only)
  * HTTP redirects are refused
  * every request has a hard overall deadline and a response-size cap
  * only two endpoints are used: POST /api/generate and GET /api/tags

Nothing here runs unless a caller explicitly builds an OllamaProposer.
Standard library only.
"""

from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .proposals import ProposalContext, render_prompt

DEFAULT_URL = "http://127.0.0.1:11434"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
DEFAULT_TIMEOUT = 60.0          # seconds, whole request
MAX_TIMEOUT = 300.0
MAX_RESPONSE_BYTES = 65536      # raw HTTP body cap
MAX_PREDICT_TOKENS = 256
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,99}$")
SYSTEM_INSTRUCTION = (
    "You are a planning assistant for a simulated recovery task. "
    "Answer with a single JSON object and nothing else."
)


class LocalModelError(RuntimeError):
    """Any adapter failure. The proposal gate treats it as a rejected proposal."""


def check_local_url(url: str) -> str:
    """Return a normalised base URL, or raise if it is not plain-HTTP loopback."""
    if not isinstance(url, str) or len(url) > 200:
        raise LocalModelError("invalid URL")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http":
        raise LocalModelError("only plain http to this machine is allowed")
    if parts.hostname not in LOOPBACK_HOSTS:
        raise LocalModelError(f"host {parts.hostname!r} is not loopback")
    if parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
        raise LocalModelError("URL must be just scheme://host:port")
    try:
        port = parts.port or 80
    except ValueError:
        raise LocalModelError("invalid port") from None
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"http://{host}:{port}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LocalModelError(f"redirect refused ({code})")


def _opener() -> urllib.request.OpenerDirector:
    # ProxyHandler({}) disables HTTP(S)_PROXY environment variables.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _request(base_url: str, path: str, payload: Optional[Dict[str, Any]], timeout: float,
             max_bytes: int) -> Dict[str, Any]:
    if path not in ("/api/generate", "/api/tags"):
        raise LocalModelError(f"endpoint {path!r} not allowed")
    url = check_local_url(base_url) + path
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if data is None else "POST",
                                 headers={"Content-Type": "application/json"})
    deadline = time.monotonic() + timeout
    try:
        with _opener().open(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise LocalModelError(f"HTTP {resp.status}")
            chunks: List[bytes] = []
            size = 0
            while True:
                if time.monotonic() > deadline:
                    raise LocalModelError("deadline exceeded")
                chunk = resp.read(4096)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise LocalModelError("response too large")
                chunks.append(chunk)
    except LocalModelError:
        raise
    except urllib.error.HTTPError as exc:  # non-2xx reply: close it (Python 3.14 warns otherwise)
        exc.close()
        raise LocalModelError(f"HTTP {exc.code}") from None
    except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
        raise LocalModelError(f"request failed: {type(exc).__name__}") from None
    try:
        body = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise LocalModelError("server reply is not JSON") from None
    if not isinstance(body, dict):
        raise LocalModelError("server reply is not a JSON object")
    return body


def list_local_models(base_url: str = DEFAULT_URL, timeout: float = 10.0) -> List[str]:
    body = _request(base_url, "/api/tags", None, timeout, MAX_RESPONSE_BYTES)
    models = body.get("models")
    if not isinstance(models, list):
        raise LocalModelError("unexpected /api/tags reply")
    return sorted(m["name"] for m in models if isinstance(m, dict) and isinstance(m.get("name"), str))


class OllamaProposer:
    """M4 Proposer backed by a local Ollama server. Untrusted output."""

    def __init__(self, model: str, base_url: str = DEFAULT_URL, timeout: float = DEFAULT_TIMEOUT,
                 seed: int = 0, max_response_bytes: int = MAX_RESPONSE_BYTES,
                 think: Optional[bool] = None) -> None:
        if not isinstance(model, str) or not _MODEL_RE.match(model):
            raise LocalModelError("invalid model name")
        if not 0 < timeout <= MAX_TIMEOUT:
            raise LocalModelError(f"timeout must be in (0, {MAX_TIMEOUT}]")
        if not 0 < max_response_bytes <= MAX_RESPONSE_BYTES:
            raise LocalModelError("invalid response size cap")
        self.model = model
        self.base_url = check_local_url(base_url)
        self.timeout = float(timeout)
        self.seed = int(seed)
        self.max_response_bytes = max_response_bytes
        if think not in (None, True, False):
            raise LocalModelError("think must be None, True or False")
        self.think = think  # None: do not send the field (server default)
        self.name = f"ollama:{model}"
        self.calls = 0
        self.errors = 0
        self.seconds = 0.0

    def propose(self, context: ProposalContext) -> str:
        self.calls += 1
        payload = {
            "model": self.model,
            "system": SYSTEM_INSTRUCTION,
            "prompt": render_prompt(context),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": self.seed, "num_predict": MAX_PREDICT_TOKENS},
        }
        if self.think is not None:  # for "thinking" models such as qwen3: think=False skips reasoning
            payload["think"] = self.think
        start = time.monotonic()
        try:
            body = _request(self.base_url, "/api/generate", payload, self.timeout, self.max_response_bytes)
            text = body.get("response")
            if not isinstance(text, str):
                raise LocalModelError("reply has no text 'response' field")
            return text
        except LocalModelError:
            self.errors += 1
            raise
        finally:
            self.seconds += time.monotonic() - start

    def stats(self) -> Dict[str, Any]:
        return {"model": self.model, "think": self.think, "calls": self.calls, "errors": self.errors,
                "mean_seconds": round(self.seconds / self.calls, 3) if self.calls else None}
