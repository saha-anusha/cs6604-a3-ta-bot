"""
Mock vLLM endpoint for local testing and Gradescope Tier-1 sanity checks.

Serves two OpenAI-compatible routes on an ephemeral localhost port:
  - POST /v1/chat/completions  → canned text reply
  - POST /v1/embeddings        → deterministic fixed-dim vectors

Usage
-----
    from mock_endpoint import start_mock_endpoint, stop_mock_endpoint

    base_url, server = start_mock_endpoint()
    # base_url is e.g. "http://127.0.0.1:54321"
    # Use it as your TABot config["endpoint"]
    ...
    stop_mock_endpoint(server)

Or run directly:
    python mock_endpoint.py          # prints the base URL, runs until Ctrl-C
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_DIM = 128  # fixed embedding dimension
CANNED_REPLY = (
    "This is a mock response from the test endpoint. "
    "It does not perform real inference."
)
MODEL_NAME = "mock-model"


# ---------------------------------------------------------------------------
# Deterministic embedding: hash the input text → fixed-dim float vector
# ---------------------------------------------------------------------------
def _deterministic_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """SHA-256-seeded deterministic float vector, always the same for the same input."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand hash to fill `dim` floats by repeated hashing
    raw = b""
    seed = h
    while len(raw) < dim * 4:
        seed = hashlib.sha256(seed).digest()
        raw += seed
    # Unpack as floats in [-1, 1]
    floats = list(struct.unpack(f"<{dim}f", raw[: dim * 4]))
    # Normalise to unit length
    norm = max(sum(x * x for x in floats) ** 0.5, 1e-12)
    return [x / norm for x in floats]


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class _MockHandler(BaseHTTPRequestHandler):
    """Handles /v1/chat/completions and /v1/embeddings."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Suppress default stderr logging."""

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    def _json_response(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        """Health check."""
        if self.path in ("/health", "/v1/models"):
            self._json_response({"status": "ok", "model": MODEL_NAME})
        else:
            self._json_response({"error": "not found"}, 404)

    def do_POST(self) -> None:  # noqa: N802
        body = self._read_body()

        if self.path == "/v1/chat/completions":
            self._handle_completions(body)
        elif self.path == "/v1/embeddings":
            self._handle_embeddings(body)
        else:
            self._json_response({"error": "not found"}, 404)

    # ---- /v1/chat/completions ----
    def _handle_completions(self, body: dict) -> None:
        resp = {
            "id": "mock-cmpl-001",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", MODEL_NAME),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": CANNED_REPLY},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        self._json_response(resp)

    # ---- /v1/embeddings ----
    def _handle_embeddings(self, body: dict) -> None:
        inputs = body.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]

        data = []
        for i, text in enumerate(inputs):
            data.append(
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": _deterministic_embedding(str(text)),
                }
            )

        resp = {
            "object": "list",
            "data": data,
            "model": body.get("model", MODEL_NAME),
            "usage": {"prompt_tokens": len(inputs), "total_tokens": len(inputs)},
        }
        self._json_response(resp)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def start_mock_endpoint(
    host: str = "127.0.0.1", port: int = 0
) -> tuple[str, HTTPServer]:
    """
    Start the mock endpoint on an ephemeral port (port=0).

    Returns
    -------
    (base_url, server)
        base_url : str like "http://127.0.0.1:54321"
        server   : HTTPServer — pass to stop_mock_endpoint() when done.
    """
    server = HTTPServer((host, port), _MockHandler)
    actual_port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://{host}:{actual_port}"
    return base_url, server


def stop_mock_endpoint(server: HTTPServer) -> None:
    """Shut down a running mock endpoint."""
    server.shutdown()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    url, srv = start_mock_endpoint()
    print(f"Mock vLLM endpoint running at {url}")
    print("Press Ctrl-C to stop.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
        srv.shutdown()
