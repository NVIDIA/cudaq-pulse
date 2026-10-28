# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A local HTTP server for the console: JSON routes, server-sent events for
calibration runs, and the built web app.

It runs the code it is sent, so it listens on the loopback interface only.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from . import adapter

WEB_DIST = Path(__file__).resolve().parent.parent / "web" / "dist"


def _get_routes() -> dict[str, Callable[[dict[str, str]], Any]]:
    return {
        "/api/health": lambda _: {"ok": True, "version": adapter.version()},
        "/api/meta": lambda _: {
            "version": adapter.version(),
            "targets": adapter.targets(),
            "passes": adapter.pass_names(),
            "default_passes": adapter.default_passes(),
            "schedules": adapter.SCHEDULES,
            "examples": adapter.examples(),
        },
        "/api/example": lambda q: {"source": adapter.example_source(q["file"])},
        "/api/device": lambda q: adapter.device_info(q["target"]),
    }


def _post_routes() -> dict[str, Callable[[dict[str, Any]], Any]]:
    def kernels(body: dict[str, Any]) -> Any:
        try:
            return {"ok": True, "kernels": adapter.kernels(body["source"])}
        except adapter.SourceError as error:
            return {"ok": False, "error": error.as_dict()}

    return {
        "/api/kernels": kernels,
        "/api/compile": adapter.compile_stages,
        "/api/simulate": adapter.simulate_populations,
    }


def _streams() -> dict[str, Callable[[dict[str, str]], Iterator[dict[str, Any]]]]:
    return {
        "/api/calibration/drift": lambda q: adapter.drift_tracking(
            seed=int(q.get("seed", 21)), iterations=int(q.get("iterations", 6000))
        ),
        "/api/calibration/search": lambda q: adapter.binary_search(rounds=int(q.get("rounds", 15))),
        "/api/calibration/chain": lambda _: adapter.calibration_chain(),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "nvqconsole"
    GET = _get_routes()
    POST = _post_routes()
    STREAMS = _streams()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if not self.path.startswith("/api/health"):
            print(
                f"  {self.command} {self.path.split('?')[0]} {args[1] if len(args) > 1 else ''}",
                flush=True,
            )

    def _json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, allow_nan=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _fail(self, error: BaseException, status: int = 500) -> None:
        self._json(
            {"ok": False, "error": {"kind": type(error).__name__, "message": str(error)}}, status
        )

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if url.path in self.STREAMS:
            self._stream(self.STREAMS[url.path](query), float(query.get("delay", 0.0)))
            return
        route = self.GET.get(url.path)
        if route is not None:
            try:
                self._json(route(query))
            except (KeyError, FileNotFoundError) as error:
                self._fail(error, 404)
            except Exception as error:  # noqa: BLE001
                self._fail(error)
            return
        self._static(url.path)

    def do_POST(self) -> None:  # noqa: N802
        route = self.POST.get(urlparse(self.path).path)
        if route is None:
            self._json({"ok": False}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            self._json(route(body))
        except Exception as error:  # noqa: BLE001
            self._fail(error)

    def _stream(self, events: Iterator[dict[str, Any]], delay: float) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        delay = min(max(delay, 0.0), 1.0)
        try:
            for event in events:
                self.wfile.write(f"data: {json.dumps(event, default=str)}\n\n".encode())
                self.wfile.flush()
                if delay and event.get("type") in ("point", "step"):
                    time.sleep(delay)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:  # noqa: BLE001
            message = {"type": "error", "kind": type(error).__name__, "message": str(error)}
            self.wfile.write(f"data: {json.dumps(message)}\n\n".encode())

    def _static(self, path: str) -> None:
        if not WEB_DIST.exists():
            self._json({"ok": False, "message": "web app not built: run with --build"}, 404)
            return
        file = (WEB_DIST / path.lstrip("/")).resolve()
        if not file.is_file() or WEB_DIST.resolve() not in file.parents:
            file = WEB_DIST / "index.html"
        data = file.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(port: int = 8765, warm: bool = True) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    print(f"NVQConsole for cudaq-pulse {adapter.version()} on http://127.0.0.1:{port}", flush=True)
    if warm:

        def warm_up() -> None:
            started = time.perf_counter()
            adapter.warm_up()
            print(f"  calibration chain ready ({time.perf_counter() - started:.1f} s)", flush=True)

        threading.Thread(target=warm_up, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
