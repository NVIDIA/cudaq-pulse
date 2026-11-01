# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""A local HTTP server for the console: JSON routes and the built web app.

It runs the code it is sent, so it listens on the loopback interface only.
"""

from __future__ import annotations

import gzip
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
            "experiments": adapter.experiments(),
        },
        "/api/example": lambda q: {"source": adapter.example_source(q["file"])},
        "/api/device": lambda q: adapter.device_info(q["target"]),
    }


def _example_request(name: str) -> dict[str, Any]:
    """The request the compiler explorer sends for an example at its defaults."""
    preset = adapter.EXAMPLE_PRESETS[name]
    return {
        "source": adapter.example_source(name),
        "kernel": preset["kernel"],
        "args": preset["args"],
        "target": adapter.DEFAULT_TARGET,
        "passes": adapter.default_passes(),
        "schedule": "alap",
        "llvm": False,
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
        "/api/run": lambda body: adapter.run_experiment(body["experiment"], body.get("params", {})),
    }


def _streams() -> dict[str, Callable[[dict[str, str]], Iterator[dict[str, Any]]]]:
    return {
        "/api/sweeps": lambda q: adapter.device_sweeps(
            q.get("target", adapter.DEFAULT_TARGET), int(q.get("seed", 3))
        ),
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

    def _send(self, data: bytes, content_type: str, cache: str, status: int = 200) -> None:
        gzipped = len(data) > 2048 and "gzip" in self.headers.get("Accept-Encoding", "")
        if gzipped:
            data = gzip.compress(data, compresslevel=1)
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        if gzipped:
            self.send_header("Content-Encoding", "gzip")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, allow_nan=False, default=str, separators=(",", ":")).encode()
        self._send(data, "application/json", "no-store", status)

    def _fail(self, error: BaseException, status: int = 500) -> None:
        self._json(
            {"ok": False, "error": {"kind": type(error).__name__, "message": str(error)}}, status
        )

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if url.path in self.STREAMS:
            self._stream(self.STREAMS[url.path](query))
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

    def _stream(self, events: Iterator[dict[str, Any]]) -> None:
        """Server-sent events, one per finished piece of work."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for event in events:
                # A cached run is replayed at a pace the eye can follow.
                if event.get("cached") and event.get("type") == "qudit":
                    time.sleep(0.1)
                data = json.dumps(event, default=str, separators=(",", ":"))
                self.wfile.write(f"data: {data}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _static(self, path: str) -> None:
        if not WEB_DIST.exists():
            self._json({"ok": False, "message": "web app not built: run with --build"}, 404)
            return
        file = (WEB_DIST / path.lstrip("/")).resolve()
        if not file.is_file() or WEB_DIST.resolve() not in file.parents:
            file = WEB_DIST / "index.html"
        # Vite hashes asset names, so they never change under a name.
        immutable = file.parent.name == "assets"
        self._send(
            file.read_bytes(),
            mimetypes.guess_type(file.name)[0] or "application/octet-stream",
            "public, max-age=31536000, immutable" if immutable else "no-cache",
        )


def warm_all() -> None:
    """Every run the demo starts with, then the examples' compiles and the
    slow GHZ simulation; cached runs return at once."""
    started = time.perf_counter()
    adapter.warm_up(lambda name, s: print(f"  {name} ready ({s:.1f} s)", flush=True))
    print(f"  all runs ready ({time.perf_counter() - started:.1f} s)", flush=True)
    started = time.perf_counter()
    for name in adapter.EXAMPLE_PRESETS:
        adapter.compile_stages(_example_request(name))
    adapter.warm_simulations(["17_ghz_state_prep.py"], _example_request, points=61, levels=2)
    print(f"  examples ready ({time.perf_counter() - started:.1f} s)", flush=True)


def serve(port: int = 8765, warm: bool = True, runs: Path | None = None) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    print(f"NVQConsole for cudaq-pulse {adapter.version()} on http://127.0.0.1:{port}", flush=True)
    if runs is not None and runs.exists():
        loaded = adapter.load_runs(runs)
        print(
            f"  precomputed runs {'loaded' if loaded else 'skipped: another cudaq-pulse'}",
            flush=True,
        )
    if warm:
        threading.Thread(target=warm_all, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
