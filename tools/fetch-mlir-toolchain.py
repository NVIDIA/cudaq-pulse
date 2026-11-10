#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fetch, verify, and unpack the pinned MLIR toolchain into .cache/mlir/.

The toolchain archives are built by tools/build-mlir-toolchain.sh and pinned
by name and SHA-256 in tools/mlir-toolchain.lock.json. Every archive is
checksum-verified before it is unpacked; there is no unverified path.

    python tools/fetch-mlir-toolchain.py                  # download + unpack
    python tools/fetch-mlir-toolchain.py --print-prefix   # CMAKE_PREFIX_PATH
    python tools/fetch-mlir-toolchain.py --from-file X.tar.xz [--sha256 HEX]

Uses only the Python standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import platform
import shutil
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = REPO_ROOT / "tools" / "mlir-toolchain.lock.json"
DEFAULT_CACHE = REPO_ROOT / ".cache" / "mlir"
MARKER = ".cudaq-pulse-sha256"
# Seconds a download may wait to connect, or for more data, before it fails.
DEFAULT_TIMEOUT = 60.0
# The platforms .github/workflows/mlir-toolchain.yml builds a toolchain for.
PLATFORMS = ("linux-x86_64", "linux-aarch64", "macos-aarch64")


class FetchError(Exception):
    pass


def host_platform() -> str:
    systems = {"Linux": "linux", "Darwin": "macos"}
    machines = {"x86_64": "x86_64", "AMD64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
    system, machine = platform.system(), platform.machine()
    plat = f"{systems.get(system, system)}-{machines.get(machine, machine)}"
    if plat not in PLATFORMS:
        raise FetchError(
            f"no MLIR toolchain is built for {system}/{machine}; prebuilt toolchains "
            f"exist for {', '.join(PLATFORMS)}. Build one with "
            f"tools/build-mlir-toolchain.sh and configure with -DMLIR_DIR=<prefix>/lib/cmake/mlir."
        )
    return plat


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock() -> dict:
    return json.loads(LOCK_FILE.read_text())


def toolchain_name(lock: dict, plat: str) -> str:
    return f"mlir-toolchain-{lock['llvm_version']}-{plat}"


def expected_sha256(lock: dict, archive_name: str, explicit: str | None) -> str:
    pinned = lock["assets"].get(archive_name)
    if explicit and pinned and explicit.lower() != pinned:
        raise FetchError(
            f"--sha256 {explicit} contradicts the pinned checksum {pinned} "
            f"for {archive_name} in {LOCK_FILE.name}"
        )
    if explicit:
        return explicit.lower()
    if pinned:
        return pinned
    raise FetchError(
        f"{LOCK_FILE.name} pins no checksum for {archive_name}. The toolchain "
        f"for this platform has not been published yet: run the "
        f"mlir-toolchain workflow (or tools/build-mlir-toolchain.sh) and record "
        f"the archive checksum in {LOCK_FILE.name}, or pass --from-file with "
        f"--sha256 for a toolchain you built yourself."
    )


def download_error(url: str, error: Exception, timeout: float) -> FetchError:
    """The FetchError that explains why downloading `url` failed with `error`."""
    by_hand = (
        "download the archive another way (for example with `gh release download`) "
        "and pass it with --from-file"
    )
    if isinstance(error, urllib.error.HTTPError):
        if error.code == 404:
            return FetchError(
                f"download of {url} failed: HTTP 404, the archive is not published "
                f"there or the release is not public; {by_hand}"
            )
        return FetchError(f"download of {url} failed: HTTP {error.code} {error.reason}")
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    if isinstance(reason, TimeoutError):
        return FetchError(
            f"download of {url} timed out: no data for {timeout:g} s. Check the "
            f"network or proxy, raise --timeout, or {by_hand}"
        )
    if isinstance(error, urllib.error.URLError):
        return FetchError(f"download of {url} failed: cannot connect ({reason}); {by_hand}")
    return FetchError(f"download of {url} failed: {error}")


def download(url: str, destination: Path, timeout: float) -> None:
    print(f"Downloading {url}", file=sys.stderr)
    partial = destination.with_name(destination.name + ".part")
    try:
        with (
            urllib.request.urlopen(url, timeout=timeout) as response,
            partial.open("wb") as handle,
        ):
            shutil.copyfileobj(response, handle)
            received = handle.tell()
            expected = response.headers.get("Content-Length")
        if expected is not None and expected.isdigit() and received != int(expected):
            raise FetchError(
                f"download of {url} was cut off after {received} of {expected} bytes; "
                f"run the fetch again"
            )
    except (OSError, http.client.HTTPException) as error:
        partial.unlink(missing_ok=True)
        raise download_error(url, error, timeout) from error
    except FetchError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(destination)


def unpack(archive: Path, sha256: str, name: str, cache_dir: Path) -> Path:
    actual = sha256_of(archive)
    if actual != sha256:
        raise FetchError(
            f"checksum mismatch for {archive}\n  expected {sha256}\n  actual   {actual}"
        )
    target = cache_dir / name
    if (target / MARKER).is_file() and (target / MARKER).read_text().strip() == sha256:
        return target
    if not hasattr(tarfile, "data_filter"):
        raise FetchError("unpacking requires Python >= 3.10.12 (tarfile filters)")
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_dir) as staging:
        with tarfile.open(archive, "r:xz") as tar:
            tar.extractall(staging, filter="data")
        extracted = Path(staging) / name
        if not (extracted / "lib" / "cmake" / "mlir" / "MLIRConfig.cmake").is_file():
            raise FetchError(f"{archive} does not contain {name}/lib/cmake/mlir/MLIRConfig.cmake")
        (extracted / MARKER).write_text(sha256 + "\n")
        if target.exists():
            shutil.rmtree(target)
        extracted.replace(target)
    return target


def positive_seconds(text: str) -> float:
    try:
        seconds = float(text)
    except ValueError:
        seconds = math.nan
    if not seconds > 0 or math.isinf(seconds):
        raise argparse.ArgumentTypeError(f"expected a positive number of seconds, got {text!r}")
    return seconds


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--from-file", type=Path, help="unpack this local archive instead of downloading"
    )
    parser.add_argument(
        "--sha256", help="expected checksum (required with --from-file when unpinned)"
    )
    parser.add_argument("--platform", help="override the host platform, e.g. linux-x86_64")
    parser.add_argument("--base-url", help="override the release download URL")
    parser.add_argument(
        "--timeout",
        type=positive_seconds,
        # A string default goes through `type` too, so a bad variable is reported.
        default=os.environ.get("CUDAQ_PULSE_MLIR_FETCH_TIMEOUT", str(DEFAULT_TIMEOUT)),
        help="seconds the download may wait to connect or for more data "
        f"(default {DEFAULT_TIMEOUT:g}, or CUDAQ_PULSE_MLIR_FETCH_TIMEOUT)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("CUDAQ_PULSE_MLIR_CACHE", DEFAULT_CACHE)),
    )
    parser.add_argument(
        "--print-prefix", action="store_true", help="print the unpacked toolchain path and exit"
    )
    parser.add_argument(
        "--print-archive-name", action="store_true", help="print the release asset name and exit"
    )
    parser.add_argument(
        "--list-platforms",
        action="store_true",
        help="print the platforms with a prebuilt toolchain, one per line, and exit",
    )
    args = parser.parse_args(argv)

    if args.list_platforms:
        print("\n".join(PLATFORMS))
        return 0

    try:
        lock = load_lock()
        plat = args.platform or host_platform()
        name = toolchain_name(lock, plat)
        archive_name = f"{name}.tar.xz"

        if args.print_archive_name:
            print(archive_name)
            return 0

        if args.print_prefix:
            prefix = args.cache_dir / name
            if not (prefix / MARKER).is_file():
                raise FetchError(
                    f"the MLIR toolchain is not unpacked at {prefix}; run "
                    f"`python tools/fetch-mlir-toolchain.py` first"
                )
            print(prefix)
            return 0

        if args.from_file:
            archive = args.from_file.resolve()
            if not archive.is_file():
                raise FetchError(f"{archive} does not exist")
            sha256 = expected_sha256(lock, archive.name, args.sha256)
        else:
            sha256 = expected_sha256(lock, archive_name, args.sha256)
            archive = args.cache_dir / "downloads" / archive_name
            if not archive.is_file() or sha256_of(archive) != sha256:
                archive.parent.mkdir(parents=True, exist_ok=True)
                base_url = (args.base_url or lock["base_url"]).rstrip("/")
                download(f"{base_url}/{archive_name}", archive, args.timeout)

        print(unpack(archive, sha256, name, args.cache_dir))
        return 0
    except FetchError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
