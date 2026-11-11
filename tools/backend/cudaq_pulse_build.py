# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The build backend of cudaq-pulse: scikit-build-core's, with an sdist that
stores the file each of its symbolic links points to.

packaging/cudaq-pulse-tools links LICENSE, NOTICE and LICENSES/ to the
repository's texts. scikit-build-core stores a file symlink as a symlink
unless sdist.resolve-symlinks says otherwise, which needs minimum-version 1.0,
so this backend replaces each one in the sdist it builds with a copy of the
file it points to in the same sdist. A link to anything else fails the build.
"""

from __future__ import annotations

import gzip
import io
import posixpath
import tarfile
from pathlib import Path

from scikit_build_core import build as _scikit_build
from scikit_build_core.build import *  # noqa: F403

__all__ = list(_scikit_build.__all__)


def dereference_sdist(archive: Path) -> None:
    """Rewrite the sdist at ``archive`` with each symlink replaced by a copy
    of its target, keeping the member order and the gzip timestamp."""
    with gzip.GzipFile(archive) as compressed:
        data = compressed.read()
        mtime = compressed.mtime
    out = io.BytesIO()
    with (
        tarfile.open(fileobj=io.BytesIO(data)) as source,
        tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT) as sdist,
    ):
        members = source.getmembers()
        by_name = {m.name: m for m in members}
        for member in members:
            content = member
            if member.issym():
                path = posixpath.join(posixpath.dirname(member.name), member.linkname)
                content = by_name.get(posixpath.normpath(path))
                if content is None or not content.isfile():
                    raise RuntimeError(
                        f"{member.name} links to {member.linkname}, which is not a file "
                        "in the sdist"
                    )
                member = content.replace(name=member.name)
                # Its own (a path header) would rename the copy back.
                member.pax_headers = {}
            sdist.addfile(member, source.extractfile(content) if content.isfile() else None)
    with gzip.GzipFile(archive, "wb", compresslevel=9, mtime=mtime) as compressed:
        compressed.write(out.getvalue())


def build_sdist(sdist_directory: str, config_settings: dict | None = None) -> str:
    name = _scikit_build.build_sdist(sdist_directory, config_settings)
    dereference_sdist(Path(sdist_directory) / name)
    return name
