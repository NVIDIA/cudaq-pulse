#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Installs the console: the newest nvqconsole-bundle artifact of this branch
# when the codespace can download it (a minute), otherwise cudaq-pulse built
# from this checkout (several minutes).

set -euo pipefail

bundle="$HOME/nvqconsole-bundle"
repo="${GITHUB_REPOSITORY:-NVIDIA/cudaq-pulse}"
branch="$(git rev-parse --abbrev-ref HEAD)"

fetch_bundle() {
  local run
  run="$(gh run list --repo "$repo" --workflow nvqconsole.yml --branch "$branch" \
    --status success --limit 1 --json databaseId --jq '.[0].databaseId')" || return 1
  [[ -n "$run" ]] || return 1
  rm -rf "$bundle"
  gh run download "$run" --repo "$repo" --name nvqconsole-bundle --dir "$bundle"
}

if fetch_bundle; then
  head -1 "$bundle/README.txt"
  python -m pip install --find-links "$bundle/wheels" cudaq-pulse
  exit 0
fi

echo "No nvqconsole-bundle artifact could be downloaded; building cudaq-pulse from source."
rm -rf "$bundle"
archive="$(python tools/fetch-mlir-toolchain.py --print-archive-name)"
tag="$(python -c 'import json; print(json.load(open("tools/mlir-toolchain.lock.json"))["release_tag"])')"
mkdir -p .cache/mlir/downloads
gh release download "$tag" --repo "$repo" --pattern "$archive" --dir .cache/mlir/downloads --clobber
python tools/fetch-mlir-toolchain.py
python -m pip install .
(cd tools/console/web && npm ci --no-audit --no-fund && npm run build)
