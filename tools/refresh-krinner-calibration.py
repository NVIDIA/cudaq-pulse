#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Rebuild the shipped Krinner calibration from a working calibration file.

    PYTHONPATH=build/dev/python python tools/refresh-krinner-calibration.py WORKING.json

The working file has the same schema (every leaf a {value, units, kind,
source, uncertainty, cooldown[, method]} object) and may carry provenance
that does not belong in the package: the scripts that produced a value,
working notes and status fields. This tool

- drops the working-only fields and the script names in sources and methods;
- keeps the device-wide charge coupling g_c of calibration A
  (coupling.charge_coupling_g_c.calibration_A, fitted to Hellings et al.'s
  X2-D4 measurement), which every pair and the CZ knobs use, and stores each
  pair's idle exchange with it (coupling.charge_coupling_g_c.idle_exchange);
- refuses to write if any local path, script name or working note remains.

Per-pair CZ knobs (cz.knobs.per_pair) are copied as they are, so rerunning it
after the working file gains pairs refreshes them.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "core/frontend/cudaq_pulse/targets/calibrations/krinner17.json"

# Fields that only describe the working file.
_DROP = {
    "generated_by",
    "status",
    "todo",
    "redigitized_source",
    "recommended",
    "per_pair_status",
    "per_pair_files",
}
_SCRIPT = re.compile(r"(?<![\w.])[\w./-]+\.py\b")
_NOTES = (
    (re.compile(r"\s*\(user decision[^)]*\)"), ""),
    (re.compile(r"\bdraft\b", re.IGNORECASE), ""),
    (re.compile(r" and the digitizer script"), ""),
    (re.compile(r"\s*\((?:two_transmon|cz_pairs|cz_snz)[\w.]*\)"), ""),
    (re.compile(r"chosen by the user \(\d{4}-\d{2}-\d{2}\)"), "chosen at calibration"),
)
_FORBIDDEN = re.compile(
    r"\.py\b|/Users/|_scratch|user decision|\bTODO\b|\bdraft\b|two_transmon|cz_pairs|cz_snz"
    r"|calibration/|by the user",
    re.IGNORECASE,
)

TITLE = "Krinner et al., Nature 605, 669 (2022), arXiv:2112.03708: device calibration"
# The charge coupling every pair uses.
DEVICE_COUPLING = "calibration_A"
IDLE_METHOD = (
    "J_qq = g_c <0|n_a|1> <0|n_d|1> of the charge-coupled pair with the device-wide g_c "
    "(coupling.charge_coupling_g_c.calibration_A), at the qubits' idle sweet spots"
)


def _clean_text(text: str) -> str:
    text = _SCRIPT.sub("", text)
    for pattern, replacement in _NOTES:
        text = pattern.sub(replacement, text)
    # Tidy what removing script names leaves behind.
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)
    text = re.sub(r"\(\s*[,;]\s*", "(", text)
    text = re.sub(r"\s*[,;]\s*\)", ")", text)
    text = re.sub(r"\s+([,.;)])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def _clean(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _clean(v) for k, v in node.items() if k not in _DROP}
    if isinstance(node, list):
        return [_clean(v) for v in node]
    if isinstance(node, str):
        return _clean_text(node)
    return node


def _leftovers(node: Any, path: str = "") -> list[str]:
    if isinstance(node, dict):
        return [hit for k, v in node.items() for hit in _leftovers(v, f"{path}/{k}")]
    if isinstance(node, list):
        return [hit for i, v in enumerate(node) for hit in _leftovers(v, f"{path}/{i}")]
    if isinstance(node, str) and _FORBIDDEN.search(node):
        return [f"{path}: {node[:100]}"]
    return []


_IDLE_FLUX = {"upper": 0.0, "lower": 0.5}


def _sweet_spot(calibration: dict[str, Any], label: str) -> str:
    spot: str = calibration["qubits"][label]["sweet_spot_idle"]
    return spot


def _idle_exchange(calibration: dict[str, Any]) -> dict[str, Any]:
    """Each pair's exchange at the idle points with the device-wide g_c, stored
    so that building the target needs no diagonalization."""
    from cudaq_pulse.targets import _krinner17 as published
    from cudaq_pulse.targets.transmon_model import ChargeCoupledPair, TransmonCircuit

    def circuit(label: str) -> TransmonCircuit:
        q = calibration["qubits"][label]
        return TransmonCircuit(
            q["ec"]["value"], q["ej_max"]["value"], q["asymmetry_d"]["value"], label
        )

    g_c = calibration["coupling"]["charge_coupling_g_c"][DEVICE_COUPLING]["value"]
    exchange = {}
    for edge in published.EDGES:
        data, auxiliary = circuit(edge.data), circuit(edge.auxiliary)
        idle = (
            _IDLE_FLUX[_sweet_spot(calibration, edge.data)],
            _IDLE_FLUX[_sweet_spot(calibration, edge.auxiliary)],
        )
        exchange[f"{edge.auxiliary}-{edge.data}"] = {
            "value": ChargeCoupledPair(data, auxiliary, g_c).j_qq_hz(*idle),
            "units": "Hz",
            "kind": "fitted",
            "source": "derived from the circuits and the device-wide charge coupling",
            "uncertainty": None,
            "cooldown": calibration["coupling"]["charge_coupling_g_c"][DEVICE_COUPLING]["cooldown"],
            "method": IDLE_METHOD,
            # (auxiliary, data) idle frequencies the value holds for.
            "idle_frequencies_hz": [auxiliary.f01_hz(idle[1]), data.f01_hz(idle[0])],
        }
    return exchange


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("working", type=Path, help="working calibration file")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    calibration = _clean(json.loads(args.working.read_text(encoding="utf-8")))
    calibration["_schema"]["title"] = TITLE
    calibration["_schema"]["target"] = "transmon_krinner_17q"
    calibration["_schema"]["default_cooldown"] = "K22-main"
    couplings = calibration["coupling"]["charge_coupling_g_c"]
    couplings["used"] = DEVICE_COUPLING
    couplings["idle_exchange"] = _idle_exchange(calibration)
    calibration["cz"]["knobs"]["units_note"] += (
        "; Calibration.flux_for_detuning and detuning_for_flux convert detunings to flux quanta "
        "for qubits with flux_quanta_capable"
    )
    left = _leftovers(calibration)
    if left:
        print("not written; non-public text remains:\n  " + "\n  ".join(left), file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=1, ensure_ascii=False) + "\n")
    shown = args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output
    print(f"wrote {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
