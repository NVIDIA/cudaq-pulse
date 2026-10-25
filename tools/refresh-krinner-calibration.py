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
- recomputes the per-pair charge coupling g_c (coupling.charge_coupling_g_c
  .per_pair): each pair's |11> <-> |20> coupling equals its published value at
  its interaction point (7 MHz device-wide, or a per-pair override);
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
_DROP = {"generated_by", "status", "todo", "redigitized_source", "recommended", "per_pair_status"}
_SCRIPT = re.compile(r"(?<![\w.])[\w./-]+\.py\b")
_NOTES = (
    (re.compile(r"\s*\(user decision[^)]*\)"), ""),
    (re.compile(r"\bdraft\b", re.IGNORECASE), ""),
    (re.compile(r" and the digitizer script"), ""),
)
_FORBIDDEN = re.compile(r"\.py\b|/Users/|_scratch|user decision|\bTODO\b|\bdraft\b", re.IGNORECASE)

TITLE = "Krinner et al., Nature 605, 669 (2022), arXiv:2112.03708: device calibration"
COUPLING_METHOD = (
    "per pair: g_c such that half the smallest |11>-|20> splitting of the charge-coupled pair "
    "(charge basis, 5 levels each, g_c n_A n_D, no RWA) equals the published g_11_20 with the data "
    "qubit at the pair's interaction frequency and the auxiliary's 1-2 transition there "
    "(cudaq_pulse.targets.transmon_model.anchored_charge_coupling_hz)"
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


def _per_pair_coupling(calibration: dict[str, Any]) -> dict[str, Any]:
    from cudaq_pulse.targets import _krinner17 as published
    from cudaq_pulse.targets.transmon_model import (
        ChargeCoupledPair,
        TransmonCircuit,
        anchored_charge_coupling_hz,
    )

    def circuit(label: str) -> TransmonCircuit:
        q = calibration["qubits"][label]
        return TransmonCircuit(
            q["ec"]["value"], q["ej_max"]["value"], q["asymmetry_d"]["value"], label
        )

    overrides = calibration["coupling"]["overrides"]
    default = calibration["coupling"]["device_default"]["g_11_20"]
    per_pair = {}
    for edge in published.EDGES:
        name = f"{edge.auxiliary}-{edge.data}"
        anchor = overrides.get(name, {}).get("g_11_20", default)
        at_hz = anchor.get("at_interaction_frequency_hz", edge.interaction_frequency_hz)
        data, auxiliary = circuit(edge.data), circuit(edge.auxiliary)
        g_c = anchored_charge_coupling_hz(data, auxiliary, anchor["value"], at_hz)
        idle = (
            _IDLE_FLUX[_sweet_spot(calibration, edge.data)],
            _IDLE_FLUX[_sweet_spot(calibration, edge.auxiliary)],
        )
        pair = ChargeCoupledPair(data, auxiliary, g_c)
        per_pair[name] = {
            "value": g_c,
            "units": "Hz",
            "kind": "fitted",
            "source": anchor["source"],
            "uncertainty": None,
            "cooldown": anchor["cooldown"],
            "method": COUPLING_METHOD,
            "g_11_20_hz": anchor["value"],
            "at_interaction_frequency_hz": at_hz,
            # The exchange at the idle points, (auxiliary, data) frequencies.
            "idle_frequencies_hz": [auxiliary.f01_hz(idle[1]), data.f01_hz(idle[0])],
            "idle_exchange_hz": pair.j_qq_hz(*idle),
        }
    return per_pair


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
    couplings["per_pair"] = _per_pair_coupling(calibration)
    couplings["used"] = "per_pair"
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
