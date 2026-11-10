# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Timeline visualization for pulse schedules.

Plots a per-line timeline of the drives, readouts and waits of a compiled
kernel using matplotlib.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .._errors import InvalidValueError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..compiler import Lowered, ScheduledEvent

_DRIVE_COLOR = "#4C72B0"
_READOUT_COLOR = "#DD8452"
_WAIT_COLOR = "#CCCCCC"
_SUBMIT_COLOR = "#55A868"
_CLASSICAL_KINDS = ("submit", "answer")


def plot_schedule(
    schedule: Lowered | Sequence[ScheduledEvent],
    figsize: tuple[float, float] = (14, 4),
    title: str | None = None,
) -> Figure:
    """Plot a pulse schedule timeline, one row per drive or readout line.

    The links of classical calls get rows of their own, below the lines: a
    bar while a request is sent and a marker where its answer is taken.

    Args:
        schedule: A compiled kernel call (``kernel.lower(...)``) or its ``schedule``.
        figsize: Figure size (width, height).
        title: Optional figure title.

    Returns:
        matplotlib Figure.
    """
    events = schedule if isinstance(schedule, Sequence) else schedule.schedule
    names = {} if isinstance(schedule, Sequence) else dict(schedule.options.machine.link_names)
    link_label = {v: k for k, v in names.items()}
    # Rows are lines, then the links of classical calls.
    spans: list[tuple[str, tuple[bool, int], int, int]] = []
    for ev in events:
        if ev.start_cycles is None or ev.duration_cycles is None:
            raise InvalidValueError(
                "the module is not scheduled; lower the kernel with its float arguments"
            )
        spans.append(
            (ev.kind, (ev.kind in _CLASSICAL_KINDS, ev.line), ev.start_cycles, ev.duration_cycles)
        )
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    line_ids: list[tuple[bool, int]] = []
    for _, line, _, _ in spans:
        if line not in line_ids:
            line_ids.append(line)
    line_ids.sort(key=lambda row: row[0])

    line_to_row = {lid: i for i, lid in enumerate(line_ids)}
    n_rows = max(len(line_ids), 1)

    fig, ax = plt.subplots(figsize=figsize)

    for kind, line, x, w in spans:
        y = line_to_row[line]
        if kind == "drive":
            rect = mpatches.FancyBboxPatch(
                (x, y - 0.35),
                w,
                0.7,
                boxstyle="round,pad=0.02",
                facecolor=_DRIVE_COLOR,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
            )
            ax.add_patch(rect)
        elif kind == "readout":
            rect = mpatches.FancyBboxPatch(
                (x, y - 0.35),
                w,
                0.7,
                boxstyle="round,pad=0.02",
                facecolor=_READOUT_COLOR,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
            )
            ax.add_patch(rect)
        elif kind == "submit":
            ax.add_patch(
                mpatches.Rectangle(
                    (x, y - 0.35),
                    max(w, 1),
                    0.7,
                    facecolor=_SUBMIT_COLOR,
                    edgecolor="black",
                    linewidth=0.5,
                )
            )
        elif kind == "answer":
            ax.plot([x], [y], marker="v", color=_SUBMIT_COLOR, markersize=7)
        elif kind == "wait" and w > 0:
            wait_rect = mpatches.Rectangle(
                (x, y - 0.3),
                w,
                0.6,
                facecolor=_WAIT_COLOR,
                edgecolor="gray",
                linewidth=0.3,
                alpha=0.5,
                hatch="//",
            )
            ax.add_patch(wait_rect)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(
        [(f"link {link_label.get(n, n)}" if link else f"line {n}") for link, n in line_ids]
    )
    ax.set_xlabel("Time (clock cycles)")
    ax.set_ylim(-0.5, n_rows - 0.5)

    max_time = max((x + w for _, _, x, w in spans), default=100)
    ax.set_xlim(-max_time * 0.02, max_time * 1.05)

    if title:
        ax.set_title(title)

    legend_handles = [
        mpatches.Patch(color=_DRIVE_COLOR, label="Drive"),
        mpatches.Patch(color=_READOUT_COLOR, label="Readout"),
        mpatches.Patch(color=_WAIT_COLOR, label="Wait", hatch="//"),
    ]
    if any(link for link, _ in line_ids):
        legend_handles.append(mpatches.Patch(color=_SUBMIT_COLOR, label="Classical call"))
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def save_schedule(
    schedule: Lowered | Sequence[ScheduledEvent],
    path: str,
    **kwargs: Any,
) -> None:
    """Plot and save a schedule to a file."""
    fig = plot_schedule(schedule, **kwargs)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
