# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Timeline visualization for pulse schedules.

Plots a per-line timeline of the drives, readouts and waits of a compiled
kernel using matplotlib.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from ..compile import CompiledKernel, ScheduledEvent

_DRIVE_COLOR = "#4C72B0"
_READOUT_COLOR = "#DD8452"
_WAIT_COLOR = "#CCCCCC"


def plot_schedule(
    schedule: CompiledKernel | list[ScheduledEvent],
    figsize: tuple[float, float] = (14, 4),
    title: str | None = None,
) -> Figure:
    """Plot a pulse schedule timeline, one row per drive or readout line.

    Args:
        schedule: A scheduled ``CompiledKernel`` or its ``events()``.
        figsize: Figure size (width, height).
        title: Optional figure title.

    Returns:
        matplotlib Figure.
    """
    events = schedule if isinstance(schedule, list) else schedule.events()
    spans: list[tuple[str, int, int, int]] = []
    for ev in events:
        if ev.start_vtu is None or ev.duration_vtu is None:
            raise ValueError("the kernel is not scheduled; evaluate its parameters first")
        spans.append((ev.kind, ev.line, ev.start_vtu, ev.duration_vtu))
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    line_ids: list[int] = []
    for _, line, _, _ in spans:
        if line not in line_ids:
            line_ids.append(line)

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
    ax.set_yticklabels([f"line {lid}" for lid in line_ids])
    ax.set_xlabel("Time (VTU)")
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
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def save_schedule(
    schedule: CompiledKernel | list[ScheduledEvent],
    path: str,
    **kwargs: Any,
) -> None:
    """Plot and save a schedule to a file."""
    fig = plot_schedule(schedule, **kwargs)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
