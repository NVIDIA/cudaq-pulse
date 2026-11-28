# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Draw the figure of the "Idle Replacement" tutorial.

Usage: python docs/tutorials/figures/idle_replacement.py

Needs matplotlib (the ``[viz]`` extra). Writes, with its ``-dark`` twin,
next to this file:

- ``conditional_phase.svg``, D5's conditional phase from Z2's readout with
  each fill of the page.
"""

from _style import page_namespace, plot_axes, save_themed


def conditional_phase(page) -> None:
    pulse, compiled, d5 = page["pulse"], page["compiled"], page["d5"]
    measure, spaced = page["conditional_phase"], page["evenly_spaced"]
    labels = ["no fill", "1 flip", "2 flips", "4 flips"]
    values = [measure(compiled)]
    for count in (1, 2, 4):
        rule = pulse.IdleReplacement(spaced(count), threshold=160 * pulse.ns, qudits=[d5])
        values.append(measure(compiled.replace_idle(rule)))

    def figure(p):
        fig, ax = plot_axes(p, height_in=2.6)
        positions = range(len(values))
        ax.bar(positions, values, width=0.5, color=p.drive)
        ax.axhline(0, color=p.rule, lw=0.8)
        for x, value in zip(positions, values, strict=True):
            ax.annotate(
                f"{value:+.2f}°".replace("-", "\u2212"),
                (x, value),
                xytext=(0, 4 if value >= 0 else -4),
                textcoords="offset points",
                ha="center",
                va="bottom" if value >= 0 else "top",
                color=p.text,
                fontsize=8.5,
            )
        ax.set_xticks(list(positions), labels)
        ax.set_ylabel("D5 conditional phase (degrees)")
        top = max(abs(v) for v in values) * 1.3
        ax.set_ylim(-top, top)
        return fig

    alt = (
        "D5's conditional phase from Z2's readout, in degrees, for each fill of its idle "
        "window: "
        + ", ".join(f"{label} {value:+.2f}" for label, value in zip(labels, values, strict=True))
        + ". One flip cancels most of it, two flips overshoot, four cancel it again."
    )
    save_themed(figure, "conditional_phase", title="Conditional phase by fill", desc=alt)


def main() -> None:
    conditional_phase(page_namespace("idle_replacement.rst"))


if __name__ == "__main__":
    main()
