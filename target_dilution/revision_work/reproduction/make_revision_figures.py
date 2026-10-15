#!/usr/bin/env python3
"""Build additive, revision-only figures without replacing the original plots."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TARGET_DIR = Path(__file__).resolve().parents[2]
OUT = TARGET_DIR / "manuscript" / "figures"
CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def figure_2_pair_indexed() -> None:
    source = TARGET_DIR / "phase2/results/verdict.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    pairs = payload["pairs"]
    labels = [f"{p['source']}→{p['target']}" for p in pairs]
    runner_up = [int(p["runner_up_class"]) for p in pairs]
    sink = [
        Counter(int(run["sink_class"]) for run in p["shared_target"]).most_common(1)[0][0]
        for p in pairs
    ]
    matches = [float(p["sink_matches_runner_up_rate"]) for p in pairs]

    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    colors = ["#27864b" if m == 1.0 else "#d89024" if m > 0.0 else "#9c4f4f" for m in matches]
    x = np.arange(len(labels))
    for i, (ru, sc, color) in enumerate(zip(runner_up, sink, colors)):
        ax.plot([i - 0.12, i + 0.12], [ru, sc], color=color, alpha=0.65, linewidth=1.2, zorder=1)
    ax.scatter(x - 0.12, runner_up, marker="D", s=48, facecolors="white", edgecolors=colors,
               linewidths=1.6, label="Clean runner-up", zorder=3)
    ax.scatter(x + 0.12, sink, marker="o", s=52, c=colors, edgecolors="black",
               linewidths=0.35, label="Shared-target modal sink", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"{i}  {name}" for i, name in enumerate(CLASS_NAMES)])
    ax.set_xlim(-0.55, len(labels) - 0.45)
    ax.set_ylim(-0.5, 9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Ordered pair in selected development panel")
    ax.set_ylabel("Fashion-MNIST class")
    ax.set_title("Clean runner-up and shared-target sink by pair (development results)")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(loc="upper left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "fig2b_pair_indexed.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig2b_pair_indexed.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_3_landing() -> None:
    cells = [
        [
            ("Clean, shared target", "Class 0\nrecall 0.953"),
            ("Clean, shared target", "Class 6 (Shirt)\nshare 0.885"),
        ],
        [
            ("Label flip", "Shared target: class 6\nshare 0.851\nMonopoly: class 1, ASR 0.949"),
            ("Label flip", "Class 6 (Shirt)\nat both endpoints"),
        ],
    ]
    colors = [["#d6e3f0", "#f4e4c8"], ["#f4e4c8", "#f4e4c8"]]
    fig, ax = plt.subplots(figsize=(7.3, 4.0))
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2.25)
    ax.set_axis_off()
    ax.set_title("Modal class-0 prediction in development Table 6", pad=18)
    for col, scale in enumerate((r"Honest $\rho=0$", r"Honest $\rho=1$")):
        ax.text(col + 0.5, 2.02, scale, ha="center", va="center", fontsize=11, weight="bold")
    for row, row_label in enumerate(("Clean", "Attack")):
        ax.text(-0.04, 1.46 - row, row_label, ha="right", va="center", fontsize=10, weight="bold")
        y = 1 - row
        for col in range(2):
            rect = plt.Rectangle((col + 0.04, y + 0.08), 0.92, 0.78,
                                 facecolor=colors[row][col], edgecolor="#1f4e79", linewidth=1.2)
            ax.add_patch(rect)
            heading, body = cells[row][col]
            ax.text(col + 0.5, y + 0.70, heading, ha="center", va="center", fontsize=8.5,
                    color="#1f4e79", weight="bold")
            ax.text(col + 0.5, y + 0.39, body, ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_landing_table6.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig_landing_table6.pdf", bbox_inches="tight")
    plt.close(fig)


def _mean_sd(cell: str) -> tuple[float, float]:
    left, right = cell.split("+/-")
    return float(left.strip()), float(right.strip())


def figure_7_test_dose() -> None:
    report = TARGET_DIR / "revision_work/fashion_test_v1/primary_results.md"
    rows = []
    for line in report.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| 0.0 | 0.") and not line.startswith("| 0.0 | 1.") \
                and not line.startswith("| 1.0 | 0.") and not line.startswith("| 1.0 | 1."):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 18:
            continue
        rho = float(cells[0])
        mt = float(cells[2])
        asr, asr_sd = _mean_sd(cells[6])
        entries, entries_sd = _mean_sd(cells[10])
        rows.append((rho, mt, asr, asr_sd, entries, entries_sd))
    if len(rows) != 10:
        raise ValueError(f"Expected 10 held-out rows, found {len(rows)}")

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.2, 4.1))
    styles = {0.0: ("#1f4e79", "o", r"Honest $\rho=0$"),
              1.0: ("#c0392b", "s", r"Honest $\rho=1$")}
    for rho in (0.0, 1.0):
        selected = sorted((row for row in rows if row[0] == rho), key=lambda row: row[1])
        color, marker, label = styles[rho]
        x = np.array([row[1] for row in selected])
        left.errorbar(x, [row[2] for row in selected], yerr=[row[3] for row in selected],
                      fmt=marker + "-", color=color, capsize=3, linewidth=1.6,
                      markersize=5.5, label=label)
        right.errorbar(x, [row[4] for row in selected], yerr=[row[5] for row in selected],
                       fmt=marker + "-", color=color, capsize=3, linewidth=1.6,
                       markersize=5.5, label=label)
    left.set_title("Intended ASR")
    left.set_ylabel("Rate on 1,000 class-0 test images")
    left.set_ylim(-0.03, 1.05)
    left.legend(framealpha=0.95)
    right.set_title("New target entries")
    right.set_ylabel("Images entering class 1 per 1,000")
    right.set_ylim(-25, 1030)
    for ax in (left, right):
        ax.set_xlabel(r"Measured target monopoly $M_t$")
        ax.set_xlim(0.05, 1.05)
        ax.grid(True, alpha=0.22)
    fig.suptitle("Locked Fashion-MNIST test evaluation, pair 0→1")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_fashion_test_dose.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "fig7_fashion_test_dose.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure_2_pair_indexed()
    figure_3_landing()
    figure_7_test_dose()
    print("wrote three revision-only figures; original figures were not replaced")


if __name__ == "__main__":
    main()
