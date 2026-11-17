import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTPUT = Path(__file__).resolve().parent

PROGRAMS = [
    ("Sequential", "sequential.txt"),
    ("Static", "static.txt"),
    ("Dynamic", "dynamic.txt"),
    ("Approach 1", "approach1.txt"),
    ("Approach 2", "approach2.txt"),
    ("Approach 4", "approach4.txt"),
    ("Bit-parallel", "hwt_bitparallel.txt"),
    ("SIMD", "hwt_simd.txt"),
]
TIME_PATTERN = re.compile(
    r"(?:Sequential execution time|Static execution time|"
    r"Dynamic execution time|Execution time):\s*([\d.]+)\s*ms"
)


def main():
    measurements = []
    for label, filename in PROGRAMS:
        text = (RESULTS / filename).read_text(encoding="utf-8")
        times = [float(value) for value in TIME_PATTERN.findall(text)]
        if not times:
            raise ValueError(f"No execution times found in {filename}")
        measurements.append((label, times))

    csv_path = OUTPUT / "runtime_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Program", "Runs", "Mean runtime (ms)", "Std dev (ms)"])
        for label, times in measurements:
            writer.writerow(
                [
                    label,
                    len(times),
                    f"{np.mean(times):.3f}",
                    f"{np.std(times):.3f}",
                ]
            )

    ranked = sorted(
        (
            (label, float(np.mean(times)), float(np.std(times)))
            for label, times in measurements
        ),
        key=lambda item: item[1],
    )
    labels = [item[0] for item in ranked]
    means = np.array([item[1] for item in ranked])
    deviations = np.array([item[2] for item in ranked])
    colors = plt.cm.viridis(np.linspace(0.18, 0.88, len(ranked)))

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=0.19, right=0.94, top=0.84, bottom=0.12)
    fig.suptitle(
        "Runtime comparison of all 8 implementations",
        x=0.19,
        y=0.97,
        ha="left",
        fontsize=16,
        weight="bold",
    )
    fig.text(
        0.19,
        0.925,
        "1,000 sequence pairs  |  five recorded runs per program  |  lower is faster",
        fontsize=10,
        color="#475467",
    )
    positions = np.arange(len(ranked))
    bars = ax.barh(
        positions,
        means,
        xerr=deviations,
        color=colors,
        edgecolor="white",
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#344054"},
    )

    ax.set_xscale("log")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean execution time (ms, logarithmic scale)")
    ax.set_xlim(left=10, right=float(np.max(means + deviations)) * 2.0)
    ax.grid(axis="x", which="both", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)

    for bar, mean, deviation in zip(bars, means, deviations):
        ax.annotate(
            f"{mean:,.1f} ± {deviation:,.1f} ms",
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color="#1d2939",
        )

    fig.savefig(OUTPUT / "runtime_comparison.png", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT / 'runtime_comparison.png'}")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
