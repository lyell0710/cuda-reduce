import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "project-proof" / "data" / "benchmark_results.csv"
FIG_PATH = ROOT / "project-proof" / "docs" / "figures" / "correctness_check.png"
VERSION_ORDER = ("baseline", "v0", "v1", "v2", "v3", "v4")


def load_benchmark_rows():
    with CSV_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def pick_colors(count: int):
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(count)]


rows = load_benchmark_rows()
row_by_version = {row["version"]: row for row in rows}
ordered_rows = [row_by_version[v] for v in VERSION_ORDER if v in row_by_version]
extra_rows = [row for row in rows if row["version"] not in VERSION_ORDER]
plot_rows = ordered_rows + extra_rows

cpu_value = float(plot_rows[0]["cpu_result"])
labels = ["CPU"] + [f"{row['version']} GPU" for row in plot_rows]
values = [cpu_value] + [float(row["gpu_result"]) for row in plot_rows]
colors = pick_colors(len(labels))

plt.figure(figsize=(8.5, 4.5))
bars = plt.bar(labels, values, color=colors)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.5e}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

plt.title("CUDA Reduction Correctness Check")
plt.ylabel("Result Value")
plt.tight_layout()
FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(FIG_PATH, dpi=200)
plt.close()
