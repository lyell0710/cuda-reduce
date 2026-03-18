import matplotlib.pyplot as plt

labels = ["baseline", "v0"]
latency_ms = [347.757, 0.46624]

plt.figure(figsize=(7, 4))
bars = plt.bar(labels, latency_ms, color=["#4C78A8", "#F58518"])
plt.yscale("log")

for bar, value in zip(bars, latency_ms):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.5f} ms" if value < 1 else f"{value:.3f} ms",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.title("CUDA Reduction Latency Comparison (Log Scale)")
plt.ylabel("Latency (ms, log scale)")
plt.xlabel("Version")
plt.tight_layout()
plt.savefig("project-proof/docs/figures/latency_comparison_log.png", dpi=200)
plt.show()
