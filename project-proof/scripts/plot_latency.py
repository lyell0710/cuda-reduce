import matplotlib.pyplot as plt

labels = ["baseline", "v0"]
latency_ms = [347.757, 0.46624]

speedup = latency_ms[0] / latency_ms[1]

plt.figure(figsize=(7, 4))
bars = plt.bar(labels, latency_ms, color=["#4C78A8", "#F58518"])

for bar, value in zip(bars, latency_ms):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.5f} ms" if value < 1 else f"{value:.3f} ms",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.title("CUDA Reduction Latency Comparison")
plt.ylabel("Latency (ms)")
plt.xlabel("Version")
plt.tight_layout()
plt.savefig("project-proof/docs/figures/latency_comparison.png", dpi=200)
plt.show()

print(f"Speedup (v0 vs baseline): {speedup:.2f}x")
