import matplotlib.pyplot as plt

labels = ["CPU", "baseline GPU", "v0 GPU"]
values = [1.67772e7, 1.67772e7, 1.67772e7]

plt.figure(figsize=(7, 4))
bars = plt.bar(labels, values, color=["#54A24B", "#4C78A8", "#F58518"])

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.5e}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.title("CUDA Reduction Correctness Check")
plt.ylabel("Result Value")
plt.tight_layout()
plt.savefig("project-proof/docs/figures/correctness_check.png", dpi=200)
plt.show()
