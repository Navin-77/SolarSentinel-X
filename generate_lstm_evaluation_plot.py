import matplotlib.pyplot as plt
import numpy as np

cm = np.array([
    [20, 0, 0],
    [0, 19, 0],
    [1, 2, 17]
])

classes = ["Critical", "Healthy", "Warning"]

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(cm)

ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("LSTM Confusion Matrix")

for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()

plt.savefig(
    "results/lstm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("LSTM confusion matrix generated successfully!")
print("Saved as: results/lstm_confusion_matrix.png")