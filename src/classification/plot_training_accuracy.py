import json
import matplotlib.pyplot as plt

REPORT = "data/reports/resnet18_results.json"
OUTPUT = "data/reports/resnet18_accuracy_curve.png"

with open(REPORT, "r", encoding="utf-8") as f:
    data = json.load(f)

history = data["history"]

epochs = [x["epoch"] for x in history]
train_acc = [x["train_accuracy"] for x in history]
val_acc = [x["val_accuracy"] for x in history]

plt.figure(figsize=(10, 6))

plt.plot(epochs, train_acc, marker="o", label="Training Accuracy")
plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("AgriVision AI - ResNet18 Training Accuracy")
plt.ylim(80, 100.5)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(OUTPUT, dpi=200)
print(f"Graph saved to: {OUTPUT}")

plt.show()
