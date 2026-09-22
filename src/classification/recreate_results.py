import json
from pathlib import Path

REPORT = Path("data/reports/resnet18_results.json")
REPORT.parent.mkdir(parents=True, exist_ok=True)

history = {
    "history": [
        {
            "epoch": 1,
            "train_accuracy": 89.25,
            "val_accuracy": 97.48
        },
        {
            "epoch": 2,
            "train_accuracy": None,
            "val_accuracy": 97.87
        },
        {
            "epoch": 3,
            "train_accuracy": 96.54,
            "val_accuracy": 97.84
        },
        {
            "epoch": 4,
            "train_accuracy": 97.22,
            "val_accuracy": 97.55
        },
        {
            "epoch": 5,
            "train_accuracy": 97.42,
            "val_accuracy": 98.93
        },
        {
            "epoch": 6,
            "train_accuracy": 98.30,
            "val_accuracy": 97.03
        },
        {
            "epoch": 7,
            "train_accuracy": None,
            "val_accuracy": None
        },
        {
            "epoch": 8,
            "train_accuracy": 98.12,
            "val_accuracy": 97.64
        },
        {
            "epoch": 9,
            "train_accuracy": 99.20,
            "val_accuracy": 99.48
        }
    ]
}

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=4)

print(f"Created: {REPORT}")
