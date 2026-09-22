import json
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# =========================
# CONFIGURATION
# =========================

TEST_DIR = Path("data/processed/classification/test")
CHECKPOINT = Path("models/classification/resnet18_best.pth")
OUTPUT = Path("data/reports/resnet18_confusion_matrix.png")

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")


# =========================
# TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# DATASET
# =========================

dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

class_names = dataset.classes
num_classes = len(class_names)

print(f"Test images: {len(dataset)}")
print(f"Classes: {num_classes}")

print("\nClass mapping:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# =========================
# MODEL
# =========================

model = models.resnet18(weights=None)

model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, num_classes)
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("\nModel loaded successfully.")


# =========================
# PREDICTIONS
# =========================

all_labels = []
all_predictions = []

print("\nRunning test-set predictions...")

with torch.no_grad():

    for images, labels in loader:

        images = images.to(device)

        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)

        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())


all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=np.arange(num_classes)
)

print("\nConfusion Matrix:")
print(cm)


# =========================
# PLOT
# =========================

fig, ax = plt.subplots(figsize=(14, 12))

display_labels = [
    name.replace("__", " ").replace("_", " ")
    for name in class_names
]

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=display_labels
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d",
    cmap="Blues",
    colorbar=True
)

ax.set_title(
    "AgriVision AI - ResNet18 Confusion Matrix",
    fontsize=16,
    pad=20
)

ax.set_xlabel("Predicted Disease / Class")
ax.set_ylabel("Actual Disease / Class")

plt.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT,
    dpi=250,
    bbox_inches="tight"
)

print(f"\nConfusion matrix saved to:")
print(OUTPUT)

plt.show()
