from pathlib import Path
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_DIR = Path("data/processed/classification/test")
MODEL_PATH = Path("models/classification/resnet18_best.pth")
REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
BATCH_SIZE = 32

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

print("=" * 70)
print("          AGRIVISION AI - FINAL TEST EVALUATION")
print("=" * 70)

print(f"\nDevice: {DEVICE}")
print(f"Model:  {MODEL_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

classes = checkpoint["classes"]
num_classes = len(classes)

model = models.resnet18(weights=None)

model.fc = nn.Sequential(
    nn.Dropout(p=0.30),
    nn.Linear(512, num_classes)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print(f"\nClasses: {num_classes}")
print(f"Test images: {len(test_dataset):,}")

all_labels = []
all_predictions = []

print("\nRunning inference...")

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(
        test_loader,
        start=1
    ):

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        if batch_index % 20 == 0:

            print(
                f"Processed "
                f"{min(batch_index * BATCH_SIZE, len(test_dataset)):,}"
                f"/{len(test_dataset):,}"
            )

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )
)

print("\n")
print("=" * 70)
print("                 FINAL TEST RESULTS")
print("=" * 70)

print(f"\nTest Accuracy:  {accuracy * 100:.2f}%")
print(f"Test Precision: {precision * 100:.2f}%")
print(f"Test Recall:    {recall * 100:.2f}%")
print(f"Test F1 Score:  {f1 * 100:.2f}%")

print("\n")
print("=" * 70)
print("               CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=classes,
    digits=4,
    zero_division=0
)

print(report)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

results = {
    "model": "ResNet18",
    "checkpoint": str(MODEL_PATH),
    "test_images": len(test_dataset),
    "classes": classes,
    "accuracy": accuracy,
    "precision_weighted": precision,
    "recall_weighted": recall,
    "f1_weighted": f1,
    "confusion_matrix": cm.tolist(),
    "classification_report": report
}

output_path = REPORT_DIR / "resnet18_final_test_results.json"

with open(
    output_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        results,
        f,
        indent=4
    )

print("=" * 70)
print("TEST EVALUATION COMPLETE")
print("=" * 70)

print(f"\nResults saved to:")
print(output_path)
