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
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
    / "test"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "efficientnet_b0_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "efficientnet_b0_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def load_model(class_names):

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model = models.efficientnet_b0(
        weights=None
    )

    input_features = (
        model.classifier[1].in_features
    )

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            input_features,
            len(class_names),
        ),
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)
    model.eval()

    return model


def main():

    print("=" * 70)
    print("AgriVision AI - EfficientNet-B0 Test Evaluation")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Model: {MODEL_PATH}")

    dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print()
    print(
        f"Test images: {len(dataset)}"
    )

    print(
        f"Classes: {len(dataset.classes)}"
    )

    model = load_model(
        dataset.classes
    )

    print(
        "EfficientNet-B0 loaded successfully."
    )

    all_predictions = []
    all_labels = []

    total = 0

    print()
    print("Evaluating...")

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.numpy()
            )

            total += images.size(0)

            if total % 500 < 32:
                print(
                    f"Processed: {total}/"
                    f"{len(dataset)}"
                )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=dataset.classes,
        zero_division=0,
    )

    results = {
        "model": "EfficientNet-B0",
        "test_images": len(dataset),
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "classification_report": report,
    }

    output_path = (
        OUTPUT_DIR
        / "efficientnet_b0_test_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("EFFICIENTNET-B0 TEST RESULTS")
    print("=" * 70)

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall: "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1: "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{macro_f1 * 100:.2f}%"
    )

    print()
    print(
        f"Results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()