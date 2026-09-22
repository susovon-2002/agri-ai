"""
AgriVision AI - ResNet18 Confidence Calibration

Uses the validation set to learn a temperature parameter,
then evaluates raw and calibrated confidence on the held-out
test set.

Outputs:
    data/reports/calibration/
        calibration_results.json
        reliability_diagram.png
        confidence_comparison.csv
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import brier_score_loss


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
    / "train"
)

VAL_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
    / "val"
)

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
    / "resnet18_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "calibration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 32
IMAGE_SIZE = 224

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]


transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def load_model():

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            model.fc.in_features,
            len(CLASS_NAMES),
        ),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)
    model.eval()

    return model


@torch.no_grad()
def collect_logits(model, directory):

    dataset = datasets.ImageFolder(
        directory,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    logits_list = []
    labels_list = []

    for images, labels in loader:

        images = images.to(
            DEVICE
        )

        logits = model(
            images
        )

        logits_list.append(
            logits.cpu()
        )

        labels_list.append(
            labels
        )

    logits = torch.cat(
        logits_list
    )

    labels = torch.cat(
        labels_list
    )

    return logits, labels


def optimize_temperature(
    logits,
    labels,
):

    """
    Learn temperature T on validation data.

    T > 1 generally softens overconfident predictions.
    T < 1 generally sharpens underconfident predictions.
    """

    temperature = torch.nn.Parameter(
        torch.ones(
            1,
            device=DEVICE,
        )
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.LBFGS(
        [temperature],
        lr=0.01,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    logits = logits.to(DEVICE)
    labels = labels.to(DEVICE)

    def closure():

        optimizer.zero_grad()

        loss = criterion(
            logits / temperature.clamp(
                min=0.05,
                max=10.0,
            ),
            labels,
        )

        loss.backward()

        return loss

    optimizer.step(
        closure
    )

    value = float(
        temperature.detach().cpu().item()
    )

    return max(
        0.05,
        min(10.0, value),
    )


def expected_calibration_error(
    probabilities,
    labels,
    number_of_bins=15,
):

    confidences = probabilities.max(
        axis=1
    )

    predictions = probabilities.argmax(
        axis=1
    )

    accuracies = (
        predictions == labels
    )

    bins = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    ece = 0.0

    bin_results = []

    for index in range(
        number_of_bins
    ):

        lower = bins[index]
        upper = bins[index + 1]

        if index == number_of_bins - 1:

            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )

        else:

            mask = (
                (confidences >= lower)
                & (confidences < upper)
            )

        count = int(
            mask.sum()
        )

        if count == 0:
            continue

        bin_confidence = float(
            confidences[mask].mean()
        )

        bin_accuracy = float(
            accuracies[mask].mean()
        )

        weight = count / len(
            labels
        )

        ece += weight * abs(
            bin_accuracy
            - bin_confidence
        )

        bin_results.append(
            {
                "lower": float(lower),
                "upper": float(upper),
                "count": count,
                "confidence": bin_confidence,
                "accuracy": bin_accuracy,
            }
        )

    return ece, bin_results


def multiclass_brier_score(
    probabilities,
    labels,
):

    one_hot = np.zeros_like(
        probabilities
    )

    one_hot[
        np.arange(len(labels)),
        labels,
    ] = 1.0

    return float(
        np.mean(
            np.sum(
                (
                    probabilities
                    - one_hot
                ) ** 2,
                axis=1,
            )
        )
    )


def evaluate(
    logits,
    labels,
    temperature,
):

    logits = logits.numpy()
    labels = labels.numpy()

    raw_probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy()

    calibrated_probabilities = torch.softmax(
        torch.tensor(
            logits / temperature
        ),
        dim=1,
    ).numpy()

    raw_predictions = (
        raw_probabilities.argmax(
            axis=1
        )
    )

    calibrated_predictions = (
        calibrated_probabilities.argmax(
            axis=1
        )
    )

    raw_confidence = (
        raw_probabilities.max(
            axis=1
        )
    )

    calibrated_confidence = (
        calibrated_probabilities.max(
            axis=1
        )
    )

    raw_ece, raw_bins = (
        expected_calibration_error(
            raw_probabilities,
            labels,
        )
    )

    calibrated_ece, calibrated_bins = (
        expected_calibration_error(
            calibrated_probabilities,
            labels,
        )
    )

    raw_brier = multiclass_brier_score(
        raw_probabilities,
        labels,
    )

    calibrated_brier = multiclass_brier_score(
        calibrated_probabilities,
        labels,
    )

    results = {
        "temperature": temperature,
        "raw_accuracy": float(
            np.mean(
                raw_predictions == labels
            )
        ),
        "calibrated_accuracy": float(
            np.mean(
                calibrated_predictions
                == labels
            )
        ),
        "raw_mean_confidence": float(
            raw_confidence.mean()
        ),
        "calibrated_mean_confidence": float(
            calibrated_confidence.mean()
        ),
        "raw_ece": raw_ece,
        "calibrated_ece": calibrated_ece,
        "raw_brier_score": raw_brier,
        "calibrated_brier_score": calibrated_brier,
        "raw_bins": raw_bins,
        "calibrated_bins": calibrated_bins,
    }

    return (
        results,
        raw_confidence,
        calibrated_confidence,
        labels,
    )


def create_reliability_diagram(
    raw_bins,
    calibrated_bins,
):

    plt.figure(
        figsize=(9, 7)
    )

    raw_x = [
        item["confidence"]
        for item in raw_bins
    ]

    raw_y = [
        item["accuracy"]
        for item in raw_bins
    ]

    calibrated_x = [
        item["confidence"]
        for item in calibrated_bins
    ]

    calibrated_y = [
        item["accuracy"]
        for item in calibrated_bins
    ]

    plt.plot(
        raw_x,
        raw_y,
        marker="o",
        label="Raw",
    )

    plt.plot(
        calibrated_x,
        calibrated_y,
        marker="o",
        label="Temperature Scaled",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    plt.xlabel(
        "Mean Confidence"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.title(
        "AgriVision AI - Confidence Calibration"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "reliability_diagram.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Reliability diagram saved:\n"
        f"{output_path}"
    )


def main():

    print("=" * 70)
    print(
        "AgriVision AI - Confidence Calibration"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print()
    print("Loading ResNet18...")

    model = load_model()

    print(
        "ResNet18 loaded successfully."
    )

    print()
    print(
        "Collecting validation logits..."
    )

    val_logits, val_labels = (
        collect_logits(
            model,
            VAL_DIR,
        )
    )

    print(
        f"Validation images: "
        f"{len(val_labels)}"
    )

    print()
    print(
        "Optimizing temperature..."
    )

    temperature = optimize_temperature(
        val_logits,
        val_labels,
    )

    print(
        f"Learned temperature: "
        f"{temperature:.6f}"
    )

    print()
    print(
        "Collecting test logits..."
    )

    test_logits, test_labels = (
        collect_logits(
            model,
            TEST_DIR,
        )
    )

    print(
        f"Test images: "
        f"{len(test_labels)}"
    )

    print()
    print(
        "Evaluating calibration..."
    )

    (
        results,
        raw_confidence,
        calibrated_confidence,
        labels,
    ) = evaluate(
        test_logits,
        test_labels,
        temperature,
    )

    print()
    print("=" * 70)
    print(
        "CALIBRATION RESULTS"
    )
    print("=" * 70)

    print(
        f"Temperature: "
        f"{results['temperature']:.6f}"
    )

    print(
        f"Raw accuracy: "
        f"{results['raw_accuracy'] * 100:.2f}%"
    )

    print(
        f"Calibrated accuracy: "
        f"{results['calibrated_accuracy'] * 100:.2f}%"
    )

    print(
        f"Raw mean confidence: "
        f"{results['raw_mean_confidence'] * 100:.2f}%"
    )

    print(
        f"Calibrated mean confidence: "
        f"{results['calibrated_mean_confidence'] * 100:.2f}%"
    )

    print(
        f"Raw ECE: "
        f"{results['raw_ece']:.6f}"
    )

    print(
        f"Calibrated ECE: "
        f"{results['calibrated_ece']:.6f}"
    )

    print(
        f"Raw Brier score: "
        f"{results['raw_brier_score']:.6f}"
    )

    print(
        f"Calibrated Brier score: "
        f"{results['calibrated_brier_score']:.6f}"
    )

    comparison = pd.DataFrame(
        {
            "true_label": labels,
            "raw_confidence": raw_confidence,
            "calibrated_confidence": calibrated_confidence,
        }
    )

    csv_path = (
        OUTPUT_DIR
        / "confidence_comparison.csv"
    )

    comparison.to_csv(
        csv_path,
        index=False,
    )

    json_path = (
        OUTPUT_DIR
        / "calibration_results.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    create_reliability_diagram(
        results["raw_bins"],
        results["calibrated_bins"],
    )

    print()
    print(
        f"CSV: {csv_path}"
    )

    print(
        f"JSON: {json_path}"
    )

    print()
    print("=" * 70)
    print(
        "CONFIDENCE CALIBRATION FINISHED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()