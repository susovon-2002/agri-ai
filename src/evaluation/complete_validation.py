from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm


# ============================================================
# AgriVision AI - Complete Validation Engine
# ============================================================

TEST_DIR = Path(
    "data/processed/classification/test"
)

CHECKPOINT = Path(
    "models/classification/resnet18_best.pth"
)

OUTPUT_DIR = Path(
    "data/reports/final_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("AgriVision AI - Complete Model Validation")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Test directory: {TEST_DIR}")
print(f"Checkpoint: {CHECKPOINT}")


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASET
# ============================================================

dataset = ImageFolder(
    TEST_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)

class_names = dataset.classes
num_classes = len(class_names)

print(f"Classes: {num_classes}")
print(f"Test images: {len(dataset)}")


# ============================================================
# MODEL
# ============================================================

model = models.resnet18(
    weights=None
)

model.fc = torch.nn.Sequential(

    torch.nn.Dropout(0.3),

    torch.nn.Linear(
        512,
        num_classes
    )
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model = model.to(
    DEVICE
)

model.eval()

print("ResNet18 loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

all_targets = []
all_predictions = []
all_confidences = []

start_time = time.time()

print()
print("Running complete test-set evaluation...")
print()

with torch.no_grad():

    progress = tqdm(
        loader,
        desc="Validation",
        unit="batch"
    )

    for images, targets in progress:

        images = images.to(
            DEVICE
        )

        targets = targets.to(
            DEVICE
        )

        outputs = model(
            images
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidences, predictions = torch.max(
            probabilities,
            dim=1
        )

        all_targets.extend(
            targets.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_confidences.extend(
            confidences.cpu().numpy()
        )


elapsed = time.time() - start_time

y_true = np.array(
    all_targets
)

y_pred = np.array(
    all_predictions
)

confidences = (
    np.array(
        all_confidences
    ) * 100
)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_df.to_csv(
    OUTPUT_DIR /
    "classification_report.csv"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(16, 14)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "AgriVision AI - ResNet18 Confusion Matrix",
    fontsize=18
)

plt.colorbar()

tick_marks = np.arange(
    num_classes
)

plt.xticks(
    tick_marks,
    class_names,
    rotation=90,
    fontsize=8
)

plt.yticks(
    tick_marks,
    class_names,
    fontsize=8
)

threshold = cm.max() / 2.0

for i in range(
    num_classes
):

    for j in range(
        num_classes
    ):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )

plt.ylabel(
    "Actual Class"
)

plt.xlabel(
    "Predicted Class"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "confusion_matrix.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

per_class_accuracy = []

for class_id, class_name in enumerate(
    class_names
):

    mask = (
        y_true == class_id
    )

    total = np.sum(
        mask
    )

    correct = np.sum(
        y_pred[mask] == class_id
    )

    class_accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    per_class_accuracy.append({

        "class":
            class_name,

        "samples":
            int(total),

        "correct":
            int(correct),

        "accuracy_percent":
            round(
                class_accuracy,
                4
            )
    })


per_class_df = pd.DataFrame(
    per_class_accuracy
)

per_class_df.to_csv(
    OUTPUT_DIR /
    "per_class_accuracy.csv",
    index=False
)


# ============================================================
# PER-CLASS ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(16, 9)
)

plt.bar(
    per_class_df["class"],
    per_class_df[
        "accuracy_percent"
    ]
)

plt.title(
    "AgriVision AI - Per-Class Accuracy",
    fontsize=18
)

plt.ylabel(
    "Accuracy (%)"
)

plt.xlabel(
    "Disease / Class"
)

plt.ylim(
    90,
    100.5
)

plt.xticks(
    rotation=70,
    ha="right"
)

for i, value in enumerate(
    per_class_df[
        "accuracy_percent"
    ]
):

    plt.text(
        i,
        value + 0.1,
        f"{value:.2f}%",
        ha="center",
        fontsize=8
    )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "per_class_accuracy.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CONFIDENCE DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.hist(
    confidences,
    bins=20
)

plt.axvline(
    np.mean(confidences),
    linestyle="--",
    label=(
        f"Mean = "
        f"{np.mean(confidences):.2f}%"
    )
)

plt.title(
    "AgriVision AI - Prediction Confidence Distribution",
    fontsize=18
)

plt.xlabel(
    "Prediction Confidence (%)"
)

plt.ylabel(
    "Number of Images"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "confidence_distribution.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CONFIDENCE BY CLASS
# ============================================================

confidence_by_class = []

for class_id, class_name in enumerate(
    class_names
):

    mask = (
        y_pred == class_id
    )

    values = confidences[
        mask
    ]

    confidence_by_class.append({

        "class":
            class_name,

        "mean_confidence_percent":
            round(
                float(
                    np.mean(values)
                ) if len(values)
                else 0,
                4
            ),

        "minimum_confidence_percent":
            round(
                float(
                    np.min(values)
                ) if len(values)
                else 0,
                4
            ),

        "maximum_confidence_percent":
            round(
                float(
                    np.max(values)
                ) if len(values)
                else 0,
                4
            )
    })


confidence_df = pd.DataFrame(
    confidence_by_class
)

confidence_df.to_csv(
    OUTPUT_DIR /
    "confidence_by_class.csv",
    index=False
)


# ============================================================
# SEVERITY RESULTS
# ============================================================

severity_file = Path(
    "data/reports/severity/severity_results.csv"
)

severity_summary = {}

if severity_file.exists():

    try:

        severity_df = pd.read_csv(
            severity_file
        )

        possible_columns = [
            "severity",
            "Severity",
            "severity_level",
            "Severity Level"
        ]

        severity_column = None

        for column in possible_columns:

            if column in severity_df.columns:

                severity_column = column
                break

        if severity_column is not None:

            counts = (
                severity_df[
                    severity_column
                ]
                .value_counts()
                .to_dict()
            )

            severity_summary = {
                str(k): int(v)
                for k, v in counts.items()
            }

            plt.figure(
                figsize=(10, 7)
            )

            labels = list(
                severity_summary.keys()
            )

            values = list(
                severity_summary.values()
            )

            bars = plt.bar(
                labels,
                values
            )

            plt.title(
                "AgriVision AI - Severity Distribution",
                fontsize=18
            )

            plt.xlabel(
                "Severity Level"
            )

            plt.ylabel(
                "Number of Images"
            )

            for bar, value in zip(
                bars,
                values
            ):

                plt.text(
                    bar.get_x()
                    + bar.get_width() / 2,
                    value,
                    str(value),
                    ha="center",
                    va="bottom"
                )

            plt.tight_layout()

            plt.savefig(
                OUTPUT_DIR /
                "severity_distribution.png",
                dpi=200,
                bbox_inches="tight"
            )

            plt.close()

    except Exception as error:

        severity_summary = {
            "error":
                str(error)
        }


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_records = []

for index in range(
    len(y_true)
):

    true_id = int(
        y_true[index]
    )

    pred_id = int(
        y_pred[index]
    )

    prediction_records.append({

        "image":
            str(
                dataset.samples[index][0]
            ),

        "actual_class":
            class_names[true_id],

        "predicted_class":
            class_names[pred_id],

        "correct":
            bool(
                true_id == pred_id
            ),

        "confidence_percent":
            round(
                float(
                    confidences[index]
                ),
                4
            )
    })


prediction_df = pd.DataFrame(
    prediction_records
)

prediction_df.to_csv(
    OUTPUT_DIR /
    "test_predictions.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

summary = {

    "project":
        "AgriVision AI",

    "model":
        "ResNet18",

    "test_samples":
        int(len(dataset)),

    "classes":
        int(num_classes),

    "accuracy_percent":
        round(
            accuracy * 100,
            4
        ),

    "weighted_precision_percent":
        round(
            precision * 100,
            4
        ),

    "weighted_recall_percent":
        round(
            recall * 100,
            4
        ),

    "weighted_f1_percent":
        round(
            f1 * 100,
            4
        ),

    "macro_precision_percent":
        round(
            macro_precision * 100,
            4
        ),

    "macro_recall_percent":
        round(
            macro_recall * 100,
            4
        ),

    "macro_f1_percent":
        round(
            macro_f1 * 100,
            4
        ),

    "mean_confidence_percent":
        round(
            float(
                np.mean(confidences)
            ),
            4
        ),

    "median_confidence_percent":
        round(
            float(
                np.median(confidences)
            ),
            4
        ),

    "minimum_confidence_percent":
        round(
            float(
                np.min(confidences)
            ),
            4
        ),

    "maximum_confidence_percent":
        round(
            float(
                np.max(confidences)
            ),
            4
        ),

    "evaluation_time_seconds":
        round(
            elapsed,
            2
        ),

    "severity_distribution":
        severity_summary
}


with open(
    OUTPUT_DIR /
    "validation_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        summary,
        file,
        indent=4
    )


# ============================================================
# TEXT REPORT
# ============================================================

report_lines = [

    "=" * 70,

    "AGRIVISION AI - FINAL VALIDATION REPORT",

    "=" * 70,

    "",

    "MODEL",

    "ResNet18",

    "",

    "TEST DATA",

    f"Test samples: {len(dataset)}",

    f"Classes: {num_classes}",

    "",

    "OVERALL PERFORMANCE",

    f"Accuracy          : {accuracy * 100:.4f}%",

    f"Precision         : {precision * 100:.4f}%",

    f"Recall            : {recall * 100:.4f}%",

    f"F1 Score          : {f1 * 100:.4f}%",

    "",

    "MACRO PERFORMANCE",

    f"Macro Precision   : {macro_precision * 100:.4f}%",

    f"Macro Recall      : {macro_recall * 100:.4f}%",

    f"Macro F1          : {macro_f1 * 100:.4f}%",

    "",

    "CONFIDENCE",

    f"Mean              : {np.mean(confidences):.4f}%",

    f"Median            : {np.median(confidences):.4f}%",

    f"Minimum           : {np.min(confidences):.4f}%",

    f"Maximum           : {np.max(confidences):.4f}%",

    "",

    "EVALUATION TIME",

    f"{elapsed:.2f} seconds",

    "",

    "OUTPUT FILES",

    "classification_report.csv",

    "per_class_accuracy.csv",

    "confidence_by_class.csv",

    "test_predictions.csv",

    "confusion_matrix.png",

    "per_class_accuracy.png",

    "confidence_distribution.png",

    "severity_distribution.png",

    "validation_summary.json",

    "FINAL_VALIDATION_REPORT.txt",

    "",

    "=" * 70
]


with open(
    OUTPUT_DIR /
    "FINAL_VALIDATION_REPORT.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(
            report_lines
        )
    )


# ============================================================
# TERMINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("COMPLETE VALIDATION FINISHED")
print("=" * 70)

print()
print(
    f"Accuracy  : {accuracy * 100:.4f}%"
)

print(
    f"Precision : {precision * 100:.4f}%"
)

print(
    f"Recall    : {recall * 100:.4f}%"
)

print(
    f"F1 Score  : {f1 * 100:.4f}%"
)

print(
    f"Macro F1  : {macro_f1 * 100:.4f}%"
)

print(
    f"Mean Conf.: {np.mean(confidences):.4f}%"
)

print()
print(
    f"Results saved to:"
)

print(
    OUTPUT_DIR
)

print()
print("=" * 70)
