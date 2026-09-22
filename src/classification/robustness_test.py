"""
AgriVision AI - ResNet18 Robustness Testing

Evaluates the trained ResNet18 under controlled image
degradations using the same PlantVillage test set.

Tests:
1. Original
2. Brightness reduction
3. Brightness increase
4. Contrast reduction
5. Gaussian blur
6. Gaussian noise
7. Rotation
8. JPEG compression
9. Resize degradation

Outputs:
    data/reports/robustness/
        robustness_results.csv
        robustness_results.json
        robustness_comparison.png
"""

from pathlib import Path
import io
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image, ImageEnhance, ImageFilter

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
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
    / "resnet18_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

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

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


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


def collect_image_paths():

    image_paths = []
    labels = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_dir = (
            TEST_DIR / class_name
        )

        if not class_dir.exists():
            continue

        files = sorted(
            [
                p
                for p in class_dir.iterdir()
                if (
                    p.is_file()
                    and p.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]
        )

        for path in files:

            image_paths.append(path)
            labels.append(class_index)

    return image_paths, labels


def apply_perturbation(
    image,
    condition,
):
    """
    Apply a deterministic image degradation.
    """

    if condition == "Original":
        return image

    if condition == "Brightness -30%":
        return ImageEnhance.Brightness(
            image
        ).enhance(0.70)

    if condition == "Brightness +30%":
        return ImageEnhance.Brightness(
            image
        ).enhance(1.30)

    if condition == "Contrast -30%":
        return ImageEnhance.Contrast(
            image
        ).enhance(0.70)

    if condition == "Gaussian Blur":
        return image.filter(
            ImageFilter.GaussianBlur(
                radius=2.0
            )
        )

    if condition == "Gaussian Noise":

        array = np.asarray(
            image
        ).astype(
            np.float32
        )

        rng = np.random.default_rng(
            42
        )

        noise = rng.normal(
            loc=0.0,
            scale=20.0,
            size=array.shape,
        )

        noisy = np.clip(
            array + noise,
            0,
            255,
        ).astype(
            np.uint8
        )

        return Image.fromarray(
            noisy
        )

    if condition == "Rotation +15°":
        return image.rotate(
            15,
            resample=Image.Resampling.BILINEAR,
            expand=False,
        )

    if condition == "JPEG Quality 30":
        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=30,
        )

        buffer.seek(0)

        return Image.open(
            buffer
        ).convert(
            "RGB"
        )

    if condition == "Resize Degradation":

        small = image.resize(
            (56, 56),
            Image.Resampling.BILINEAR,
        )

        return small.resize(
            (256, 256),
            Image.Resampling.BILINEAR,
        )

    raise ValueError(
        f"Unknown condition: {condition}"
    )


class RobustnessDataset(Dataset):

    def __init__(
        self,
        image_paths,
        labels,
        condition,
    ):

        self.image_paths = image_paths
        self.labels = labels
        self.condition = condition

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(
        self,
        index,
    ):

        image_path = self.image_paths[
            index
        ]

        label = self.labels[
            index
        ]

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        image = apply_perturbation(
            image,
            self.condition,
        )

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR,
        )

        tensor = transforms.ToTensor()(
            image
        )

        tensor = normalize(
            tensor
        )

        return tensor, label


def evaluate_condition(
    model,
    image_paths,
    labels,
    condition,
):

    dataset = RobustnessDataset(
        image_paths,
        labels,
        condition,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    predictions = []
    actual = []

    start = time.time()

    processed = 0

    with torch.no_grad():

        for images, batch_labels in loader:

            images = images.to(
                DEVICE
            )

            outputs = model(
                images
            )

            batch_predictions = (
                outputs.argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            predictions.extend(
                batch_predictions
            )

            actual.extend(
                batch_labels.numpy()
            )

            processed += len(
                batch_labels
            )

            if (
                processed % 1000 == 0
                or processed == len(dataset)
            ):
                print(
                    f"    "
                    f"{processed}/"
                    f"{len(dataset)}"
                )

    accuracy = accuracy_score(
        actual,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            actual,
            predictions,
            average="weighted",
            zero_division=0,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            actual,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    elapsed = time.time() - start

    return {
        "condition": condition,
        "images": len(dataset),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "time_seconds": elapsed,
    }


def create_plot(results):

    dataframe = pd.DataFrame(
        results
    )

    plt = __import__(
        "matplotlib.pyplot",
        fromlist=["plt"],
    )

    plt.figure(
        figsize=(13, 7)
    )

    plt.bar(
        dataframe["condition"],
        dataframe["accuracy"] * 100,
    )

    plt.ylabel(
        "Accuracy (%)"
    )

    plt.xlabel(
        "Image Condition"
    )

    plt.title(
        "AgriVision AI - ResNet18 Robustness Test"
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.ylim(
        0,
        100,
    )

    for index, value in enumerate(
        dataframe["accuracy"] * 100
    ):

        plt.text(
            index,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "robustness_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"\nGraph saved:\n{output_path}"
    )


def main():

    print("=" * 70)
    print(
        "AgriVision AI - ResNet18 Robustness Testing"
    )
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Test directory: {TEST_DIR}"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print()
    print("Loading ResNet18...")

    model = load_model()

    print(
        "ResNet18 loaded successfully."
    )

    print()
    print("Collecting test images...")

    image_paths, labels = (
        collect_image_paths()
    )

    print(
        f"Test images: "
        f"{len(image_paths)}"
    )

    if len(image_paths) != 3101:

        print(
            "WARNING: Expected approximately "
            "3101 test images."
        )

    conditions = [
        "Original",
        "Brightness -30%",
        "Brightness +30%",
        "Contrast -30%",
        "Gaussian Blur",
        "Gaussian Noise",
        "Rotation +15°",
        "JPEG Quality 30",
        "Resize Degradation",
    ]

    results = []

    for condition in conditions:

        print()
        print("=" * 70)
        print(
            f"Testing: {condition}"
        )
        print("=" * 70)

        result = evaluate_condition(
            model,
            image_paths,
            labels,
            condition,
        )

        results.append(
            result
        )

        print()
        print(
            f"Accuracy: "
            f"{result['accuracy'] * 100:.2f}%"
        )

        print(
            f"Precision: "
            f"{result['precision'] * 100:.2f}%"
        )

        print(
            f"Recall: "
            f"{result['recall'] * 100:.2f}%"
        )

        print(
            f"F1: "
            f"{result['f1'] * 100:.2f}%"
        )

        print(
            f"Macro F1: "
            f"{result['macro_f1'] * 100:.2f}%"
        )

        print(
            f"Time: "
            f"{result['time_seconds']:.1f}s"
        )

    dataframe = pd.DataFrame(
        results
    )

    csv_path = (
        OUTPUT_DIR
        / "robustness_results.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    json_path = (
        OUTPUT_DIR
        / "robustness_results.json"
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

    create_plot(
        results
    )

    print()
    print("=" * 70)
    print("ROBUSTNESS TEST FINISHED")
    print("=" * 70)

    print()
    print(dataframe[
        [
            "condition",
            "accuracy",
            "f1",
            "macro_f1",
        ]
    ].to_string(
        index=False
    ))

    print()
    print(
        f"CSV: {csv_path}"
    )

    print(
        f"JSON: {json_path}"
    )


if __name__ == "__main__":
    main()