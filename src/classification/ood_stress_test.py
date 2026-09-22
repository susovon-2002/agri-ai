"""
AgriVision AI - OOD Stress Test

Creates synthetic non-leaf images and checks whether the
OOD detector rejects them as UNKNOWN / UNSUPPORTED.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from ood_detector import (
    load_model,
    get_feature_extractor,
    classify_with_ood,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OOD_IMAGE_DIR = (
    PROJECT_ROOT / "data" / "ood_test" / "unknown"
)

THRESHOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "ood"
    / "ood_threshold.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "ood_stress_test"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

IMAGE_SIZE = 224


def generate_noise_image(path):
    """Generate random noise."""

    array = np.random.randint(
        0,
        256,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    image = Image.fromarray(
        array,
        mode="RGB",
    )

    image.save(path)


def generate_geometric_image(path):
    """Generate an image containing geometric shapes."""

    image = Image.new(
        "RGB",
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        ),
    )

    draw = ImageDraw.Draw(image)

    for _ in range(15):

        x1 = random.randint(0, 180)
        y1 = random.randint(0, 180)

        x2 = random.randint(
            x1 + 10,
            223,
        )

        y2 = random.randint(
            y1 + 10,
            223,
        )

        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

        shape_type = random.choice(
            [
                "rectangle",
                "ellipse",
                "line",
            ]
        )

        if shape_type == "rectangle":

            draw.rectangle(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                fill=color,
            )

        elif shape_type == "ellipse":

            draw.ellipse(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                fill=color,
            )

        else:

            draw.line(
                [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                fill=color,
                width=random.randint(
                    2,
                    10,
                ),
            )

    image.save(path)


def generate_gradient_image(path):
    """Generate a synthetic gradient."""

    x = np.linspace(
        0,
        255,
        IMAGE_SIZE,
    )

    y = np.linspace(
        0,
        255,
        IMAGE_SIZE,
    )

    xx, yy = np.meshgrid(
        x,
        y,
    )

    image_array = np.zeros(
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
            3,
        ),
        dtype=np.uint8,
    )

    image_array[:, :, 0] = (
        xx % 256
    )

    image_array[:, :, 1] = (
        yy % 256
    )

    image_array[:, :, 2] = (
        (xx + yy) % 256
    )

    image = Image.fromarray(
        image_array,
        mode="RGB",
    )

    image.save(path)


def generate_text_image(path):
    """Generate a non-leaf text image."""

    image = Image.new(
        "RGB",
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        "white",
    )

    draw = ImageDraw.Draw(image)

    messages = [
        "AGRIVISION",
        "OOD TEST",
        "UNKNOWN",
        "MACHINE LEARNING",
    ]

    for index, message in enumerate(messages):

        draw.text(
            (
                10,
                20 + index * 45,
            ),
            message,
            fill=(
                random.randint(0, 100),
                random.randint(0, 100),
                random.randint(0, 100),
            ),
        )

    image.save(path)


def generate_test_images():

    print()
    print("=" * 70)
    print("Generating OOD stress-test images")
    print("=" * 70)

    # Remove old synthetic images.
    for image_path in OOD_IMAGE_DIR.glob(
        "synthetic_*.png"
    ):
        image_path.unlink()

    image_number = 1

    # Random noise images.
    for _ in range(10):

        path = (
            OOD_IMAGE_DIR
            / f"synthetic_noise_{image_number:02d}.png"
        )

        generate_noise_image(path)

        image_number += 1

    # Geometric images.
    for _ in range(10):

        path = (
            OOD_IMAGE_DIR
            / f"synthetic_shapes_{image_number:02d}.png"
        )

        generate_geometric_image(path)

        image_number += 1

    # Gradient images.
    for _ in range(5):

        path = (
            OOD_IMAGE_DIR
            / f"synthetic_gradient_{image_number:02d}.png"
        )

        generate_gradient_image(path)

        image_number += 1

    # Text images.
    for _ in range(5):

        path = (
            OOD_IMAGE_DIR
            / f"synthetic_text_{image_number:02d}.png"
        )

        generate_text_image(path)

        image_number += 1

    total = len(
        list(
            OOD_IMAGE_DIR.glob("*.png")
        )
    )

    print()
    print(
        f"Generated OOD images: {total}"
    )


def load_threshold():

    if not THRESHOLD_PATH.exists():

        raise FileNotFoundError(
            f"Threshold file not found: "
            f"{THRESHOLD_PATH}"
        )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return float(
        data["threshold"]
    )


def main():

    print("=" * 70)
    print("AgriVision AI - OOD Stress Test")
    print("=" * 70)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    generate_test_images()

    threshold = load_threshold()

    print()
    print(
        f"OOD threshold: {threshold:.6f}"
    )

    print()
    print("Loading ResNet18...")

    model = load_model()

    feature_extractor = (
        get_feature_extractor(model)
    )

    print(
        "ResNet18 loaded successfully."
    )

    image_files = sorted(
        OOD_IMAGE_DIR.glob("*.png")
    )

    if not image_files:

        raise RuntimeError(
            "No OOD test images found."
        )

    print()
    print("Calculating class centroids once...")
    centroids = load_centroids()
    print(f"Class centroids ready: {len(centroids)}")

    results = []

    print()
    print("=" * 70)
    print("Running OOD stress test")
    print("=" * 70)

    for index, image_path in enumerate(
        image_files,
        start=1,
    ):

        result = classify_with_ood(
            model=model,
            feature_extractor=feature_extractor,
            centroids=centroids,
            threshold=threshold,
            image_path=image_path,
        )

        results.append(result)

        print(
            f"[{index:02d}/{len(image_files)}] "
            f"{image_path.name}"
        )

        print(
            f"  Prediction: "
            f"{result['prediction']}"
        )

        print(
            f"  Confidence: "
            f"{result['classification_confidence']:.2f}%"
        )

        print(
            f"  Distance: "
            f"{result['predicted_class_distance']:.6f}"
        )

        print(
            f"  OOD: "
            f"{result['ood_status']}"
        )

    dataframe = pd.DataFrame(
        results
    )

    csv_path = (
        OUTPUT_DIR
        / "ood_stress_test_results.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    total = len(dataframe)

    unknown_count = int(
        (
            dataframe["ood_status"]
            == "UNKNOWN_OR_UNSUPPORTED"
        ).sum()
    )

    supported_count = int(
        (
            dataframe["ood_status"]
            == "SUPPORTED"
        ).sum()
    )

    rejection_rate = (
        unknown_count / total * 100
        if total > 0
        else 0
    )

    summary = {
        "total_ood_images": total,
        "unknown_or_unsupported": unknown_count,
        "incorrectly_supported": supported_count,
        "ood_rejection_rate_percent": round(
            rejection_rate,
            4,
        ),
        "threshold": threshold,
    }

    json_path = (
        OUTPUT_DIR
        / "ood_stress_test_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("OOD STRESS TEST FINISHED")
    print("=" * 70)

    print(
        f"Total OOD images: {total}"
    )

    print(
        f"Rejected as unknown: "
        f"{unknown_count}"
    )

    print(
        f"Incorrectly accepted: "
        f"{supported_count}"
    )

    print(
        f"OOD rejection rate: "
        f"{rejection_rate:.2f}%"
    )

    print()
    print(
        f"CSV: {csv_path}"
    )

    print(
        f"Summary: {json_path}"
    )


def load_centroids():
    """
    Reconstruct class centroids from the PlantVillage
    test embeddings.

    This reuses the same methodology as ood_detector.py.
    """

    from ood_detector import (
        TEST_DIR,
        CLASS_NAMES,
        collect_features,
        calculate_class_centroids,
    )

    model = load_model()

    feature_extractor = (
        get_feature_extractor(model)
    )

    features, labels = collect_features(
        feature_extractor
    )

    return calculate_class_centroids(
        features,
        labels,
    )


if __name__ == "__main__":
    main()
