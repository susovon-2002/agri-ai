"""
AgriVision AI - Out-of-Distribution (OOD) Detector

Uses ResNet18 feature embeddings and class-centroid distance
to identify whether an image is similar to the PlantVillage
training distribution.

This module does NOT replace the disease classifier.
It provides an additional supported/unknown-image signal.
"""

from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = PROJECT_ROOT / "data" / "processed" / "classification" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "classification" / "resnet18_best.pth"

OUTPUT_DIR = PROJECT_ROOT / "data" / "reports" / "ood"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    """Load the trained ResNet18 checkpoint."""

    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(CLASS_NAMES)),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def get_feature_extractor(model):
    """
    Remove the final classifier and return the 512-dimensional
    ResNet18 feature extractor.
    """

    feature_extractor = nn.Sequential(
        *list(model.children())[:-1]
    )

    feature_extractor.to(DEVICE)
    feature_extractor.eval()

    return feature_extractor


@torch.no_grad()
def extract_feature(feature_extractor, image_path):
    """Extract normalized ResNet18 embedding."""

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    feature = feature_extractor(tensor)

    feature = torch.flatten(feature, 1)

    feature = torch.nn.functional.normalize(
        feature,
        p=2,
        dim=1,
    )

    return feature.squeeze(0).cpu().numpy()


def collect_features(feature_extractor):
    """
    Extract embeddings from the PlantVillage test set.

    Returns:
        features
        labels
    """

    features = []
    labels = []

    print()
    print("Extracting PlantVillage test-set embeddings...")

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_dir = TEST_DIR / class_name

        if not class_dir.exists():
            print(f"WARNING: Missing class directory: {class_name}")
            continue

        image_files = [
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        print(
            f"{class_name}: {len(image_files)} images"
        )

        for image_path in image_files:

            try:
                feature = extract_feature(
                    feature_extractor,
                    image_path,
                )

                features.append(feature)
                labels.append(class_index)

            except Exception as exc:
                print(
                    f"Skipping {image_path.name}: {exc}"
                )

    return np.asarray(features), np.asarray(labels)


def calculate_class_centroids(features, labels):
    """Calculate normalized centroid for every disease class."""

    centroids = {}

    for class_index, class_name in enumerate(CLASS_NAMES):

        class_features = features[
            labels == class_index
        ]

        if len(class_features) == 0:
            continue

        centroid = np.mean(
            class_features,
            axis=0,
        )

        norm = np.linalg.norm(centroid)

        if norm > 0:
            centroid = centroid / norm

        centroids[class_index] = centroid

    return centroids


def cosine_distance(a, b):
    """Cosine distance between two vectors."""

    similarity = np.dot(a, b)

    return float(1.0 - similarity)


def calculate_training_distances(
    features,
    labels,
    centroids,
):
    """
    Calculate each training image's distance from its
    own class centroid.

    These distances form the reference distribution.
    """

    distances = []

    for feature, label in zip(features, labels):

        if int(label) not in centroids:
            continue

        distance = cosine_distance(
            feature,
            centroids[int(label)],
        )

        distances.append(distance)

    return np.asarray(distances)


def determine_threshold(distances):
    """
    Determine an OOD threshold from the reference distribution.

    Uses the 99th percentile so that only unusually distant
    samples are considered suspicious.
    """

    threshold = float(
        np.percentile(
            distances,
            99,
        )
    )

    return threshold


@torch.no_grad()
def classify_with_ood(
    model,
    feature_extractor,
    centroids,
    threshold,
    image_path,
):
    """Run classification + OOD analysis on one image."""

    image = Image.open(image_path).convert("RGB")

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    logits = model(tensor)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    confidence, predicted_index = torch.max(
        probabilities,
        dim=1,
    )

    feature = feature_extractor(tensor)

    feature = torch.flatten(
        feature,
        1,
    )

    feature = torch.nn.functional.normalize(
        feature,
        p=2,
        dim=1,
    )

    feature = feature.squeeze(0).cpu().numpy()

    predicted_index = int(
        predicted_index.item()
    )

    confidence = float(
        confidence.item()
    )

    distances = {
        class_index: cosine_distance(
            feature,
            centroid,
        )
        for class_index, centroid in centroids.items()
    }

    nearest_class = min(
        distances,
        key=distances.get,
    )

    nearest_distance = distances[
        nearest_class
    ]

    predicted_distance = distances.get(
        predicted_index,
        nearest_distance,
    )

    is_ood = predicted_distance > threshold

    return {
        "image": str(image_path),
        "prediction": CLASS_NAMES[predicted_index],
        "prediction_index": predicted_index,
        "classification_confidence": round(
            confidence * 100,
            4,
        ),
        "predicted_class_distance": round(
            predicted_distance,
            6,
        ),
        "nearest_class": CLASS_NAMES[nearest_class],
        "nearest_class_distance": round(
            nearest_distance,
            6,
        ),
        "ood_threshold": round(
            threshold,
            6,
        ),
        "ood_status": (
            "UNKNOWN_OR_UNSUPPORTED"
            if is_ood
            else "SUPPORTED"
        ),
    }


def main():

    print("=" * 70)
    print("AgriVision AI - OOD Detection")
    print("=" * 70)

    print(f"Device: {DEVICE}")
    print(f"Test directory: {TEST_DIR}")
    print(f"Model: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Test directory not found: {TEST_DIR}"
        )

    print()
    print("Loading ResNet18...")

    model = load_model()

    print("ResNet18 loaded successfully.")

    feature_extractor = get_feature_extractor(
        model
    )

    print()
    print("Extracting features...")

    features, labels = collect_features(
        feature_extractor
    )

    print()
    print(f"Total embeddings: {len(features)}")

    if len(features) == 0:
        raise RuntimeError(
            "No embeddings were extracted."
        )

    print()
    print("Calculating class centroids...")

    centroids = calculate_class_centroids(
        features,
        labels,
    )

    print(
        f"Class centroids calculated: "
        f"{len(centroids)}"
    )

    print()
    print("Calculating reference distances...")

    training_distances = calculate_training_distances(
        features,
        labels,
        centroids,
    )

    threshold = determine_threshold(
        training_distances
    )

    print()
    print(
        f"OOD threshold: {threshold:.6f}"
    )

    threshold_info = {
        "method": "99th percentile of class-centroid cosine distances",
        "threshold": threshold,
        "reference_samples": int(
            len(training_distances)
        ),
        "classes": len(centroids),
    }

    threshold_path = (
        OUTPUT_DIR / "ood_threshold.json"
    )

    with open(
        threshold_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            threshold_info,
            file,
            indent=2,
        )

    print()
    print(
        f"Saved threshold: {threshold_path}"
    )

    # Test several PlantVillage images
    print()
    print("=" * 70)
    print("Testing OOD detector")
    print("=" * 70)

    sample_results = []

    for class_name in CLASS_NAMES:

        class_dir = TEST_DIR / class_name

        if not class_dir.exists():
            continue

        image_files = [
            p for p in class_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not image_files:
            continue

        image_path = image_files[0]

        result = classify_with_ood(
            model,
            feature_extractor,
            centroids,
            threshold,
            image_path,
        )

        sample_results.append(result)

        print()
        print(
            f"Image: {image_path.name}"
        )
        print(
            f"Prediction: {result['prediction']}"
        )
        print(
            f"Confidence: "
            f"{result['classification_confidence']:.2f}%"
        )
        print(
            f"Distance: "
            f"{result['predicted_class_distance']:.6f}"
        )
        print(
            f"OOD Status: "
            f"{result['ood_status']}"
        )

    results_path = (
        OUTPUT_DIR / "ood_sample_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            sample_results,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("OOD DETECTION TEST FINISHED")
    print("=" * 70)

    print(
        f"Threshold: {threshold:.6f}"
    )

    print(
        f"Results saved to: {results_path}"
    )


if __name__ == "__main__":
    main()