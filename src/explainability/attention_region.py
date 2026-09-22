from pathlib import Path
import random

import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


# ============================================================
# CONFIG
# ============================================================

TEST_DIR = Path("data/processed/classification/test")
CHECKPOINT = Path("models/classification/resnet18_best.pth")

OUTPUT_DIR = Path("data/reports/attention_regions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Grad-CAM activation threshold
CAM_THRESHOLD = 0.50


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD CLASSES
# ============================================================

class_names = sorted([
    folder.name
    for folder in TEST_DIR.iterdir()
    if folder.is_dir()
])

num_classes = len(class_names)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 65)
print("AgriVision AI - Disease Attention Region Analyzer")
print("=" * 65)

print(f"Device: {DEVICE}")
print(f"Classes: {num_classes}")

model = models.resnet18(weights=None)

model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(512, num_classes)
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# GRAD-CAM
# ============================================================

target_layers = [model.layer4[-1]]


# ============================================================
# SELECT ONE IMAGE PER CLASS
# ============================================================

selected_images = []

for class_name in class_names:

    class_dir = TEST_DIR / class_name

    images = []

    for extension in [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    ]:
        images.extend(class_dir.glob(extension))

    if images:
        selected_images.append(
            (class_name, random.choice(images))
        )


# ============================================================
# PROCESS
# ============================================================

results = []

for index, (true_class, image_path) in enumerate(
    selected_images,
    1
):

    print(
        f"\n[{index}/{len(selected_images)}] "
        f"{true_class}"
    )

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    original = Image.open(
        image_path
    ).convert("RGB")

    original_resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    rgb = np.array(
        original_resized
    ).astype(np.float32) / 255.0

    input_tensor = transform(
        original
    ).unsqueeze(0).to(DEVICE)


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = (
            probabilities[
                0,
                predicted_class
            ].item() * 100
        )

    predicted_name = class_names[
        predicted_class
    ]


    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    targets = [
        ClassifierOutputTarget(
            predicted_class
        )
    ]

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]


    # --------------------------------------------------------
    # CREATE PSEUDO-MASK
    # --------------------------------------------------------

    mask = (
        grayscale_cam >= CAM_THRESHOLD
    ).astype(np.uint8) * 255


    # --------------------------------------------------------
    # CLEAN MASK
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # --------------------------------------------------------
    # ESTIMATED ACTIVATION AREA
    # --------------------------------------------------------

    activation_pixels = np.sum(
        mask > 0
    )

    total_pixels = mask.shape[0] * mask.shape[1]

    activation_percentage = (
        activation_pixels /
        total_pixels
    ) * 100


    # --------------------------------------------------------
    # SEVERITY ESTIMATE
    # --------------------------------------------------------

    if predicted_name.lower().endswith("healthy"):

        severity = "Healthy"

    elif activation_percentage < 10:

        severity = "Low"

    elif activation_percentage < 25:

        severity = "Moderate"

    else:

        severity = "High"


    # --------------------------------------------------------
    # SAVE MASK
    # --------------------------------------------------------

    safe_name = (
        true_class
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    mask_path = (
        OUTPUT_DIR /
        f"{index:02d}_{safe_name}_mask.png"
    )

    cv2.imwrite(
        str(mask_path),
        mask
    )


    # --------------------------------------------------------
    # CREATE OVERLAY
    # --------------------------------------------------------

    heatmap = cv2.applyColorMap(
        mask,
        cv2.COLORMAP_JET
    )

    original_bgr = cv2.cvtColor(
        (rgb * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )

    overlay = cv2.addWeighted(
        original_bgr,
        0.65,
        heatmap,
        0.35,
        0
    )

    overlay_path = (
        OUTPUT_DIR /
        f"{index:02d}_{safe_name}_region.png"
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {
        "true_class": true_class,
        "predicted_class": predicted_name,
        "confidence_percent": round(
            confidence,
            2
        ),
        "estimated_attention_area_percent": round(
            activation_percentage,
            2
        ),
        "estimated_severity": severity,
        "image": str(image_path),
        "mask": str(mask_path),
        "overlay": str(overlay_path)
    }

    results.append(result)

    print(
        f"Prediction: {predicted_name}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print(
        f"Attention area: "
        f"{activation_percentage:.2f}%"
    )

    print(
        f"Estimated severity: {severity}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

import json

report_path = (
    OUTPUT_DIR /
    "attention_region_results.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\n" + "=" * 65)
print("ATTENTION REGION ANALYSIS COMPLETE")
print("=" * 65)

print(
    f"Results: {OUTPUT_DIR}"
)

print(
    f"Report: {report_path}"
)

print(
    "\nNOTE:"
)

print(
    "The estimated area is Grad-CAM attention area, "
    "not ground-truth disease segmentation."
)
