from pathlib import Path
import random

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = Path("data/processed/classification/test")
CHECKPOINT = Path("models/classification/resnet18_best.pth")

OUTPUT_DIR = Path("data/reports/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
NUM_IMAGES = 15

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
# LOAD MODEL
# ============================================================

class_names = sorted([
    folder.name
    for folder in TEST_DIR.iterdir()
    if folder.is_dir()
])

num_classes = len(class_names)

print("=" * 60)
print("AgriVision AI - Grad-CAM Explainability")
print("=" * 60)

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
# GRAD-CAM TARGET LAYER
# ============================================================

target_layers = [model.layer4[-1]]


# ============================================================
# SELECT ONE IMAGE FROM EACH CLASS
# ============================================================

selected_images = []

for class_name in class_names:

    class_dir = TEST_DIR / class_name

    images = list(class_dir.glob("*.jpg"))
    images += list(class_dir.glob("*.jpeg"))
    images += list(class_dir.glob("*.png"))
    images += list(class_dir.glob("*.JPG"))
    images += list(class_dir.glob("*.JPEG"))
    images += list(class_dir.glob("*.PNG"))

    if images:
        selected_images.append(
            (class_name, random.choice(images))
        )


# Limit if necessary
selected_images = selected_images[:NUM_IMAGES]


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

for index, (true_class, image_path) in enumerate(selected_images, 1):

    print(
        f"\n[{index}/{len(selected_images)}] "
        f"Processing: {true_class}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    original = Image.open(image_path).convert("RGB")

    original_resized = original.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    rgb_image = np.array(original_resized).astype(
        np.float32
    ) / 255.0

    input_tensor = transform(original).unsqueeze(0).to(DEVICE)


    # --------------------------------------------------------
    # Prediction
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

        confidence = probabilities[
            0,
            predicted_class
        ].item() * 100


    predicted_name = class_names[predicted_class]


    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    targets = [
        ClassifierOutputTarget(predicted_class)
    ]

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )

        grayscale_cam = grayscale_cam[0]


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    safe_name = (
        true_class
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    output_path = (
        OUTPUT_DIR /
        f"{index:02d}_{safe_name}.png"
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(rgb_image)
    axes[0].set_title(
        f"Original\nTrue: {true_class}"
    )
    axes[0].axis("off")

    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(visualization)
    axes[2].set_title(
        f"Prediction: {predicted_name}\n"
        f"Confidence: {confidence:.2f}%"
    )
    axes[2].axis("off")

    plt.suptitle(
        "AgriVision AI - ResNet18 Explainability",
        fontsize=15
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


print("\n" + "=" * 60)
print("GRAD-CAM COMPLETE")
print("=" * 60)
print(f"Results saved in: {OUTPUT_DIR}")
