from pathlib import Path
import json
import random

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision import transforms, models
from src.gradcam_compat import GradCAM, ClassifierOutputTarget


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = Path("data/processed/classification/test")
CHECKPOINT = Path("models/classification/resnet18_best.pth")

OUTPUT_DIR = Path("data/reports/disease_region")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
CAM_THRESHOLD = 0.50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Fixed seed so the experiment is reproducible
random.seed(42)


# ============================================================
# IMAGE TRANSFORM
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
    p.name
    for p in TEST_DIR.iterdir()
    if p.is_dir()
])

num_classes = len(class_names)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("AgriVision AI - Grad-CAM + Leaf Mask Fusion")
print("=" * 70)

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
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    model.load_state_dict(checkpoint)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# GRAD-CAM TARGET
# ============================================================

target_layers = [model.layer4[-1]]


# ============================================================
# LEAF SEGMENTATION
# ============================================================

def segment_leaf(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower_green = np.array(
        [20, 25, 20],
        dtype=np.uint8
    )

    upper_green = np.array(
        [100, 255, 255],
        dtype=np.uint8
    )

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    saturation_mask = cv2.inRange(
        hsv,
        np.array(
            [15, 20, 20],
            dtype=np.uint8
        ),
        np.array(
            [110, 255, 255],
            dtype=np.uint8
        )
    )

    mask = cv2.bitwise_or(
        green_mask,
        saturation_mask
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (7, 7)
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

    # Connected components
    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    cleaned = np.zeros_like(mask)

    image_area = IMAGE_SIZE * IMAGE_SIZE

    for label in range(1, num_labels):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area > image_area * 0.01:

            cleaned[
                labels == label
            ] = 255

    # Fill contours
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final_mask = np.zeros_like(cleaned)

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > image_area * 0.01:

            cv2.drawContours(
                final_mask,
                [contour],
                -1,
                255,
                thickness=cv2.FILLED
            )

    return final_mask


# ============================================================
# GET ONE IMAGE PER CLASS
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
        images.extend(
            class_dir.glob(extension)
        )

    if images:

        image_path = random.choice(
            images
        )

        selected_images.append(
            (class_name, image_path)
        )


# ============================================================
# PROCESS EACH CLASS
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
    # LOAD IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print("ERROR: Could not read image.")
        continue

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    rgb_float = (
        rgb_image.astype(
            np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    pil_image = transforms.ToPILImage()(
        rgb_image
    )

    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(DEVICE)


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            input_tensor
        )

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
    # LEAF MASK
    # --------------------------------------------------------

    leaf_mask = segment_leaf(
        image
    )

    leaf_binary = (
        leaf_mask > 0
    )


    # --------------------------------------------------------
    # GRAD-CAM THRESHOLD
    # --------------------------------------------------------

    cam_binary = (
        grayscale_cam >= CAM_THRESHOLD
    )


    # --------------------------------------------------------
    # FUSION
    #
    # Disease-region candidate =
    # Grad-CAM activated pixels INSIDE leaf mask
    # --------------------------------------------------------

    fused_mask = (
        leaf_binary &
        cam_binary
    ).astype(np.uint8) * 255


    # --------------------------------------------------------
    # REMOVE SMALL NOISE
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    fused_mask = cv2.morphologyEx(
        fused_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    fused_mask = cv2.morphologyEx(
        fused_mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # --------------------------------------------------------
    # AREA CALCULATION
    # --------------------------------------------------------

    leaf_pixels = np.sum(
        leaf_binary
    )

    fused_pixels = np.sum(
        fused_mask > 0
    )

    total_pixels = (
        IMAGE_SIZE *
        IMAGE_SIZE
    )

    leaf_area_percent = (
        leaf_pixels /
        total_pixels
    ) * 100

    attention_area_percent = (
        fused_pixels /
        total_pixels
    ) * 100

    # Most useful ratio:
    # activated area / detected leaf area

    if leaf_pixels > 0:

        affected_leaf_ratio = (
            fused_pixels /
            leaf_pixels
        ) * 100

    else:

        affected_leaf_ratio = 0.0


    # --------------------------------------------------------
    # HEALTHY CLASS HANDLING
    # --------------------------------------------------------

    is_healthy = (
        "healthy" in predicted_name.lower()
    )


    # --------------------------------------------------------
    # SEVERITY BASELINE
    #
    # This is ONLY a research baseline.
    # --------------------------------------------------------

    if is_healthy:

        severity = "Healthy"

    elif affected_leaf_ratio < 10:

        severity = "Low"

    elif affected_leaf_ratio < 25:

        severity = "Moderate"

    else:

        severity = "High"


    # --------------------------------------------------------
    # CREATE COLORED FUSION
    # --------------------------------------------------------

    heatmap = cv2.applyColorMap(
        fused_mask,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        image,
        0.65,
        heatmap,
        0.35,
        0
    )


    # --------------------------------------------------------
    # SAVE MASKS
    # --------------------------------------------------------

    safe_name = (
        true_class
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    leaf_mask_path = (
        OUTPUT_DIR /
        f"{index:02d}_{safe_name}_leaf_mask.png"
    )

    fused_mask_path = (
        OUTPUT_DIR /
        f"{index:02d}_{safe_name}_disease_region.png"
    )

    overlay_path = (
        OUTPUT_DIR /
        f"{index:02d}_{safe_name}_fusion.png"
    )

    cv2.imwrite(
        str(leaf_mask_path),
        leaf_mask
    )

    cv2.imwrite(
        str(fused_mask_path),
        fused_mask
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )


    # --------------------------------------------------------
    # FOUR-PANEL VISUALIZATION
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(18, 5)
    )

    axes[0].imshow(
        rgb_image
    )

    axes[0].set_title(
        f"Original\nTrue: {true_class}"
    )

    axes[0].axis("off")


    axes[1].imshow(
        leaf_mask,
        cmap="gray"
    )

    axes[1].set_title(
        f"Leaf Mask\n"
        f"{leaf_area_percent:.1f}% image"
    )

    axes[1].axis("off")


    axes[2].imshow(
        fused_mask,
        cmap="gray"
    )

    axes[2].set_title(
        "Fused Disease Region"
    )

    axes[2].axis("off")


    axes[3].imshow(
        cv2.cvtColor(
            overlay,
            cv2.COLOR_BGR2RGB
        )
    )

    axes[3].set_title(
        f"{predicted_name}\n"
        f"Confidence: {confidence:.2f}%\n"
        f"Leaf ratio: {affected_leaf_ratio:.2f}%"
    )

    axes[3].axis("off")


    plt.suptitle(
        "AgriVision AI - Disease Region Fusion",
        fontsize=15
    )

    plt.tight_layout()


    comparison_path = (
        OUTPUT_DIR /
        f"{index:02d}_{safe_name}_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    result = {

        "true_class": true_class,

        "predicted_class": predicted_name,

        "correct_prediction": (
            true_class == predicted_name
        ),

        "confidence_percent": round(
            confidence,
            2
        ),

        "leaf_area_percent_of_image": round(
            leaf_area_percent,
            2
        ),

        "gradcam_threshold": CAM_THRESHOLD,

        "gradcam_attention_area_percent_of_image": round(
            attention_area_percent,
            2
        ),

        "estimated_affected_leaf_ratio_percent": round(
            affected_leaf_ratio,
            2
        ),

        "baseline_severity": severity,

        "image": str(image_path),

        "leaf_mask": str(leaf_mask_path),

        "disease_region_mask": str(
            fused_mask_path
        ),

        "fusion_visualization": str(
            comparison_path
        )
    }

    results.append(result)


    # --------------------------------------------------------
    # TERMINAL OUTPUT
    # --------------------------------------------------------

    print(
        f"Prediction: {predicted_name}"
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print(
        f"Leaf area: "
        f"{leaf_area_percent:.2f}% of image"
    )

    print(
        f"Fused attention: "
        f"{attention_area_percent:.2f}% of image"
    )

    print(
        f"Attention inside leaf: "
        f"{affected_leaf_ratio:.2f}%"
    )

    print(
        f"Baseline severity: "
        f"{severity}"
    )


# ============================================================
# SAVE JSON REPORT
# ============================================================

report_path = (
    OUTPUT_DIR /
    "disease_region_fusion_results.json"
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


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("GRAD-CAM + LEAF MASK FUSION COMPLETE")
print("=" * 70)

print(
    f"Output directory: {OUTPUT_DIR}"
)

print(
    f"Report: {report_path}"
)

print(
    "\nIMPORTANT:"
)

print(
    "The affected-leaf ratio is a Grad-CAM-derived "
    "research estimate, not ground-truth disease area."
)
