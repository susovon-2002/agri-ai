from pathlib import Path
import json
import csv

import cv2
import numpy as np
import torch

from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from src.gradcam_compat import GradCAM, ClassifierOutputTarget


# ============================================================
# AgriVision AI - Full Disease Severity Engine
# ============================================================

TEST_DIR = Path(
    "data/processed/classification/test"
)

CHECKPOINT = Path(
    "models/classification/resnet18_best.pth"
)

OUTPUT_DIR = Path(
    "data/reports/severity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

IMAGE_SIZE = 224
BATCH_SIZE = 1

CAM_THRESHOLD = 0.50

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
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
    num_workers=0
)

class_names = dataset.classes

num_classes = len(
    class_names
)

print("=" * 75)
print("AgriVision AI - Full Disease Severity Engine")
print("=" * 75)

print(f"Device: {DEVICE}")
print(f"Test images: {len(dataset)}")
print(f"Classes: {num_classes}")


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
# GRAD-CAM
# ============================================================

target_layers = [
    model.layer4[-1]
]


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

    cleaned = np.zeros_like(
        mask
    )

    image_area = (
        IMAGE_SIZE *
        IMAGE_SIZE
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        if area > image_area * 0.01:

            cleaned[
                labels == label
            ] = 255

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final_mask = np.zeros_like(
        cleaned
    )

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

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
# SEVERITY FUNCTION
# ============================================================

def calculate_severity(
    predicted_class,
    affected_percentage
):

    class_name = class_names[
        predicted_class
    ]

    if "healthy" in class_name.lower():

        return "Healthy"

    if affected_percentage < 10:

        return "Low"

    elif affected_percentage < 25:

        return "Moderate"

    else:

        return "High"


# ============================================================
# OUTPUT STORAGE
# ============================================================

results = []

severity_counts = {
    "Healthy": 0,
    "Low": 0,
    "Moderate": 0,
    "High": 0
}


# ============================================================
# PROCESS ALL TEST IMAGES
# ============================================================

print()
print("Processing complete test set...")
print()

for index in tqdm(
    range(len(dataset)),
    desc="Severity Analysis"
):

    # --------------------------------------------------------
    # ORIGINAL IMAGE PATH
    # --------------------------------------------------------

    image_path, true_label = (
        dataset.samples[index]
    )

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        continue

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    # --------------------------------------------------------
    # MODEL INPUT
    # --------------------------------------------------------

    pil_image = transforms.ToPILImage()(
        cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    )

    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(
        DEVICE
    )

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
            ].item()
            * 100
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
    # GRAD-CAM ACTIVATION
    # --------------------------------------------------------

    cam_binary = (
        grayscale_cam >=
        CAM_THRESHOLD
    )

    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    fused_mask = (
        leaf_binary &
        cam_binary
    )

    # --------------------------------------------------------
    # AREA CALCULATION
    # --------------------------------------------------------

    leaf_pixels = np.sum(
        leaf_binary
    )

    disease_region_pixels = np.sum(
        fused_mask
    )

    total_pixels = (
        IMAGE_SIZE *
        IMAGE_SIZE
    )

    leaf_area_percentage = (
        leaf_pixels /
        total_pixels
    ) * 100

    attention_area_percentage = (
        disease_region_pixels /
        total_pixels
    ) * 100

    if leaf_pixels > 0:

        affected_leaf_percentage = (
            disease_region_pixels /
            leaf_pixels
        ) * 100

    else:

        affected_leaf_percentage = 0.0

    # --------------------------------------------------------
    # SEVERITY
    # --------------------------------------------------------

    severity = calculate_severity(
        predicted_class,
        affected_leaf_percentage
    )

    severity_counts[
        severity
    ] += 1

    # --------------------------------------------------------
    # CORRECTNESS
    # --------------------------------------------------------

    correct_prediction = (
        predicted_class ==
        true_label
    )

    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    results.append({

        "image": str(
            image_path
        ),

        "true_class": class_names[
            true_label
        ],

        "predicted_class":
            predicted_name,

        "correct_prediction":
            bool(
                correct_prediction
            ),

        "confidence_percent":
            round(
                confidence,
                4
            ),

        "leaf_area_percent":
            round(
                leaf_area_percentage,
                4
            ),

        "gradcam_threshold":
            CAM_THRESHOLD,

        "attention_area_percent":
            round(
                attention_area_percentage,
                4
            ),

        "affected_leaf_percent":
            round(
                affected_leaf_percentage,
                4
            ),

        "severity":
            severity
    })


# ============================================================
# SAVE JSON
# ============================================================

json_path = (
    OUTPUT_DIR /
    "severity_results.json"
)

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_path = (
    OUTPUT_DIR /
    "severity_results.csv"
)

if results:

    fieldnames = list(
        results[0].keys()
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "total_images":
        len(results),

    "severity_distribution":
        severity_counts,

    "cam_threshold":
        CAM_THRESHOLD,

    "severity_rules": {

        "Healthy":
            "Predicted class contains healthy",

        "Low":
            "affected leaf < 10%",

        "Moderate":
            "affected leaf >= 10% and < 25%",

        "High":
            "affected leaf >= 25%"
    }
}


summary_path = (
    OUTPUT_DIR /
    "severity_summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 75)
print("SEVERITY ENGINE COMPLETE")
print("=" * 75)

print(
    f"Images processed: {len(results)}"
)

print()

print("Severity distribution:")

for severity, count in (
    severity_counts.items()
):

    percentage = (
        count /
        len(results) *
        100
    ) if results else 0

    print(
        f"{severity:10s}: "
        f"{count:5d} "
        f"({percentage:.2f}%)"
    )

print()
print(
    f"JSON: {json_path}"
)

print(
    f"CSV:  {csv_path}"
)

print(
    f"Summary: {summary_path}"
)

print()
print(
    "AgriVision AI severity engine finished."
)
