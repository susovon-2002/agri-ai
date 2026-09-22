from pathlib import Path
import random
import json

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

TEST_DIR = Path("data/processed/classification/test")
OUTPUT_DIR = Path("data/reports/segmentation")

MASK_DIR = OUTPUT_DIR / "masks"
SEGMENTED_DIR = OUTPUT_DIR / "segmented"

MASK_DIR.mkdir(parents=True, exist_ok=True)
SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 256


# ============================================================
# FIND ONE IMAGE PER CLASS
# ============================================================

class_dirs = sorted([
    p for p in TEST_DIR.iterdir()
    if p.is_dir()
])

print("=" * 65)
print("AgriVision AI - Leaf Segmentation Baseline")
print("=" * 65)

print(f"Classes found: {len(class_dirs)}")


# ============================================================
# SEGMENTATION FUNCTION
# ============================================================

def segment_leaf(image):

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # Green vegetation mask
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Saturation-based vegetation support
    # --------------------------------------------------------

    saturation_mask = cv2.inRange(
        hsv,
        np.array([15, 20, 20], dtype=np.uint8),
        np.array([110, 255, 255], dtype=np.uint8)
    )

    # Combine masks
    mask = cv2.bitwise_or(
        green_mask,
        saturation_mask
    )

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Keep significant connected components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    cleaned = np.zeros_like(mask)

    image_area = IMAGE_SIZE * IMAGE_SIZE

    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        # Ignore tiny components
        if area > image_area * 0.01:
            cleaned[labels == label] = 255

    # --------------------------------------------------------
    # Fill small holes
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final_mask = np.zeros_like(cleaned)

    if contours:

        # Keep reasonably large contours
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
# PROCESS ONE IMAGE PER CLASS
# ============================================================

results = []

for index, class_dir in enumerate(class_dirs, 1):

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

    if not images:
        print(
            f"[{index}/{len(class_dirs)}] "
            f"No image found: {class_dir.name}"
        )
        continue

    image_path = random.choice(images)

    print(
        f"\n[{index}/{len(class_dirs)}] "
        f"{class_dir.name}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        print("Could not read image.")
        continue

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # --------------------------------------------------------
    # Generate leaf mask
    # --------------------------------------------------------

    mask = segment_leaf(image)

    # --------------------------------------------------------
    # Calculate estimated leaf coverage
    # --------------------------------------------------------

    leaf_pixels = np.sum(mask > 0)

    total_pixels = mask.shape[0] * mask.shape[1]

    leaf_percentage = (
        leaf_pixels /
        total_pixels
    ) * 100

    # --------------------------------------------------------
    # Apply mask to image
    # --------------------------------------------------------

    segmented = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    # --------------------------------------------------------
    # Safe filename
    # --------------------------------------------------------

    safe_name = (
        class_dir.name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    mask_path = (
        MASK_DIR /
        f"{index:02d}_{safe_name}_mask.png"
    )

    segmented_path = (
        SEGMENTED_DIR /
        f"{index:02d}_{safe_name}_segmented.png"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cv2.imwrite(
        str(mask_path),
        mask
    )

    cv2.imwrite(
        str(segmented_path),
        segmented
    )

    result = {
        "class": class_dir.name,
        "image": str(image_path),
        "mask": str(mask_path),
        "segmented": str(segmented_path),
        "estimated_leaf_area_percent": round(
            leaf_percentage,
            2
        )
    }

    results.append(result)

    print(
        f"Estimated leaf coverage: "
        f"{leaf_percentage:.2f}%"
    )


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    OUTPUT_DIR /
    "leaf_segmentation_results.json"
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
print("LEAF SEGMENTATION COMPLETE")
print("=" * 65)

print(f"Results: {OUTPUT_DIR}")
print(f"Report: {report_path}")

print(
    "\nIMPORTANT:"
)

print(
    "This is a baseline leaf/background segmentation. "
    "It is not ground-truth disease segmentation."
)
