from pathlib import Path
import cv2
import matplotlib.pyplot as plt

TEST_DIR = Path("data/processed/classification/test")
MASK_DIR = Path("data/reports/segmentation/masks")
SEG_DIR = Path("data/reports/segmentation/segmented")

OUTPUT = Path(
    "data/reports/segmentation/"
    "segmentation_comparison.png"
)

classes = sorted([
    p.name for p in TEST_DIR.iterdir()
    if p.is_dir()
])

fig, axes = plt.subplots(
    len(classes),
    3,
    figsize=(12, 45)
)

for row, class_name in enumerate(classes):

    safe_name = (
        class_name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    mask_file = next(
        MASK_DIR.glob(f"*_{safe_name}_mask.png")
    )

    segmented_file = next(
        SEG_DIR.glob(f"*_{safe_name}_segmented.png")
    )

    original_files = []

    class_dir = TEST_DIR / class_name

    for ext in [
        "*.jpg", "*.jpeg", "*.png",
        "*.JPG", "*.JPEG", "*.PNG"
    ]:
        original_files.extend(class_dir.glob(ext))

    original_file = original_files[0]

    original = cv2.cvtColor(
        cv2.imread(str(original_file)),
        cv2.COLOR_BGR2RGB
    )

    mask = cv2.imread(
        str(mask_file),
        cv2.IMREAD_GRAYSCALE
    )

    segmented = cv2.cvtColor(
        cv2.imread(str(segmented_file)),
        cv2.COLOR_BGR2RGB
    )

    axes[row, 0].imshow(original)
    axes[row, 0].set_title(
        class_name.replace("_", " "),
        fontsize=9
    )
    axes[row, 0].axis("off")

    axes[row, 1].imshow(
        mask,
        cmap="gray"
    )
    axes[row, 1].set_title("Leaf Mask")
    axes[row, 1].axis("off")

    axes[row, 2].imshow(segmented)
    axes[row, 2].set_title("Segmented Leaf")
    axes[row, 2].axis("off")


fig.suptitle(
    "AgriVision AI - Leaf Segmentation Validation",
    fontsize=18,
    y=0.995
)

plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=200,
    bbox_inches="tight"
)

print(f"Saved: {OUTPUT}")

plt.show()
