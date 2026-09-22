from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
import shutil
import json

# ============================================================
# AGRIVISION AI - DATASET SPLITTER
# ============================================================

# IMPORTANT:
# Use ONLY the canonical PlantVillage directory.
# Do NOT use PlantVillage\PlantVillage.
DATASET_ROOT = Path(
    r"C:\Users\PC\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage"
)

OUTPUT_ROOT = Path("data/processed/classification")

TRAIN_DIR = OUTPUT_ROOT / "train"
VAL_DIR = OUTPUT_ROOT / "val"
TEST_DIR = OUTPUT_ROOT / "test"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def find_images():

    images = []

    for class_dir in DATASET_ROOT.iterdir():

        if not class_dir.is_dir():
            continue

        # Ignore the nested duplicate directory
        if class_dir.name == "PlantVillage":
            continue

        for image_path in class_dir.iterdir():

            if (
                image_path.is_file()
                and image_path.suffix.lower() in IMAGE_EXTENSIONS
            ):
                images.append(image_path)

    return images


def create_directories(classes):

    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:

        for class_name in classes:

            (split_dir / class_name).mkdir(
                parents=True,
                exist_ok=True
            )


def copy_files(files, destination):

    for image_path in files:

        class_name = image_path.parent.name

        target_dir = destination / class_name

        target_path = target_dir / image_path.name

        shutil.copy2(
            image_path,
            target_path
        )


def main():

    print("=" * 70)
    print("             AGRIVISION AI DATASET SPLITTER")
    print("=" * 70)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():

        print("\nERROR: Dataset directory not found.")
        print(DATASET_ROOT)

        return

    print("\nDataset source:")
    print(DATASET_ROOT)

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = find_images()

    print(f"\nCanonical images found: {len(images):,}")

    if not images:

        print("\nERROR: No images found.")

        return

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    class_counter = Counter(
        image.parent.name
        for image in images
    )

    classes = sorted(class_counter.keys())

    print(f"Number of classes: {len(classes)}")

    print("\nOriginal class distribution:")
    print("-" * 70)

    for class_name in classes:

        print(
            f"{class_name:<65}"
            f"{class_counter[class_name]:>6}"
        )

    # --------------------------------------------------------
    # Create directories
    # --------------------------------------------------------

    print("\nCreating dataset directories...")

    create_directories(classes)

    # --------------------------------------------------------
    # Split each class independently
    # --------------------------------------------------------

    split_report = {}

    for class_name in classes:

        class_images = [
            image
            for image in images
            if image.parent.name == class_name
        ]

        # First split:
        # 70% train
        # 30% temporary
        train_files, temp_files = train_test_split(
            class_images,
            test_size=(VAL_SIZE + TEST_SIZE),
            random_state=RANDOM_STATE,
            shuffle=True
        )

        # Second split:
        # 15% validation
        # 15% test
        val_files, test_files = train_test_split(
            temp_files,
            test_size=0.50,
            random_state=RANDOM_STATE,
            shuffle=True
        )

        print(
            f"\n{class_name}"
        )

        print(
            f"  Total: {len(class_images):,}"
        )

        print(
            f"  Train: {len(train_files):,}"
        )

        print(
            f"  Val:   {len(val_files):,}"
        )

        print(
            f"  Test:  {len(test_files):,}"
        )

        copy_files(
            train_files,
            TRAIN_DIR
        )

        copy_files(
            val_files,
            VAL_DIR
        )

        copy_files(
            test_files,
            TEST_DIR
        )

        split_report[class_name] = {
            "total": len(class_images),
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
        }

    # --------------------------------------------------------
    # Calculate totals
    # --------------------------------------------------------

    total_train = sum(
        value["train"]
        for value in split_report.values()
    )

    total_val = sum(
        value["validation"]
        for value in split_report.values()
    )

    total_test = sum(
        value["test"]
        for value in split_report.values()
    )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    report = {
        "dataset_source": str(DATASET_ROOT),
        "total_images": len(images),
        "number_of_classes": len(classes),
        "classes": classes,
        "split_ratio": {
            "train": TRAIN_SIZE,
            "validation": VAL_SIZE,
            "test": TEST_SIZE,
        },
        "totals": {
            "train": total_train,
            "validation": total_val,
            "test": total_test,
        },
        "class_distribution": split_report,
    }

    report_path = Path(
        "data/reports/dataset_split_report.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("                 SPLIT COMPLETE")
    print("=" * 70)

    print(
        f"\nTotal images:       {len(images):,}"
    )

    print(
        f"Training images:    {total_train:,}"
    )

    print(
        f"Validation images:  {total_val:,}"
    )

    print(
        f"Testing images:     {total_test:,}"
    )

    print(
        f"\nClasses:            {len(classes)}"
    )

    print("\nDataset location:")

    print(
        OUTPUT_ROOT
    )

    print("\nReport:")

    print(
        report_path
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
