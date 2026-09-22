from pathlib import Path
from collections import Counter
from PIL import Image
import json


DATASET_ROOT = Path(
    r"C:\Users\PC\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage"
)

OUTPUT_DIR = Path("data/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def find_image_files(root: Path):
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def get_class_name(image_path: Path):
    return image_path.parent.name


def inspect_dataset():
    print("=" * 70)
    print("              AGRIVISION AI DATASET INSPECTOR")
    print("=" * 70)

    if not DATASET_ROOT.exists():
        print("\nERROR: Dataset directory does not exist.")
        print(DATASET_ROOT)
        return

    print(f"\nDataset root:")
    print(DATASET_ROOT)

    image_files = find_image_files(DATASET_ROOT)

    print(f"\nTotal image files: {len(image_files):,}")

    if not image_files:
        print("No images found.")
        return

    class_counter = Counter(
        get_class_name(image)
        for image in image_files
    )

    print(f"\nNumber of classes: {len(class_counter)}")

    print("\n" + "-" * 70)
    print("CLASS DISTRIBUTION")
    print("-" * 70)

    for class_name, count in sorted(
        class_counter.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{class_name:<65} {count:>6}")

    print("\n" + "-" * 70)
    print("IMAGE VALIDATION")
    print("-" * 70)

    corrupted = []
    dimensions = Counter()

    for index, image_path in enumerate(image_files, start=1):

        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                dimensions[img.size] += 1

        except Exception as error:
            corrupted.append({
                "path": str(image_path),
                "error": str(error)
            })

        if index % 5000 == 0:
            print(f"Checked {index:,}/{len(image_files):,} images")

    print(f"\nCorrupted images: {len(corrupted):,}")

    print("\n" + "-" * 70)
    print("IMAGE DIMENSIONS")
    print("-" * 70)

    for dimension, count in dimensions.most_common(20):
        print(f"{str(dimension):<20} {count:>8}")

    report = {
        "dataset_root": str(DATASET_ROOT),
        "total_images": len(image_files),
        "number_of_classes": len(class_counter),
        "classes": dict(class_counter),
        "corrupted_images": corrupted,
        "image_dimensions": {
            str(k): v
            for k, v in dimensions.items()
        },
    }

    report_path = OUTPUT_DIR / "dataset_inspection.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)
    print(f"\nReport saved to:")
    print(report_path)


if __name__ == "__main__":
    inspect_dataset()