from pathlib import Path
from collections import defaultdict
from PIL import Image
import hashlib
import json

DATASET_ROOT = Path(
    r"C:\Users\PC\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage"
)

OUTPUT_DIR = Path("data/reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def find_images(root: Path):
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def calculate_hash(image_path: Path):
    """
    Calculate SHA-256 hash of the original image file.
    Identical files will have the same hash.
    """
    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def detect_duplicates():

    print("=" * 70)
    print("              AGRIVISION AI DUPLICATE DETECTOR")
    print("=" * 70)

    if not DATASET_ROOT.exists():
        print("\nERROR: Dataset directory does not exist.")
        print(DATASET_ROOT)
        return

    print(f"\nDataset root:")
    print(DATASET_ROOT)

    image_files = find_images(DATASET_ROOT)

    print(f"\nTotal images: {len(image_files):,}")

    if not image_files:
        print("No images found.")
        return

    hash_map = defaultdict(list)

    print("\n" + "-" * 70)
    print("CALCULATING SHA-256 HASHES")
    print("-" * 70)

    for index, image_path in enumerate(image_files, start=1):

        try:
            file_hash = calculate_hash(image_path)
            hash_map[file_hash].append(str(image_path))

        except Exception as error:
            print(f"\nERROR: {image_path}")
            print(error)

        if index % 5000 == 0:
            print(
                f"Processed {index:,}/{len(image_files):,} images"
            )

    duplicate_groups = {
        file_hash: paths
        for file_hash, paths in hash_map.items()
        if len(paths) > 1
    }

    duplicate_images = sum(
        len(paths) - 1
        for paths in duplicate_groups.values()
    )

    unique_images = len(hash_map)

    print("\n" + "=" * 70)
    print("DUPLICATE ANALYSIS")
    print("=" * 70)

    print(f"\nTotal images:        {len(image_files):,}")
    print(f"Unique files:        {unique_images:,}")
    print(f"Duplicate groups:    {len(duplicate_groups):,}")
    print(f"Duplicate images:    {duplicate_images:,}")

    if duplicate_groups:
        print("\n" + "-" * 70)
        print("DUPLICATE GROUPS")
        print("-" * 70)

        for number, (file_hash, paths) in enumerate(
            duplicate_groups.items(),
            start=1
        ):

            print(f"\nGroup {number}")
            print(f"Hash: {file_hash}")

            for path in paths:
                print(f"  {path}")

            if number >= 20:
                print(
                    "\nOnly the first 20 duplicate groups are displayed."
                )
                break

    else:
        print("\nNo exact duplicate files detected.")

    report = {
        "dataset_root": str(DATASET_ROOT),
        "total_images": len(image_files),
        "unique_images": unique_images,
        "duplicate_groups": len(duplicate_groups),
        "duplicate_images": duplicate_images,
        "duplicates": duplicate_groups,
    }

    report_path = OUTPUT_DIR / "duplicate_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n" + "=" * 70)
    print("DUPLICATE DETECTION COMPLETE")
    print("=" * 70)

    print(f"\nReport saved to:")
    print(report_path)


if __name__ == "__main__":
    detect_duplicates()