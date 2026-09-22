from pathlib import Path
import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "plantvillage"


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AGRIVISION AI - PLANTVILLAGE DATASET DOWNLOAD")
    print("=" * 60)

    print("\nDownloading PlantVillage dataset...")

    dataset_path = kagglehub.dataset_download(
        "emmarex/plantdisease"
    )

    print("\nDataset downloaded to:")
    print(dataset_path)

    print("\nCopying/using the downloaded dataset from KaggleHub.")
    print("Download completed successfully.")


if __name__ == "__main__":
    main()