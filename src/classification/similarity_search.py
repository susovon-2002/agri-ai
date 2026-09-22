"""
AgriVision AI - Image Similarity Search

Uses the trained ResNet18 feature space to find the
most visually similar PlantVillage test images.

The system:
1. Loads ResNet18
2. Extracts 512-dimensional embeddings
3. Builds a searchable embedding index
4. Saves the index to disk
5. Finds Top-K similar images using cosine similarity
"""

from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "classification"
    / "test"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "classification"
    / "resnet18_best.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "similarity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

INDEX_PATH = (
    OUTPUT_DIR
    / "resnet18_embedding_index.npz"
)

METADATA_PATH = (
    OUTPUT_DIR
    / "resnet18_embedding_metadata.json"
)

CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_resnet18():

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(
            model.fc.in_features,
            len(CLASS_NAMES),
        ),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)
    model.eval()

    return model


def create_feature_extractor(model):

    extractor = nn.Sequential(
        *list(model.children())[:-1]
    )

    extractor.to(DEVICE)
    extractor.eval()

    return extractor


@torch.no_grad()
def extract_embedding(
    extractor,
    image_path,
):
    """
    Extract a normalized 512-dimensional
    ResNet18 embedding.
    """

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    features = extractor(
        tensor
    )

    features = torch.flatten(
        features,
        1,
    )

    features = torch.nn.functional.normalize(
        features,
        p=2,
        dim=1,
    )

    return (
        features
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def build_embedding_index(
    extractor,
):
    """
    Extract embeddings for every image
    in the PlantVillage test set.
    """

    embeddings = []
    paths = []
    labels = []

    print()
    print("=" * 70)
    print("BUILDING RESNET18 EMBEDDING INDEX")
    print("=" * 70)

    total_images = 0

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):

        class_dir = (
            TEST_DIR / class_name
        )

        if not class_dir.exists():

            print(
                f"WARNING: Missing class: "
                f"{class_name}"
            )

            continue

        image_files = sorted(
            [
                p
                for p in class_dir.iterdir()
                if (
                    p.is_file()
                    and p.suffix.lower()
                    in IMAGE_EXTENSIONS
                )
            ]
        )

        print(
            f"{class_name}: "
            f"{len(image_files)} images"
        )

        for image_path in image_files:

            try:

                embedding = extract_embedding(
                    extractor,
                    image_path,
                )

                embeddings.append(
                    embedding
                )

                paths.append(
                    str(image_path.relative_to(
                        PROJECT_ROOT
                    ))
                )

                labels.append(
                    class_index
                )

                total_images += 1

            except Exception as exc:

                print(
                    f"Skipping "
                    f"{image_path.name}: "
                    f"{exc}"
                )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    np.savez_compressed(
        INDEX_PATH,
        embeddings=embeddings,
        labels=labels,
    )

    metadata = {
        "total_images": total_images,
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "classes": CLASS_NAMES,
        "image_paths": paths,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print(
        f"Total indexed images: "
        f"{total_images}"
    )

    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Index saved to: "
        f"{INDEX_PATH}"
    )

    print(
        f"Metadata saved to: "
        f"{METADATA_PATH}"
    )

    return (
        embeddings,
        labels,
        paths,
    )


def load_index():

    if (
        not INDEX_PATH.exists()
        or not METADATA_PATH.exists()
    ):
        return None

    data = np.load(
        INDEX_PATH
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(file)

    return (
        data["embeddings"],
        data["labels"],
        metadata["image_paths"],
    )


def cosine_similarity(
    query_embedding,
    database_embeddings,
):
    """
    Calculate cosine similarity.

    All embeddings are already L2-normalized,
    so cosine similarity is simply a dot product.
    """

    scores = np.dot(
        database_embeddings,
        query_embedding,
    )

    return scores


def search_similar_images(
    query_image,
    extractor,
    embeddings,
    labels,
    paths,
    top_k=5,
):
    """
    Find the Top-K visually similar images.
    """

    query_embedding = extract_embedding(
        extractor,
        query_image,
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings,
    )

    # Sort from highest similarity
    # to lowest similarity.
    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):

        class_index = int(
            labels[index]
        )

        results.append(
            {
                "rank": rank,
                "image": paths[index],
                "class_index": class_index,
                "class_name": CLASS_NAMES[
                    class_index
                ],
                "similarity": float(
                    scores[index]
                ),
                "similarity_percent": float(
                    scores[index] * 100
                ),
            }
        )

    return results


def print_results(
    query_image,
    results,
):
    print()
    print("=" * 70)
    print("SIMILARITY SEARCH RESULTS")
    print("=" * 70)

    print(
        f"Query image: "
        f"{query_image}"
    )

    print()

    for result in results:

        print(
            f"{result['rank']}. "
            f"{result['class_name']}"
        )

        print(
            f"   Similarity: "
            f"{result['similarity_percent']:.2f}%"
        )

        print(
            f"   Image: "
            f"{result['image']}"
        )

        print()


def save_results(
    query_image,
    results,
):
    output = {
        "query_image": str(
            query_image
        ),
        "top_k": len(results),
        "results": results,
    }

    output_path = (
        OUTPUT_DIR
        / "similarity_search_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print(
        f"Results saved to: "
        f"{output_path}"
    )


def main():

    print("=" * 70)
    print("AgriVision AI - Image Similarity Search")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: "
            f"{MODEL_PATH}"
        )

    if not TEST_DIR.exists():

        raise FileNotFoundError(
            f"Test directory not found: "
            f"{TEST_DIR}"
        )

    print()
    print("Loading ResNet18...")

    model = load_resnet18()

    extractor = create_feature_extractor(
        model
    )

    print(
        "ResNet18 feature extractor loaded."
    )

    index = load_index()

    if index is None:

        print()
        print(
            "No saved embedding index found."
        )

        embeddings, labels, paths = (
            build_embedding_index(
                extractor
            )
        )

    else:

        embeddings, labels, paths = index

        print()
        print(
            f"Loaded saved index: "
            f"{len(embeddings)} images"
        )

    # Use one known test image for the first test.
    query_image = (
        TEST_DIR
        / "Tomato_Late_blight"
        / next(
            p.name
            for p in (
                TEST_DIR
                / "Tomato_Late_blight"
            ).iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        )
    )

    print()
    print(
        f"Testing query image:"
    )

    print(
        query_image
    )

    results = search_similar_images(
        query_image=query_image,
        extractor=extractor,
        embeddings=embeddings,
        labels=labels,
        paths=paths,
        top_k=5,
    )

    print_results(
        query_image,
        results,
    )

    save_results(
        query_image,
        results,
    )

    print()
    print("=" * 70)
    print("SIMILARITY SEARCH FINISHED")
    print("=" * 70)


if __name__ == "__main__":
    main()