"""
AgriVision AI - Feature Space Visualization

Visualizes ResNet18's 512-dimensional feature embeddings
using PCA and t-SNE.

Input:
    data/reports/similarity/resnet18_embedding_index.npz
    data/reports/similarity/resnet18_embedding_metadata.json

Output:
    data/reports/feature_space/
        pca_feature_space.png
        tsne_feature_space.png
        feature_space_summary.json
"""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INDEX_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "similarity"
    / "resnet18_embedding_index.npz"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "similarity"
    / "resnet18_embedding_metadata.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "feature_space"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_embeddings():

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Embedding index not found:\n{INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found:\n{METADATA_PATH}"
        )

    data = np.load(INDEX_PATH)

    embeddings = data["embeddings"]
    labels = data["labels"]

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    class_names = metadata["classes"]

    return embeddings, labels, class_names


def create_pca_plot(
    embeddings,
    labels,
    class_names,
):

    print()
    print("Running PCA...")

    pca = PCA(
        n_components=2,
        random_state=42,
    )

    reduced = pca.fit_transform(
        embeddings
    )

    explained_variance = (
        pca.explained_variance_ratio_
    )

    print(
        f"PCA component 1: "
        f"{explained_variance[0] * 100:.2f}%"
    )

    print(
        f"PCA component 2: "
        f"{explained_variance[1] * 100:.2f}%"
    )

    plt.figure(
        figsize=(14, 10)
    )

    for class_index, class_name in enumerate(
        class_names
    ):

        mask = labels == class_index

        if not np.any(mask):
            continue

        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=10,
            alpha=0.6,
            label=class_name,
        )

    plt.title(
        "AgriVision AI - ResNet18 Feature Space (PCA)"
    )

    plt.xlabel(
        f"PC1 ({explained_variance[0] * 100:.2f}% variance)"
    )

    plt.ylabel(
        f"PC2 ({explained_variance[1] * 100:.2f}% variance)"
    )

    plt.legend(
        fontsize=7,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "pca_feature_space.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"PCA plot saved:\n{output_path}"
    )

    return explained_variance


def create_tsne_plot(
    embeddings,
    labels,
    class_names,
):

    print()
    print("Preparing t-SNE...")

    # PCA first reduces 512 dimensions to 50.
    # This makes t-SNE substantially more efficient.
    pca = PCA(
        n_components=50,
        random_state=42,
    )

    reduced_50 = pca.fit_transform(
        embeddings
    )

    print(
        "Running t-SNE on 3,101 images..."
    )

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
        random_state=42,
    )

    tsne_result = tsne.fit_transform(
        reduced_50
    )

    plt.figure(
        figsize=(14, 10)
    )

    for class_index, class_name in enumerate(
        class_names
    ):

        mask = labels == class_index

        if not np.any(mask):
            continue

        plt.scatter(
            tsne_result[mask, 0],
            tsne_result[mask, 1],
            s=10,
            alpha=0.6,
            label=class_name,
        )

    plt.title(
        "AgriVision AI - ResNet18 Feature Space (t-SNE)"
    )

    plt.xlabel(
        "t-SNE Component 1"
    )

    plt.ylabel(
        "t-SNE Component 2"
    )

    plt.legend(
        fontsize=7,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "tsne_feature_space.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"t-SNE plot saved:\n{output_path}"
    )


def main():

    print("=" * 70)
    print("AgriVision AI - Feature Space Visualization")
    print("=" * 70)

    embeddings, labels, class_names = (
        load_embeddings()
    )

    print()
    print(
        f"Images: {len(embeddings)}"
    )

    print(
        f"Embedding dimensions: "
        f"{embeddings.shape[1]}"
    )

    print(
        f"Classes: {len(class_names)}"
    )

    explained_variance = create_pca_plot(
        embeddings,
        labels,
        class_names,
    )

    create_tsne_plot(
        embeddings,
        labels,
        class_names,
    )

    summary = {
        "total_images": int(
            len(embeddings)
        ),
        "embedding_dimension": int(
            embeddings.shape[1]
        ),
        "number_of_classes": int(
            len(class_names)
        ),
        "pca_explained_variance": [
            float(x)
            for x in explained_variance
        ],
        "pca_total_explained_variance": float(
            explained_variance.sum()
        ),
        "tsne_perplexity": 30,
        "tsne_iterations": 1000,
        "classes": class_names,
    }

    summary_path = (
        OUTPUT_DIR
        / "feature_space_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    print()
    print("=" * 70)
    print("FEATURE SPACE VISUALIZATION FINISHED")
    print("=" * 70)

    print(
        f"PCA: {OUTPUT_DIR / 'pca_feature_space.png'}"
    )

    print(
        f"t-SNE: {OUTPUT_DIR / 'tsne_feature_space.png'}"
    )

    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    main()