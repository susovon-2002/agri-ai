from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT = Path(
    "data/reports/severity/severity_results.csv"
)

OUTPUT_DIR = Path(
    "data/reports/severity/graphs"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)

print("=" * 70)
print("AgriVision AI - Severity Analytics")
print("=" * 70)

print(f"Images: {len(df)}")

print(
    f"Diseases/classes: "
    f"{df['predicted_class'].nunique()}"
)


# ============================================================
# 1. SEVERITY DISTRIBUTION
# ============================================================

severity_order = [
    "Healthy",
    "Low",
    "Moderate",
    "High"
]

severity_counts = (
    df["severity"]
    .value_counts()
    .reindex(
        severity_order,
        fill_value=0
    )
)

plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    severity_counts.index,
    severity_counts.values
)

plt.title(
    "AgriVision AI - Severity Distribution"
)

plt.xlabel(
    "Severity Level"
)

plt.ylabel(
    "Number of Images"
)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    severity_counts.values
):

    plt.text(
        bar.get_x() +
        bar.get_width() / 2,
        bar.get_height(),
        str(value),
        ha="center",
        va="bottom"
    )

plt.tight_layout()

severity_graph = (
    OUTPUT_DIR /
    "severity_distribution.png"
)

plt.savefig(
    severity_graph,
    dpi=250
)

plt.close()

print(
    f"Saved: {severity_graph}"
)


# ============================================================
# 2. DISEASE-WISE SEVERITY
# ============================================================

severity_table = pd.crosstab(
    df["predicted_class"],
    df["severity"]
)

severity_table = severity_table.reindex(
    columns=severity_order,
    fill_value=0
)

severity_table = severity_table.sort_index()

ax = severity_table.plot(
    kind="bar",
    figsize=(15, 8)
)

plt.title(
    "AgriVision AI - Disease-wise Severity Distribution"
)

plt.xlabel(
    "Disease / Class"
)

plt.ylabel(
    "Number of Images"
)

plt.xticks(
    rotation=75,
    ha="right"
)

plt.legend(
    title="Severity"
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

disease_severity_graph = (
    OUTPUT_DIR /
    "disease_wise_severity.png"
)

plt.savefig(
    disease_severity_graph,
    dpi=250
)

plt.close()

print(
    f"Saved: {disease_severity_graph}"
)


# ============================================================
# 3. AVERAGE AFFECTED LEAF %
# ============================================================

affected = (
    df[
        ~df["predicted_class"]
        .str.contains(
            "healthy",
            case=False,
            na=False
        )
    ]
)

mean_affected = (
    affected
    .groupby("predicted_class")
    ["affected_leaf_percent"]
    .mean()
    .sort_values(
        ascending=False
    )
)

plt.figure(
    figsize=(14, 8)
)

bars = plt.bar(
    mean_affected.index,
    mean_affected.values
)

plt.title(
    "AgriVision AI - Average Estimated Affected Leaf Region"
)

plt.xlabel(
    "Disease"
)

plt.ylabel(
    "Affected Leaf Region (%)"
)

plt.xticks(
    rotation=75,
    ha="right"
)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    mean_affected.values
):

    plt.text(
        bar.get_x() +
        bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()

affected_graph = (
    OUTPUT_DIR /
    "average_affected_leaf_region.png"
)

plt.savefig(
    affected_graph,
    dpi=250
)

plt.close()

print(
    f"Saved: {affected_graph}"
)


# ============================================================
# 4. AVERAGE CONFIDENCE BY DISEASE
# ============================================================

confidence = (
    df
    .groupby("predicted_class")
    ["confidence_percent"]
    .mean()
    .sort_values(
        ascending=False
    )
)

plt.figure(
    figsize=(14, 8)
)

bars = plt.bar(
    confidence.index,
    confidence.values
)

plt.title(
    "AgriVision AI - Average Prediction Confidence"
)

plt.xlabel(
    "Disease / Class"
)

plt.ylabel(
    "Average Confidence (%)"
)

plt.xticks(
    rotation=75,
    ha="right"
)

plt.ylim(
    0,
    100
)

plt.grid(
    axis="y",
    alpha=0.3
)

for bar, value in zip(
    bars,
    confidence.values
):

    plt.text(
        bar.get_x() +
        bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()

confidence_graph = (
    OUTPUT_DIR /
    "average_confidence_by_disease.png"
)

plt.savefig(
    confidence_graph,
    dpi=250
)

plt.close()

print(
    f"Saved: {confidence_graph}"
)


# ============================================================
# 5. SAVE SUMMARY TABLE
# ============================================================

summary = (
    df
    .groupby("predicted_class")
    .agg(
        images=(
            "predicted_class",
            "size"
        ),
        average_confidence=(
            "confidence_percent",
            "mean"
        ),
        average_leaf_area=(
            "leaf_area_percent",
            "mean"
        ),
        average_affected_leaf=(
            "affected_leaf_percent",
            "mean"
        )
    )
    .reset_index()
)

summary["average_confidence"] = (
    summary["average_confidence"]
    .round(2)
)

summary["average_leaf_area"] = (
    summary["average_leaf_area"]
    .round(2)
)

summary["average_affected_leaf"] = (
    summary["average_affected_leaf"]
    .round(2)
)

summary_path = (
    OUTPUT_DIR /
    "severity_disease_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print(
    f"Saved: {summary_path}"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("SEVERITY ANALYTICS COMPLETE")
print("=" * 70)

print(
    f"Graphs directory: {OUTPUT_DIR}"
)

print()
print(
    "Generated:"
)

print(
    "1. severity_distribution.png"
)

print(
    "2. disease_wise_severity.png"
)

print(
    "3. average_affected_leaf_region.png"
)

print(
    "4. average_confidence_by_disease.png"
)

print(
    "5. severity_disease_summary.csv"
)
