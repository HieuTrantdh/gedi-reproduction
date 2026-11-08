import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer, StandardScaler


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT, "data")
PLOT_DIR = os.path.join(ROOT, "output", "plots")
PAPER_FEATURES = [
    "ratio_variants_per_number_of_traces",
    "ratio_most_common_variant",
    "ratio_top_10_variants",
    "epa_normalized_variant_entropy",
    "epa_normalized_sequence_entropy",
    "epa_normalized_sequence_entropy_linear_forgetting",
    "epa_normalized_sequence_entropy_exponential_forgetting",
]


def plot_convex_hull(points, axis, color):
    hull = ConvexHull(points)
    boundary = points[hull.vertices]
    axis.plot(
        np.append(boundary[:, 0], boundary[0, 0]),
        np.append(boundary[:, 1], boundary[0, 1]),
        color=color,
    )
    axis.fill(boundary[:, 0], boundary[:, 1], color=color, alpha=0.1)
    return hull.volume


def main():
    real = pd.read_csv(os.path.join(DATA_DIR, "BaselineED_feat.csv"))
    generated = pd.read_csv(os.path.join(DATA_DIR, "GenED_feat.csv"))
    real = real[["log", *PAPER_FEATURES]].assign(source="Real")
    generated = generated[["log", *PAPER_FEATURES]].assign(source="Generated")
    combined = pd.concat([real, generated], ignore_index=True)

    values = combined[PAPER_FEATURES]
    imputed = SimpleImputer(strategy="mean").fit_transform(values)
    transformed = {
        "raw": imputed,
        "l2_normalized": Normalizer(norm="l2").fit_transform(imputed),
        "standardized": StandardScaler().fit_transform(imputed),
    }

    os.makedirs(PLOT_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=120)
    summary = []

    for axis, (name, matrix) in zip(axes, transformed.items()):
        pca = PCA(n_components=2)
        coordinates = pca.fit_transform(matrix)
        plot_data = pd.DataFrame(coordinates, columns=["PC1", "PC2"])
        plot_data["source"] = combined["source"].to_numpy()
        sns.scatterplot(
            data=plot_data,
            x="PC1",
            y="PC2",
            hue="source",
            style="source",
            alpha=0.9,
            s=45,
            ax=axis,
        )
        generated_points = plot_data.loc[
            plot_data["source"] == "Generated", ["PC1", "PC2"]
        ].to_numpy()
        real_points = plot_data.loc[
            plot_data["source"] == "Real", ["PC1", "PC2"]
        ].to_numpy()
        generated_area = plot_convex_hull(generated_points, axis, "red")
        real_area = plot_convex_hull(real_points, axis, "blue")
        axis.set_title(name)
        axis.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.2f}%)")
        axis.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.2f}%)")
        summary.append(
            {
                "representation": name,
                "pc1_explained_percent": pca.explained_variance_ratio_[0] * 100,
                "pc2_explained_percent": pca.explained_variance_ratio_[1] * 100,
                "generated_hull_area": generated_area,
                "real_hull_area": real_area,
            }
        )

    fig.tight_layout()
    figure_path = os.path.join(PLOT_DIR, "reproduction_fig4_5_pca.png")
    summary_path = os.path.join(PLOT_DIR, "reproduction_fig4_5_summary.csv")
    fig.savefig(figure_path, dpi=300, transparent=True)
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"Saved figure: {figure_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Real logs: {len(real)}; generated logs: {len(generated)}")


if __name__ == "__main__":
    main()
