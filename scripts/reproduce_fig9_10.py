import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import kendalltau, pearsonr


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT, "data")
PLOT_DIR = os.path.join(ROOT, "output", "plots")
P_THRESHOLD = 0.05

FEATURE_COLUMNS = [
    "log",
    "ratio_variants_per_number_of_traces",
    "ratio_most_common_variant",
    "ratio_top_10_variants",
    "epa_normalized_variant_entropy",
    "epa_normalized_sequence_entropy",
    "epa_normalized_sequence_entropy_linear_forgetting",
    "epa_normalized_sequence_entropy_exponential_forgetting",
]
METRIC_COLUMNS = [
    "log",
    "fitness_heu",
    "precision_heu",
    "fscore_heu",
    "size_heu",
    "cfc_heu",
    "fitness_ilp",
    "precision_ilp",
    "fscore_ilp",
    "size_ilp",
    "cfc_ilp",
    "fitness_imf",
    "precision_imf",
    "fscore_imf",
    "size_imf",
    "cfc_imf",
]
TESTS = {"pearsonr": pearsonr, "kendalltau": kendalltau}


def load_data(benchmark_name):
    features = pd.read_csv(os.path.join(DATA_DIR, "BaselineED_feat.csv"))
    benchmark = pd.read_csv(os.path.join(DATA_DIR, benchmark_name))
    if benchmark_name == "GenBaselineED_bench.csv":
        benchmark["log"] = benchmark["log"].map(
            lambda value: value.replace("Gen", "").replace("genEL", "").rsplit("_", 7)[0]
        )
    merged = features[FEATURE_COLUMNS].merge(
        benchmark[METRIC_COLUMNS], on="log", how="inner"
    )
    return merged.dropna().reset_index(drop=True)


def compute_correlations(data, test_name):
    test = TESTS[test_name]
    feature_names = FEATURE_COLUMNS[1:]
    metric_names = METRIC_COLUMNS[1:]
    coefficients = pd.DataFrame(index=metric_names, columns=feature_names, dtype=float)
    p_values = coefficients.copy()
    for metric in metric_names:
        for feature in feature_names:
            statistic, p_value = test(data[feature], data[metric])
            coefficients.loc[metric, feature] = statistic
            p_values.loc[metric, feature] = p_value
    return coefficients, p_values


def abbreviate(name):
    return {
        "ratio_variants_per_number_of_traces": "rvpnot",
        "ratio_most_common_variant": "rmcv",
        "ratio_top_10_variants": "rt10v",
        "epa_normalized_variant_entropy": "enve",
        "epa_normalized_sequence_entropy": "ense",
        "epa_normalized_sequence_entropy_linear_forgetting": "enself",
        "epa_normalized_sequence_entropy_exponential_forgetting": "enseef",
    }.get(name, name)


def format_matrix(matrix):
    result = matrix.copy()
    result.index = [abbreviate(value) for value in result.index]
    result.columns = [abbreviate(value) for value in result.columns]
    return result


def save_heatmap(matrix, name, title, vmin=-1, vmax=1):
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        format_matrix(matrix),
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        cbar=True,
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Event Log features")
    axis.set_ylabel("Process Discovery metrics")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    path = os.path.join(PLOT_DIR, f"reproduction_fig9_10_{name}.png")
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    real = load_data("BaselineED_bench.csv")
    generated = load_data("GenBaselineED_bench.csv")
    details = []
    summary = []

    for test_name in TESTS:
        real_corr, real_p = compute_correlations(real, test_name)
        generated_corr, generated_p = compute_correlations(generated, test_name)
        difference = generated_corr.abs().subtract(real_corr.abs())
        filtered = generated_corr.where(generated_p <= P_THRESHOLD, 0.0)

        save_heatmap(
            difference,
            f"{test_name}_corrDiff",
            f"{test_name} absolute correlation difference (GenRT - Real)",
            vmin=-1,
            vmax=1,
        )
        save_heatmap(
            filtered,
            f"{test_name}_filtered",
            f"{test_name} significant GenRT correlations",
            vmin=-1,
            vmax=1,
        )

        for metric in METRIC_COLUMNS[1:]:
            for feature in FEATURE_COLUMNS[1:]:
                details.append(
                    {
                        "test": test_name,
                        "metric": metric,
                        "feature": feature,
                        "real_coefficient": real_corr.loc[metric, feature],
                        "real_p_value": real_p.loc[metric, feature],
                        "genrt_coefficient": generated_corr.loc[metric, feature],
                        "genrt_p_value": generated_p.loc[metric, feature],
                        "absolute_coefficient_difference": difference.loc[metric, feature],
                        "genrt_significant": int(
                            generated_p.loc[metric, feature] <= P_THRESHOLD
                        ),
                    }
                )
        summary.append(
            {
                "test": test_name,
                "real_rows": len(real),
                "genrt_rows": len(generated),
                "genrt_significant_pairs": int(
                    (generated_p <= P_THRESHOLD).to_numpy().sum()
                ),
                "mean_absolute_coefficient_difference": difference.to_numpy().mean(),
            }
        )

    detail_path = os.path.join(PLOT_DIR, "reproduction_fig9_10_details.csv")
    summary_path = os.path.join(PLOT_DIR, "reproduction_fig9_10_summary.csv")
    pd.DataFrame(details).to_csv(detail_path, index=False)
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    print(f"Real rows: {len(real)}; GenRT rows: {len(generated)}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved details: {detail_path}")


if __name__ == "__main__":
    main()
