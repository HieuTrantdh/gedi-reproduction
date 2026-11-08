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
DATA_SOURCES = ("BaselineED", "GenBaselineED", "GenED")
TESTS = {"kendalltau": kendalltau, "pearsonr": pearsonr}


def load_merged_data(data_source):
    features = pd.read_csv(os.path.join(DATA_DIR, f"{data_source}_feat.csv"))
    benchmark = pd.read_csv(os.path.join(DATA_DIR, f"{data_source}_bench.csv"))
    if data_source == "GenBaselineED":
        features["log"] = features["log"].map(lambda value: "Gen" + value)
        benchmark["log"] = benchmark["log"].map(
            lambda value: "Gen" + value.replace("genEL", "").rsplit("_", 7)[0]
        )
    features = features[FEATURE_COLUMNS]
    benchmark = benchmark[METRIC_COLUMNS]
    merged = features.merge(benchmark, on="log", how="inner")
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged.dropna().reset_index(drop=True)


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


def compute_test(data, test_name):
    test = TESTS[test_name]
    features = FEATURE_COLUMNS[1:]
    metrics = METRIC_COLUMNS[1:]
    result = pd.DataFrame(index=metrics, columns=features, dtype=float)
    rows = []
    for metric in metrics:
        for feature in features:
            if len(data) < 2:
                statistic, p_value, significant = float("nan"), float("nan"), 0
            else:
                statistic, p_value = test(data[feature], data[metric])
                significant = int(p_value <= P_THRESHOLD)
            result.loc[metric, feature] = significant
            rows.append(
                {
                    "feature": feature,
                    "metric": metric,
                    "statistic": statistic,
                    "p_value": p_value,
                    "significant": significant,
                }
            )
    result.index = [abbreviate(metric) for metric in result.index]
    result.columns = [abbreviate(feature) for feature in result.columns]
    return result, pd.DataFrame(rows)


def save_heatmap(result, data_source, test_name):
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        result,
        annot=True,
        cmap="viridis",
        annot_kws={"size": 9},
        vmin=-1,
        vmax=1,
        cbar=True,
        ax=axis,
    )
    axis.set_title(f"{test_name} significant feature-metric pairs: {data_source}")
    axis.set_xlabel("Event Log features")
    axis.set_ylabel("Process Discovery metrics")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    path = os.path.join(PLOT_DIR, f"reproduction_fig7_8_{test_name}_{data_source}.png")
    figure.savefig(path, dpi=300)
    plt.close(figure)
    return path


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    summary_rows = []
    detail_rows = []
    for data_source in DATA_SOURCES:
        data = load_merged_data(data_source)
        for test_name in TESTS:
            result, details = compute_test(data, test_name)
            details["data_source"] = data_source
            details["test"] = test_name
            detail_rows.append(details)
            save_heatmap(result, data_source, test_name)
            summary_rows.append(
                {
                    "data_source": data_source,
                    "test": test_name,
                    "rows_after_dropna": len(data),
                    "total_pairs": len(details),
                    "significant_pairs": int(details["significant"].sum()),
                }
            )

    summary_path = os.path.join(PLOT_DIR, "reproduction_fig7_8_summary.csv")
    detail_path = os.path.join(PLOT_DIR, "reproduction_fig7_8_details.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.concat(detail_rows, ignore_index=True).to_csv(detail_path, index=False)
    print(f"Saved summary: {summary_path}")
    print(f"Saved details: {detail_path}")


if __name__ == "__main__":
    main()
