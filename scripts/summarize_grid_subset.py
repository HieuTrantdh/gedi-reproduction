import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


TARGETS = [
    (0.1, 0.4, "e010"),
    (0.2, 0.4, "e020"),
    (0.3, 0.4, "e030"),
]
ENTROPY = "epa_normalized_sequence_entropy_linear_forgetting"
RATIO = "ratio_top_10_variants"
MINERS = ("ind", "heu", "imf")


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="output/reproduction_grid_2obj_subset/2_enself_rt10v",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    feature_dir = input_dir / "features"
    benchmark_dir = input_dir / "benchmark"
    rows = []

    for target_entropy, target_ratio, tag in TARGETS:
        log_files = sorted(input_dir.glob(f"*{tag}_r040*.xes"))
        feature_files = sorted(feature_dir.glob(f"*{tag}_r040*.json"))
        benchmark_files = sorted(benchmark_dir.glob(f"*{tag}_r040*.json"))
        if not log_files or not feature_files:
            rows.append(
                {
                    "target_entropy": target_entropy,
                    "target_ratio_top_10": target_ratio,
                    "status": "failure",
                    "actual_entropy": None,
                    "actual_ratio_top_10": None,
                    "absolute_entropy_error": None,
                    "absolute_ratio_error": None,
                }
            )
            continue

        features = load_json(feature_files[0])
        benchmark = load_json(benchmark_files[0]) if benchmark_files else {}
        row = {
            "target_entropy": target_entropy,
            "target_ratio_top_10": target_ratio,
            "status": "success",
            "log": log_files[0].stem,
            "actual_entropy": features.get(ENTROPY),
            "actual_ratio_top_10": features.get(RATIO),
            "absolute_entropy_error": abs(features[ENTROPY] - target_entropy)
            if features.get(ENTROPY) is not None
            else None,
            "absolute_ratio_error": abs(features[RATIO] - target_ratio)
            if features.get(RATIO) is not None
            else None,
        }
        for miner in MINERS:
            for metric in ("fitness", "precision", "fscore", "size", "cfc"):
                row[f"{metric}_{miner}"] = benchmark.get(f"{metric}_{miner}")
        rows.append(row)

    results = pd.DataFrame(rows)
    output_dir = input_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "grid_2obj_subset_summary.csv"
    plot_path = output_dir / "grid_2obj_subset_target_comparison.png"
    results.to_csv(csv_path, index=False)

    success = results[results["status"] == "success"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), dpi=140)
    comparisons = [
        ("target_entropy", "actual_entropy", "Linear-forgetting entropy"),
        ("target_ratio_top_10", "actual_ratio_top_10", "Top-10 variant ratio"),
    ]
    for axis, (target_col, actual_col, title) in zip(axes, comparisons):
        values = pd.concat(
            [success[target_col], success[actual_col]], ignore_index=True
        ).dropna()
        if not values.empty:
            lower = min(0.0, float(values.min()))
            upper = max(1.0, float(values.max()))
            axis.plot([lower, upper], [lower, upper], "--", color="gray")
        axis.scatter(
            success[target_col],
            success[actual_col],
            color="#4C78A8",
            s=55,
        )
        for index, (_, row) in enumerate(success.iterrows()):
            if pd.notna(row[actual_col]):
                axis.annotate(
                    f"{row['target_entropy']:.1f}",
                    (row[target_col], row[actual_col]),
                    xytext=(4, 4 + 10 * index),
                    textcoords="offset points",
                    fontsize=8,
                )
        axis.set_title(title)
        axis.set_xlabel("Target value")
        axis.set_ylabel("Observed value")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved summary: {csv_path}")
    print(f"Saved comparison plot: {plot_path}")
    print(f"Successful targets: {len(success)}/{len(results)}")


if __name__ == "__main__":
    main()
