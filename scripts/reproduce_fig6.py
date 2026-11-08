import os

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from sklearn.impute import SimpleImputer

from gedi.plotter import FeaturesPlotter


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DATA_DIR = os.path.join(ROOT, "data")
PLOT_DIR = os.path.join(ROOT, "output", "plots")

PAPER_COLUMNS = [
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
MINERS = ("heu", "ilp", "imf")
METRICS = ("fitness", "precision", "fscore", "size", "cfc")


def load_benchmark_data():
    real = pd.read_csv(os.path.join(DATA_DIR, "BaselineED_bench.csv"))
    generated = pd.read_csv(os.path.join(DATA_DIR, "GenED_bench.csv"))
    real = real[PAPER_COLUMNS].copy()
    generated = generated[PAPER_COLUMNS].copy()
    return real, generated


def create_plot(generated, real, miner):
    columns = [f"{metric}_{miner}" for metric in METRICS]
    generated_features = generated[columns].copy()
    params = {
        "output_path": PLOT_DIR,
        "input_path": os.path.join(DATA_DIR, "GenED_bench.csv"),
        "real_eventlog_path": os.path.join(DATA_DIR, "BaselineED_bench.csv"),
        "plot_type": "boxplot",
        "font_size": 18,
        "boxplot_widt": 8,
    }
    FeaturesPlotter(generated_features, params)


def create_summary(real, generated):
    rows = []
    for source, frame in (("Real", real), ("Generated", generated)):
        for miner in MINERS:
            for metric in METRICS:
                column = f"{metric}_{miner}"
                values = pd.to_numeric(frame[column], errors="coerce").dropna()
                if values.empty:
                    continue
                rows.append(
                    {
                        "source": source,
                        "miner": miner,
                        "metric": metric,
                        "n_valid": len(values),
                        "missing": int(frame[column].isna().sum()),
                        "median": values.median(),
                        "q1": values.quantile(0.25),
                        "q3": values.quantile(0.75),
                        "mean": values.mean(),
                    }
                )
    summary_path = os.path.join(PLOT_DIR, "reproduction_fig6_summary.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)
    real, generated = load_benchmark_data()

    combined = pd.concat([real, generated], ignore_index=True)
    numeric_columns = [column for column in PAPER_COLUMNS if column != "log"]
    imputed = SimpleImputer(strategy="mean").fit_transform(combined[numeric_columns])
    if pd.isna(imputed).any():
        raise RuntimeError("Benchmark imputation left missing values.")

    for miner in MINERS:
        create_plot(generated, real, miner)

    summary_path = create_summary(real, generated)
    print(f"Real rows: {len(real)}; generated rows: {len(generated)}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
