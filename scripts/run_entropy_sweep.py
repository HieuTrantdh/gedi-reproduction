import argparse
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


TARGETS = [0.1, 0.2, 0.3, 0.4, 0.5]
FEATURE_NAME = "epa_normalized_sequence_entropy_linear_forgetting"


def build_config(output_path: str) -> list[dict]:
    return [{
        "pipeline_step": "event_logs_generation",
        "targets": [
            {
                "log": f"target_entropy_{int(target * 100):03d}",
                "ratio_top_20_variants": 0.4,
                FEATURE_NAME: target,
            }
            for target in TARGETS
        ],
        "config_space": {
            "mode": [5, 20],
            "sequence": [0.01, 1],
            "choice": [0.01, 1],
            "parallel": [0.01, 1],
            "loop": [0.01, 1],
            "silent": [0.01, 1],
            "lt_dependency": [0.01, 1],
            "num_traces": [10, 100],
            "duplicate": [0],
            "or": [0],
        },
        "system_params": {"n_trials": 5, "output_path": output_path},
    }]


def feature_directory(run_directory: Path) -> Path:
    candidates = list(run_directory.glob("*/features"))
    if not candidates:
        return run_directory / "features"
    return candidates[0]


def collect_run(run_directory: Path, run_name: str) -> list[dict]:
    features = feature_directory(run_directory)
    rows = []
    for target in TARGETS:
        target_name = f"entropy_{int(target * 100):03d}"
        files = list(features.glob(f"*{target_name}*.json"))
        if not files:
            rows.append(
                {
                    "run": run_name,
                    "target_entropy": target,
                    "status": "failure",
                    "actual_entropy": None,
                    "actual_ratio": None,
                    "target_similarity": None,
                    "absolute_entropy_error": None,
                }
            )
            continue
        with files[0].open(encoding="utf-8") as handle:
            result = json.load(handle)
        actual = result.get(FEATURE_NAME)
        rows.append(
            {
                "run": run_name,
                "target_entropy": target,
                "status": "success",
                "actual_entropy": actual,
                "actual_ratio": result.get("ratio_top_20_variants"),
                "target_similarity": result.get("target_similarity"),
                "absolute_entropy_error": abs(actual - target)
                if actual is not None
                else None,
            }
        )
    return rows


def summarize(rows: list[dict], output_directory: Path) -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output_directory / "entropy_sweep_100_summary.csv", index=False)
    successful = frame[frame["status"] == "success"]
    summary = (
        frame.groupby("target_entropy", as_index=False)
        .agg(
            attempts=("status", "size"),
            successes=("status", lambda values: (values == "success").sum()),
        )
        .assign(
            success_rate=lambda data: data["successes"] / data["attempts"],
        )
    )
    statistics = (
        successful.groupby("target_entropy", as_index=False)
        .agg(
            actual_entropy_mean=("actual_entropy", "mean"),
            actual_entropy_std=("actual_entropy", "std"),
            absolute_error_mean=("absolute_entropy_error", "mean"),
            target_similarity_mean=("target_similarity", "mean"),
        )
    )
    summary = summary.merge(statistics, on="target_entropy", how="left")
    summary.to_csv(output_directory / "entropy_sweep_100_by_target.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=140)
    sns.barplot(
        data=summary,
        x="target_entropy",
        y="success_rate",
        color="#4C78A8",
        ax=axes[0],
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("Target entropy")
    axes[0].set_ylabel("Success rate")
    axes[0].set_title("Generation success by target")
    axes[0].yaxis.set_major_formatter(
        plt.FuncFormatter(lambda value, _: f"{value:.0%}")
    )
    sns.boxplot(
        data=successful,
        x="target_entropy",
        y="actual_entropy",
        color="#F58518",
        ax=axes[1],
    )
    sns.stripplot(
        data=successful,
        x="target_entropy",
        y="actual_entropy",
        color="black",
        size=2.5,
        alpha=0.45,
        ax=axes[1],
    )
    axes[1].plot(
        range(len(TARGETS)),
        TARGETS,
        color="#54A24B",
        linestyle="--",
        marker="o",
        label="Target",
    )
    axes[1].set_xlabel("Target entropy")
    axes[1].set_ylabel("Observed entropy")
    axes[1].set_title("Observed entropy across successful runs")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_directory / "entropy_sweep_100_consistency.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-runs", type=int, default=100)
    parser.add_argument("--output-root", default="output/student_entropy_sweep_100")
    parser.add_argument("--python", default=None)
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    python = args.python or "python"
    rows = []

    for number in range(1, args.total_runs + 1):
        run_name = f"run_{number:03d}"
        run_directory = root / run_name
        summary_files = list(run_directory.glob("*/features/*.json"))
        if not summary_files:
            config_path = root / f"{run_name}.json"
            config_path.write_text(
                json.dumps(
                    build_config(str(run_directory)),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            subprocess.run(
                [python, "main.py", "-a", str(config_path)],
                check=True,
            )
            config_path.unlink()
        rows.extend(collect_run(run_directory, run_name))

    summarize(rows, root)
    print(f"Saved {len(rows)} target observations to {root}")


if __name__ == "__main__":
    main()
