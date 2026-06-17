import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_history(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: float(value) if key != "epoch" else int(value) for key, value in row.items()})
    return rows


def read_metrics(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["metrics"]


def plot_histories(baseline_history, last_history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs_base = [row["epoch"] for row in baseline_history]
    epochs_last = [row["epoch"] for row in last_history]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_base, [row["train_loss"] for row in baseline_history], label="Baseline")
    plt.plot(epochs_last, [row["train_loss"] for row in last_history], label="LAST")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_base, [row["all_mAP"] for row in baseline_history], label="Baseline")
    plt.plot(epochs_last, [row["all_mAP"] for row in last_history], label="LAST")
    plt.xlabel("Epoch")
    plt.ylabel("All-search mAP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close()


def plot_comparison(baseline_metrics, last_metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = ["rank1", "rank5", "rank10", "mAP"]
    baseline_values = [baseline_metrics["rank1"], baseline_metrics["rank5"], baseline_metrics["rank10"], baseline_metrics["mAP"]]
    last_values = [last_metrics["rank1"], last_metrics["rank5"], last_metrics["rank10"], last_metrics["mAP"]]

    xs = range(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar([x - width / 2 for x in xs], baseline_values, width=width, label="Baseline")
    plt.bar([x + width / 2 for x in xs], last_values, width=width, label="LAST")
    plt.xticks(list(xs), labels)
    plt.ylabel("Score")
    plt.ylim(0.0, max(baseline_values + last_values) * 1.15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "test_comparison.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot SYSU-MM01 experiment results")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--last-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    baseline_run = Path(args.baseline_run)
    last_run = Path(args.last_run)
    output_dir = Path(args.output_dir)

    baseline_history = read_history(baseline_run / "history.csv")
    last_history = read_history(last_run / "history.csv")
    baseline_metrics = read_metrics(baseline_run / "eval_all_final.json")
    last_metrics = read_metrics(last_run / "eval_all_final.json")

    plot_histories(baseline_history, last_history, output_dir)
    plot_comparison(baseline_metrics, last_metrics, output_dir)


if __name__ == "__main__":
    main()
