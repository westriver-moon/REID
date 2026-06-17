import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def parse_args():
    parser = argparse.ArgumentParser(description="Plot high-resolution top-k sweep curves")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec in the format label=path/to/run_dir",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--tag", default="topk", help="Output filename prefix, e.g. topk50_official")
    return parser.parse_args()


def read_history(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed = {}
            for key, value in row.items():
                parsed[key] = int(value) if key == "epoch" else float(value)
            rows.append(parsed)
    return rows


def read_metrics(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)["metrics"]


def parse_run_specs(run_specs):
    parsed = []
    for spec in run_specs:
        if "=" not in spec:
            raise ValueError("Run spec must be label=path, got {}".format(spec))
        label, path = spec.split("=", 1)
        run_dir = Path(path)
        parsed.append(
            {
                "label": label,
                "run_dir": run_dir,
                "history": read_history(run_dir / "history.csv"),
                "all_metrics": read_metrics(run_dir / "eval_all_final.json"),
                "indoor_metrics": read_metrics(run_dir / "eval_indoor_final.json"),
            }
        )
    return parsed


def style_axis(axis, ylabel):
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    axis.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.45)
    axis.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.25)
    axis.minorticks_on()


def infer_epoch_count(runs):
    return max(run["history"][-1]["epoch"] for run in runs)


def plot_loss_curves(runs, output_dir, dpi, tag, epoch_count):
    figure, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True, constrained_layout=True)
    loss_keys = [
        ("train_loss", "Train Loss"),
        ("id_loss", "ID Loss"),
        ("triplet_loss", "Triplet Loss"),
    ]

    for axis, (loss_key, title) in zip(axes, loss_keys):
        for run in runs:
            epochs = [row["epoch"] for row in run["history"]]
            values = [row[loss_key] for row in run["history"]]
            axis.plot(
                epochs,
                values,
                linewidth=2.4,
                marker="o",
                markersize=4.5,
                label=run["label"],
            )
            axis.scatter(epochs[-1], values[-1], s=50)
            axis.annotate(
                "{:.4f}".format(values[-1]),
                (epochs[-1], values[-1]),
                textcoords="offset points",
                xytext=(8, 0),
                va="center",
                fontsize=9,
            )
        axis.set_title(title)
        style_axis(axis, title)
        axis.legend(loc="best", fontsize=10)

    figure.suptitle("LAST Official Top-k Sweep Loss Curves ({} Epochs)".format(epoch_count), fontsize=16)
    figure.savefig(output_dir / "{}_loss_curves.png".format(tag), dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / "{}_loss_curves.pdf".format(tag), bbox_inches="tight")
    plt.close(figure)


def plot_metric_curves(runs, output_dir, dpi, tag, epoch_count):
    figure, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    metric_keys = [
        ("all_mAP", "All-search mAP"),
        ("all_rank1", "All-search Rank-1"),
    ]

    for axis, (metric_key, title) in zip(axes, metric_keys):
        for run in runs:
            epochs = [row["epoch"] for row in run["history"]]
            values = [row[metric_key] for row in run["history"]]
            axis.plot(
                epochs,
                values,
                linewidth=2.4,
                marker="o",
                markersize=4.5,
                label=run["label"],
            )
            axis.scatter(epochs[-1], values[-1], s=50)
            axis.annotate(
                "{:.4f}".format(values[-1]),
                (epochs[-1], values[-1]),
                textcoords="offset points",
                xytext=(8, 0),
                va="center",
                fontsize=9,
            )
        axis.set_title(title)
        style_axis(axis, title)
        axis.legend(loc="best", fontsize=10)

    figure.suptitle("LAST Official Top-k Sweep Metric Curves ({} Epochs)".format(epoch_count), fontsize=16)
    figure.savefig(output_dir / "{}_metric_curves.png".format(tag), dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / "{}_metric_curves.pdf".format(tag), bbox_inches="tight")
    plt.close(figure)


def plot_final_bars(runs, output_dir, dpi, tag, epoch_count):
    labels = [run["label"] for run in runs]
    all_map = [run["all_metrics"]["mAP"] for run in runs]
    all_rank1 = [run["all_metrics"]["rank1"] for run in runs]
    indoor_map = [run["indoor_metrics"]["mAP"] for run in runs]
    indoor_rank1 = [run["indoor_metrics"]["rank1"] for run in runs]

    figure, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    bar_specs = [
        (axes[0, 0], all_map, "All-search mAP"),
        (axes[0, 1], all_rank1, "All-search Rank-1"),
        (axes[1, 0], indoor_map, "Indoor-search mAP"),
        (axes[1, 1], indoor_rank1, "Indoor-search Rank-1"),
    ]

    for axis, values, title in bar_specs:
        bars = axis.bar(labels, values)
        axis.set_title(title)
        axis.set_ylabel("Score")
        axis.grid(True, axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
        for bar, value in zip(bars, values):
            axis.annotate(
                "{:.4f}".format(value),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=9,
            )

    figure.suptitle("LAST Official Top-k Sweep Final Metrics ({} Epochs)".format(epoch_count), fontsize=16)
    figure.savefig(output_dir / "{}_final_metrics.png".format(tag), dpi=dpi, bbox_inches="tight")
    figure.savefig(output_dir / "{}_final_metrics.pdf".format(tag), bbox_inches="tight")
    plt.close(figure)


def write_summary(runs, output_dir, tag):
    summary = {}
    for run in runs:
        summary[run["label"]] = {
            "all": run["all_metrics"],
            "indoor": run["indoor_metrics"],
            "last_epoch": run["history"][-1],
        }
    with open(output_dir / "{}_summary.json".format(tag), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = parse_run_specs(args.run)
    epoch_count = infer_epoch_count(runs)
    plot_loss_curves(runs, output_dir, args.dpi, args.tag, epoch_count)
    plot_metric_curves(runs, output_dir, args.dpi, args.tag, epoch_count)
    plot_final_bars(runs, output_dir, args.dpi, args.tag, epoch_count)
    write_summary(runs, output_dir, args.tag)


if __name__ == "__main__":
    main()
