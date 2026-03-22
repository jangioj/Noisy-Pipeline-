import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_metrics", nargs="+", required=True)
    parser.add_argument("--output_plot", required=True)
    return parser.parse_args()


def load_metrics(metrics_paths):
    rows = []

    for metrics_path_str in metrics_paths:
        metrics_path = Path(metrics_path_str)

        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        required_fields = ["lang", "snr_db", "corpus_per"]
        for field in required_fields:
            if field not in metrics:
                raise ValueError(f"Missing field '{field}' in {metrics_path}")

        rows.append(
            {
                "lang": metrics["lang"],
                "snr_db": float(metrics["snr_db"]),
                "corpus_per": float(metrics["corpus_per"]),
            }
        )

    return rows


def group_rows_by_language(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["lang"]].append(row)

    for lang in grouped:
        grouped[lang] = sorted(grouped[lang], key=lambda x: x["snr_db"])

    return dict(grouped)


def compute_mean_curve(grouped_rows):
    snr_to_values = defaultdict(list)

    for lang_rows in grouped_rows.values():
        for row in lang_rows:
            snr_to_values[row["snr_db"]].append(row["corpus_per"])

    mean_rows = []
    for snr_db in sorted(snr_to_values):
        values = snr_to_values[snr_db]
        mean_rows.append(
            {
                "snr_db": snr_db,
                "corpus_per": sum(values) / len(values),
            }
        )

    return mean_rows


def main():
    args = parse_args()

    rows = load_metrics(args.input_metrics)

    if not rows:
        raise ValueError("No metrics found")

    grouped_rows = group_rows_by_language(rows)
    mean_rows = compute_mean_curve(grouped_rows)

    output_plot = Path(args.output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))

    for lang, lang_rows in sorted(grouped_rows.items()):
        snrs = [row["snr_db"] for row in lang_rows]
        pers = [row["corpus_per"] for row in lang_rows]
        plt.plot(snrs, pers, marker="o", label=lang)

    if len(grouped_rows) > 1:
        mean_snrs = [row["snr_db"] for row in mean_rows]
        mean_pers = [row["corpus_per"] for row in mean_rows]
        plt.plot(mean_snrs, mean_pers, marker="o", linestyle="--", label="mean")

    plt.xlabel("SNR (dB)")
    plt.ylabel("PER")

    if len(grouped_rows) == 1:
        only_lang = next(iter(grouped_rows))
        plt.title(f"Phoneme Error Rate vs Noise ({only_lang})")
    else:
        plt.title("Phoneme Error Rate vs Noise (all languages)")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()

    print(f"Plot written to: {output_plot}")
    print(f"Languages: {', '.join(sorted(grouped_rows))}")
    print(f"Language curves: {len(grouped_rows)}")
    print(f"Mean curve included: {len(grouped_rows) > 1}")


if __name__ == "__main__":
    main()