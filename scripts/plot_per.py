import argparse
import json
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


def main():
    args = parse_args()

    rows = load_metrics(args.input_metrics)

    if not rows:
        raise ValueError("No metrics found")

    langs = {row["lang"] for row in rows}
    if len(langs) != 1:
        raise ValueError("This plotting script expects metrics from a single language")

    lang = rows[0]["lang"]

    rows = sorted(rows, key=lambda x: x["snr_db"])

    snrs = [row["snr_db"] for row in rows]
    pers = [row["corpus_per"] for row in rows]

    output_plot = Path(args.output_plot)
    output_plot.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(snrs, pers, marker="o")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PER")
    plt.title(f"Phoneme Error Rate vs Noise ({lang})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()

    print(f"Plot written to: {output_plot}")
    print(f"Language: {lang}")
    print(f"Points: {len(rows)}")

if __name__ == "__main__":
    main()