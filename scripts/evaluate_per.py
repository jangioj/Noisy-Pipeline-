#computing PER: Phoneme Error Rate from ref_phon and pred_phon

import argparse
import json
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_metrics", required=True)
    return parser.parse_args()


def levenshtein_ops(ref_tokens, hyp_tokens):
    n = len(ref_tokens)
    m = len(hyp_tokens)

    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        cost, s, d, ins = dp[i - 1][0]
        dp[i][0] = (cost + 1, s, d + 1, ins)

    for j in range(1, m + 1):
        cost, s, d, ins = dp[0][j - 1]
        dp[0][j] = (cost + 1, s, d, ins + 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                match = dp[i - 1][j - 1]
                candidates = [match]
            else:
                cost, s, d, ins = dp[i - 1][j - 1]
                sub = (cost + 1, s + 1, d, ins)

                cost, s, d, ins = dp[i - 1][j]
                delete = (cost + 1, s, d + 1, ins)

                cost, s, d, ins = dp[i][j - 1]
                insert = (cost + 1, s, d, ins + 1)

                candidates = [sub, delete, insert]

            dp[i][j] = min(candidates, key=lambda x: (x[0], x[1] + x[2] + x[3]))

    _, s, d, ins = dp[n][m]
    return s, d, ins


def tokenise_phonemes(phon_str):
    phon_str = "".join(phon_str.split())
    if not phon_str:
        return []
    return list(phon_str)


def evaluate_manifest(input_manifest):
    total_s = 0
    total_d = 0
    total_i = 0
    total_n = 0

    utterances = 0
    utterance_pers = []

    lang = None
    snr_db = None

    with input_manifest.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {input_manifest} at line {line_num}: {e}"
                ) from e

            required_fields = ["utt_id", "lang", "ref_phon", "pred_phon"]
            for field in required_fields:
                if field not in record:
                    raise ValueError(f"Missing field '{field}' in line {line_num}")

            ref_tokens = tokenise_phonemes(record["ref_phon"])
            pred_tokens = tokenise_phonemes(record["pred_phon"])

            s, d, ins = levenshtein_ops(ref_tokens, pred_tokens)

            n = len(ref_tokens)
            if n == 0:
                raise ValueError(f"Empty ref_phon for utt_id={record['utt_id']}")

            utt_per = (s + d + ins) / n

            total_s += s
            total_d += d
            total_i += ins
            total_n += n

            utterances += 1
            utterance_pers.append(utt_per)

            if lang is None:
                lang = record["lang"]

            if snr_db is None:
                snr_db = record.get("snr_db")

    if utterances == 0:
        raise ValueError(f"No records found in {input_manifest}")

    corpus_per = (total_s + total_d + total_i) / total_n
    mean_utt_per = sum(utterance_pers) / len(utterance_pers)

    return {
        "lang": lang,
        "snr_db": snr_db,
        "num_utterances": utterances,
        "num_ref_phonemes": total_n,
        "substitutions": total_s,
        "deletions": total_d,
        "insertions": total_i,
        "corpus_per": corpus_per,
        "mean_utt_per": mean_utt_per,
    }


def main():
    args = parse_args()

    input_manifest = Path(args.input_manifest)
    output_metrics = Path(args.output_metrics)

    output_metrics.parent.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_manifest(input_manifest)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(output_metrics.parent),
        delete=False,
        suffix=".tmp",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        json.dump(metrics, tmp_file, ensure_ascii=False, indent=2)

    tmp_path.replace(output_metrics)

    print(f"Metrics written to: {output_metrics}")
    print(f"Language: {metrics['lang']}")
    print(f"SNR: {metrics['snr_db']}")
    print(f"Utterances: {metrics['num_utterances']}")
    print(f"Corpus PER: {metrics['corpus_per']:.6f}")


if __name__ == "__main__":
    main()