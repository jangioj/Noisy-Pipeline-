import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--output_audio_dir", required=True)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--snr_db", required=True, type=float)
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def add_noise(signal, snr_db, rng):
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(noise_power),
        size=signal.shape,
    )
    return signal + noise


def add_noise_to_file(input_wav, output_wav, snr_db, seed=None):
    signal, sr = sf.read(input_wav)
    if signal.ndim != 1:
        raise ValueError("Only mono audio is supported")

    rng = np.random.default_rng(seed)
    noisy_signal = add_noise(signal, snr_db, rng)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_wav, noisy_signal, sr)


def make_relative_path(path_str, project_root):
    path = Path(path_str)
    if path.is_absolute():
        return str(path.relative_to(project_root))
    return str(path)


def build_output_audio_path(output_audio_dir, utt_id, snr_db):
    if float(snr_db).is_integer():
        snr_str = str(int(snr_db))
    else:
        snr_str = str(snr_db).replace(".", "p")
    return output_audio_dir / f"{utt_id}_snr{snr_str}.wav"


def build_noisy_record(record, project_root, output_audio_dir, snr_db, seed):
    required_fields = ["utt_id", "lang", "wav_path", "ref_text", "ref_phon", "audio_md5"]
    for field in required_fields:
        if field not in record or record[field] in (None, ""):
            raise ValueError(f"Missing or empty field: {field}")

    wav_rel = make_relative_path(record["wav_path"], project_root)
    wav_abs = project_root / wav_rel

    if not wav_abs.exists():
        raise FileNotFoundError(f"Audio not found: {wav_abs}")

    output_audio_path = build_output_audio_path(output_audio_dir, record["utt_id"], snr_db)
    add_noise_to_file(
        input_wav=str(wav_abs),
        output_wav=output_audio_path,
        snr_db=snr_db,
        seed=seed,
    )

    return {
        "utt_id": record["utt_id"],
        "lang": record["lang"],
        "wav_path": str(output_audio_path.relative_to(project_root)).replace("\\", "/"),
        "ref_text": record["ref_text"],
        "ref_phon": record["ref_phon"],
        "audio_md5": record["audio_md5"],
        "snr_db": snr_db,
    }


def main():
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    input_manifest = Path(args.input_manifest)
    output_manifest = Path(args.output_manifest)
    output_audio_dir = Path(args.output_audio_dir)

    if not output_audio_dir.is_absolute():
        output_audio_dir = project_root / output_audio_dir

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_audio_dir.mkdir(parents=True, exist_ok=True)

    noisy_records = []

    with input_manifest.open("r", encoding="utf-8") as f_in:
        for line_num, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {input_manifest} at line {line_num}: {e}"
                ) from e

            noisy_record = build_noisy_record(
                record=record,
                project_root=project_root,
                output_audio_dir=output_audio_dir,
                snr_db=args.snr_db,
                seed=args.seed,
            )
            noisy_records.append(noisy_record)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(output_manifest.parent),
        delete=False,
        suffix=".tmp",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        for record in noisy_records:
            tmp_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp_path.replace(output_manifest)

    print(f"Noisy manifest written to: {output_manifest}")
    print(f"Noisy audio directory: {output_audio_dir}")
    print(f"SNR: {args.snr_db}")
    print(f"Records: {len(noisy_records)}")


if __name__ == "__main__":
    main()