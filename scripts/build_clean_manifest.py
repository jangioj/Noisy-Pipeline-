import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--espeak_lang", required=True)
    return parser.parse_args()


def phonemize_text(text, espeak_lang):
    """
    Chiama espeak-ng e restituisce la trascrizione fonemica del testo.
    """
    result = subprocess.run(
        [
            "espeak-ng",
            "-q",
            "--ipa=3",
            "-v",
            espeak_lang,
            text,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )

    phon = result.stdout.strip()
    phon = " ".join(phon.split())
    return phon


def make_relative_path(path_str, project_root):
    """
    Converte un path assoluto in path relativo alla root del progetto.
    Se è già relativo, lo lascia relativo.
    """
    path = Path(path_str)

    if path.is_absolute():
        return str(path.relative_to(project_root))

    return str(path)


def build_clean_record(record, project_root, espeak_lang):
    """
    Trasforma un record source in un record clean.
    """
    required_fields = ["utt_id", "lang", "wav_path", "ref_text", "audio_md5"]
    for field in required_fields:
        if field not in record or record[field] in (None, ""):
            raise ValueError(f"Campo mancante o vuoto: {field}")

    ref_text = record["ref_text"].strip()
    if not ref_text:
        raise ValueError(f"ref_text vuoto per utt_id={record.get('utt_id')}")

    wav_rel = make_relative_path(record["wav_path"], project_root)
    wav_abs = project_root / wav_rel

    if not wav_abs.exists():
        raise FileNotFoundError(f"Audio non trovato: {wav_abs}")

    ref_phon = phonemize_text(ref_text, espeak_lang)

    clean_record = {
        "utt_id": record["utt_id"],
        "lang": record["lang"],
        "wav_path": wav_rel.replace("\\", "/"),
        "ref_text": ref_text,
        "ref_phon": ref_phon,
        "audio_md5": record["audio_md5"],
        "snr_db": None,
    }

    return clean_record


def main():
    args = parse_args()

    input_manifest = Path(args.input_manifest)
    output_manifest = Path(args.output_manifest)
    project_root = Path(args.project_root).resolve()

    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    with input_manifest.open("r", encoding="utf-8") as f_in:
        records = []
        for line_num, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"JSON non valido in {input_manifest} alla riga {line_num}: {e}"
                ) from e

            clean_record = build_clean_record(
                record=record,
                project_root=project_root,
                espeak_lang=args.espeak_lang,
            )
            records.append(clean_record)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(output_manifest.parent),
        delete=False,
        suffix=".tmp",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

        for record in records:
            tmp_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp_path.replace(output_manifest)

    print(f"Manifest 'clean.jsonl'': {output_manifest}")
    print(f"Numero record: {len(records)}")


if __name__ == "__main__":
    main()