import argparse
import json
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True)
    parser.add_argument("--output_manifest", required=True)
    parser.add_argument("--project_root", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target_sr", type=int, default=16000)
    parser.add_argument("--espeak_library", required=True)
    return parser.parse_args()


def make_relative_path(path_str, project_root):
    path = Path(path_str)
    if path.is_absolute():
        return str(path.relative_to(project_root))
    return str(path)


def load_audio(audio_path, target_sr):
    signal, sr = sf.read(audio_path)

    if signal.ndim > 1:
        signal = signal.mean(axis=1)

    signal = signal.astype(np.float32)

    if sr != target_sr:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return signal, sr


def predict_phonemes(signal, sr, processor, model, device):
    inputs = processor(signal, sampling_rate=sr, return_tensors="pt")

    input_values = inputs.input_values.to(device)
    attention_mask = inputs.attention_mask.to(device) if "attention_mask" in inputs else None

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    pred_phon = processor.batch_decode(predicted_ids)[0]
    pred_phon = " ".join(pred_phon.split())
    return pred_phon


def build_prediction_record(record, project_root, processor, model, device, target_sr):
    required_fields = ["utt_id", "lang", "wav_path", "ref_text", "ref_phon", "audio_md5"]
    for field in required_fields:
        if field not in record or record[field] in (None, ""):
            raise ValueError(f"Missing or empty field: {field}")

    wav_rel = make_relative_path(record["wav_path"], project_root)
    wav_abs = project_root / wav_rel

    if not wav_abs.exists():
        raise FileNotFoundError(f"Audio not found: {wav_abs}")

    signal, sr = load_audio(wav_abs, target_sr=target_sr)
    pred_phon = predict_phonemes(
        signal=signal,
        sr=sr,
        processor=processor,
        model=model,
        device=device,
    )

    return {
        "utt_id": record["utt_id"],
        "lang": record["lang"],
        "wav_path": wav_rel.replace("\\", "/"),
        "ref_text": record["ref_text"],
        "ref_phon": record["ref_phon"],
        "pred_phon": pred_phon,
        "audio_md5": record["audio_md5"],
        "snr_db": record.get("snr_db"),
    }


def main():
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    input_manifest = Path(args.input_manifest)
    output_manifest = Path(args.output_manifest)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    EspeakWrapper.set_library(args.espeak_library)

    processor = Wav2Vec2Processor.from_pretrained(args.model_name)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name)
    model.to(device)
    model.eval()

    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    prediction_records = []

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

            prediction_record = build_prediction_record(
                record=record,
                project_root=project_root,
                processor=processor,
                model=model,
                device=device,
                target_sr=args.target_sr,
            )
            prediction_records.append(prediction_record)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(output_manifest.parent),
        delete=False,
        suffix=".tmp",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        for record in prediction_records:
            tmp_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    tmp_path.replace(output_manifest)

    print(f"Prediction manifest written to: {output_manifest}")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Target sampling rate: {args.target_sr}")
    print(f"eSpeak library: {args.espeak_library}")
    print(f"Records: {len(prediction_records)}")


if __name__ == "__main__":
    main()