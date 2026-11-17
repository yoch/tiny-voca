"""Free-running evaluation of a checkpoint. Cloud only.

For training sentences (memorisation), held-out sentences (generalisation) and
new sentences that are in no split, synthesise speech from TEXT ALONE (predicted
durations, pitch, voicing and energy), then

* transcribe generated and recorded audio with an ASR model and compare word
  error rates (the ASR model is an external evaluation tool, never part of the
  TTS model),
* compare timing, pitch statistics and a DTW-aligned log-mel distance with the
  recording,
* re-run the front end on the raw text and verify it reproduces the annotated
  tokens (preparation/inference consistency),
* write WAV pairs and a report for listening.

Numbers here support a listening test; they do not replace one.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch

from .config import Config
from .data import AnnotatedDataset
from .infer import load_model, speak
from .losses import SpectralLoss
from .text import normalize_for_wer, phonemize

NOVEL_SENTENCES = (
    "Le rapide renard brun saute par-dessus le chien paresseux près de la rivière calme.",
    "N'oublie pas d'appeler ta grand-mère avant le début du week-end.",
    "Combien de personnes viendront à la conférence en septembre ?",
    "Je n'arrive pas à croire qu'il soit déjà neuf heures et demie ; nous devrions partir maintenant !",
    "Tiny voca est un petit modèle de synthèse vocale entraîné de zéro en français.",
    "Après le passage de l'orage, l'air de la montagne semblait pur, frais et étonnamment doux.",
    "Pourriez-vous m'envoyer le rapport avant vendredi après-midi, s'il vous plaît ?",
    "Les chaussettes de l'archiduchesse sont-elles sèches, archi-sèches ?",
)
SEMITONES = 12 / np.log(2)


def normalizer(language="fr"):
    return lambda text: normalize_for_wer(text, language)


def spread(indices, lengths, count):
    """``count`` items spread over the length distribution (deterministic)."""
    ordered = sorted(indices, key=lambda i: (lengths[i], i))
    if len(ordered) <= count:
        return ordered
    return [ordered[int(q * (len(ordered) - 1))] for q in np.linspace(0.1, 0.9, count)]


def dtw_distance(a, b):
    """Mean absolute log-mel difference along the best monotonic path; a, b are [mel, frames]."""
    import librosa
    from scipy.spatial.distance import cdist
    cost = cdist(a.T, b.T, metric="cityblock") / a.shape[0]
    _, path = librosa.sequence.dtw(C=cost)
    return float(cost[path[:, 0], path[:, 1]].mean())


def pcm16(wave):
    """Samples for a 16-bit file and the attenuation applied (dB, 0 if none).
    A clip that peaks above full scale is scaled down as a whole, never
    hard-clipped; the attenuation is reported, not hidden."""
    wave = np.asarray(wave, dtype=np.float64)
    peak = float(np.abs(wave).max()) if wave.size else 0.0
    if peak <= 1.0:
        return wave, 0.0
    return wave / peak, round(20 * float(np.log10(peak)), 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="Directory with train/validation/test.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--train-subset", type=int, help="Restrict 'train' examples to the first N balanced rows (the --subset used in training)")
    parser.add_argument("--per-voice", type=int, default=4)
    parser.add_argument("--novel", type=int, default=len(NOVEL_SENTENCES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--asr-model", default="small")
    parser.add_argument("--asr-language", default="fr")
    parser.add_argument("--no-asr", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--pitch-range", type=float, default=1.0, help="Inference pitch-range setting under test (1 = as predicted)")
    args = parser.parse_args()
    import soundfile as sf
    from .train import balanced_indices
    model = load_model(args.checkpoint, args.device)
    c = model.config
    checkpoint_step = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["step"]
    spectral = SpectralLoss(c.sample_rate)
    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    datasets = {split: AnnotatedDataset(args.annotations / (split + ".jsonl"), c, cache=False) for split in args.splits}
    known_texts = set()
    for split in ("train", "validation", "test"):
        manifest = args.annotations / (split + ".jsonl")
        if manifest.exists():
            known_texts |= {json.loads(line)["text_hash"] for line in manifest.read_text(encoding="utf-8").split("\n") if line.strip()}

    jobs = []
    for split, dataset in datasets.items():
        candidates = list(range(len(dataset)))
        if split == "train" and args.train_subset:
            candidates = balanced_indices(dataset, args.train_subset)
        for voice in c.speakers:
            chosen = spread([i for i in candidates if dataset.rows[i]["speaker"] == voice], dataset.lengths, args.per_voice)
            jobs += [{"group": split, "voice": voice, "index": i, "id": dataset.rows[i]["id"], "text": dataset.rows[i]["text"]} for i in chosen]
    for number, text in enumerate(NOVEL_SENTENCES[:args.novel]):
        if hashlib.sha256(text.strip().encode()).hexdigest() in known_texts:
            raise ValueError("a 'novel' sentence is part of the dataset")
        jobs += [{"group": "novel", "voice": voice, "index": None, "id": f"novel{number:02d}_{voice}", "text": text} for voice in c.speakers]

    samples = []
    for job in jobs:
        wave, phones, details = speak(model, job["text"], job["voice"], speed=args.speed, seed=args.seed, pitch_range=args.pitch_range)
        wave = wave.numpy()
        generated_path = audio_dir / f"{job['id']}-generated.wav"
        samples_out, attenuation = pcm16(wave)
        sf.write(generated_path, samples_out, c.sample_rate, subtype="PCM_16")
        prediction = details["prosody"][0].float().cpu()
        voiced = prediction[:, 1] > 0
        record = {**{k: job[k] for k in ("group", "voice", "id", "text")}, "phonemes": " ".join(phones.symbols),
                  "generated": generated_path.name, "generated_seconds": round(len(wave) / c.sample_rate, 3),
                  "peak": round(float(np.abs(wave).max()), 4), "rms": round(float(np.sqrt(np.mean(wave ** 2))), 4),
                  "attenuated_db": attenuation,
                  "generated_voiced_fraction": round(float(voiced.float().mean()), 3),
                  "generated_pitch_mean_hz": round(float(c.pitch_reference_hz * prediction[voiced, 0].mean().exp()), 1) if voiced.any() else None,
                  "generated_pitch_std_semitones": round(float(SEMITONES * prediction[voiced, 0].std()), 2) if int(voiced.sum()) > 1 else None}
        if job["index"] is not None:
            dataset = datasets[job["group"]]
            row, item = dataset.rows[job["index"]], dataset[job["index"]]
            reference = item["wave"].numpy()
            reference_path = audio_dir / f"{job['id']}-reference.wav"
            sf.write(reference_path, pcm16(reference)[0], c.sample_rate, subtype="PCM_16")
            trusted = (item["prosody"][:, 1] > 0.5) & (item["pitch_trusted"] > 0.5)
            with torch.no_grad():
                mel_generated = spectral.log_mel(torch.from_numpy(wave)[None])[0].numpy()
                mel_reference = spectral.log_mel(item["wave"][None])[0].numpy()
            predicted_durations = details["durations"][0].cpu().tolist()
            record.update(
                reference=reference_path.name, reference_seconds=round(len(reference) / c.sample_rate, 3),
                tokens_match_annotation=list(phones.ids) == row["tokens"],
                duration_ratio=round(len(wave) / len(reference), 3),
                pause_frames_generated=int(sum(d for d, t in zip(predicted_durations, phones.ids) if not bool(model.phone_tokens[t]))),
                pause_frames_reference=int(sum(d for d, t in zip(row["durations"], row["tokens"]) if not bool(model.phone_tokens[t]))),
                pause_ratio=round(sum(d for d, t in zip(predicted_durations, phones.ids) if not bool(model.phone_tokens[t]))
                                  / max(1, sum(d for d, t in zip(row["durations"], row["tokens"]) if not bool(model.phone_tokens[t]))), 3),
                reference_voiced_fraction=round(float((item["prosody"][:, 1] > 0.5).float().mean()), 3),
                reference_pitch_mean_hz=round(float(c.pitch_reference_hz * item["prosody"][trusted, 0].mean().exp()), 1) if trusted.any() else None,
                reference_pitch_std_semitones=round(float(SEMITONES * item["prosody"][trusted, 0].std()), 2) if int(trusted.sum()) > 1 else None,
                mel_dtw=round(dtw_distance(mel_generated, mel_reference), 4))
        samples.append(record)
        print(json.dumps({"synthesised": record["id"], "seconds": record["generated_seconds"]}), flush=True)

    if not args.no_asr:
        import jiwer
        from faster_whisper import WhisperModel
        asr = WhisperModel(args.asr_model, device="cpu", compute_type="int8", cpu_threads=8)
        normalize = normalizer(args.asr_language)

        def transcribe(path):
            from scipy.signal import resample_poly
            audio, rate = sf.read(path, dtype="float32")
            factor = int(np.gcd(rate, 16000))      # the ASR model expects 16 kHz samples
            audio = resample_poly(audio, 16000 // factor, rate // factor).astype(np.float32)
            segments, _ = asr.transcribe(audio, language=args.asr_language, beam_size=5, condition_on_previous_text=False, temperature=0.0)
            return " ".join(segment.text.strip() for segment in segments).strip()
        for record in samples:
            truth = normalize(record["text"])
            for kind in ("generated", "reference"):
                if kind in record:
                    hypothesis = transcribe(audio_dir / record[kind])
                    record[kind + "_transcript"] = hypothesis
                    record[kind + "_wer"] = round(float(jiwer.wer(truth, normalize(hypothesis) or "<empty>")), 4)
            print(json.dumps({"transcribed": record["id"], "generated_wer": record.get("generated_wer"), "reference_wer": record.get("reference_wer")}), flush=True)

    def mean(group, voice, key):
        values = [s[key] for s in samples if s["group"] == group and s["voice"] in voice and s.get(key) is not None]
        return round(float(np.mean(values)), 4) if values else None
    summary = {}
    for group in sorted({s["group"] for s in samples}):
        summary[group] = {}
        for label, voices in (("all", c.speakers), *((v, (v,)) for v in c.speakers)):
            summary[group][label] = {"samples": sum(s["group"] == group and s["voice"] in voices for s in samples),
                                     **{key: mean(group, voices, key) for key in
                                        ("generated_wer", "reference_wer", "duration_ratio", "pause_ratio", "mel_dtw", "generated_pitch_std_semitones",
                                         "reference_pitch_std_semitones", "generated_voiced_fraction", "reference_voiced_fraction")}}
    mismatched = [s["id"] for s in samples if s.get("tokens_match_annotation") is False]
    report = {"tag": args.tag, "checkpoint": str(args.checkpoint), "step": checkpoint_step,
              "inference_parameters": sum(p.numel() for p in model.parameters()), "asr_model": None if args.no_asr else "faster-whisper " + args.asr_model,
              "note": "generated audio is free-running: text in, predicted durations/pitch/voicing/energy",
              "settings": {"speed": args.speed, "pitch_range": args.pitch_range, "seed": args.seed},
              "frontend_token_mismatches": mismatched, "summary": summary, "samples": samples}
    (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "frontend_token_mismatches": len(mismatched)}, indent=1))


if __name__ == "__main__":
    main()
