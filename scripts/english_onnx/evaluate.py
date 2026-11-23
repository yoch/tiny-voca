"""Paired, seeded acoustic comparisons of self-contained ONNX models."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
from onnx import TensorProto, helper
from scipy.fft import irfft, rfft
from scipy.ndimage import maximum_filter1d
from scipy.signal import resample_poly

from tiny_voca.text import PHONE_IDS, phonemize, set_frontend

RATE, HOP = 24000, 256
DOCS = Path(__file__).resolve().parents[2] / "docs" / "english-onnx"
PUBLIC_SHA = "fe6d918791a15b82e37465981d7f27a070e46b46635e5a3482fc8dff87a1ae46"
WHISPER_SHA = "2a166925539a16005f14ff328359f9b9adb9dc4fb631bb3b227526862e93e2ef"
SPEAKER_SHA = "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068"
F0_LAGS = np.arange(40, 481)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def controlled_public(source, output):
    """Replace the released graph's single random node with a shared noise input."""
    if sha(source) != PUBLIC_SHA:
        raise ValueError("released English ONNX hash mismatch")
    model = onnx.load(source)
    add_noise_input(model)
    onnx.checker.check_model(model)
    onnx.save(model, output)


def add_noise_input(model):
    (index,) = (
        i for i, n in enumerate(model.graph.node) if n.op_type == "RandomNormalLike"
    )
    random = model.graph.node[index]
    model.graph.input.append(
        helper.make_tensor_value_info("noise", TensorProto.FLOAT, [1, "noise_samples"])
    )
    for name, value in (("noise_start", 0), ("noise_axis", 1)):
        model.graph.initializer.append(
            helper.make_tensor(name, TensorProto.INT64, [1], [value])
        )
    shape = helper.make_node("Shape", [random.input[0]], ["noise_end"], start=1, end=2)
    identity = helper.make_node(
        "Slice",
        ["noise", "noise_start", "noise_end", "noise_axis"],
        list(random.output),
    )
    del model.graph.node[index]
    model.graph.node.insert(index, shape)
    model.graph.node.insert(index + 1, identity)


def session(path, threads=1):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def cases(split):
    set_frontend("own")
    texts = (DOCS / (split + ".txt")).read_text().splitlines()
    for i, text in enumerate(texts):
        phones = phonemize(text)
        for voice in (0, 1):
            yield {
                "id": f"{i + 1:03}-{('F2', 'M2')[voice]}",
                "text": text,
                "voice": voice,
                "tokens": list(phones.ids),
                "stress": list(phones.stress),
                "speed": [1.0, 0.86, 1.15, 1.0, 1.0][i % 5],
                "pitch_shift": [0, 0.0578, -0.0578, 0, 0][i % 5],
                "pitch_range": [1.0, 0.85, 1.15, 1.0, 1.0][i % 5],
            }


def feeds(case, *, live=False, noise=None):
    ids, stress = (case["tokens"], case["stress"])
    if live:
        phones = phonemize(case["text"])
        ids, stress = phones.ids, phones.stress
    result = {
        "tokens": np.asarray([ids], dtype=np.int64),
        "stress": np.asarray([stress], dtype=np.int64),
        "voice": np.asarray([case["voice"]], dtype=np.int64),
    }
    result.update(
        {
            key: np.asarray([case[key]], dtype=np.float32)
            for key in ("speed", "pitch_shift", "pitch_range")
        }
    )
    if noise is not None:
        result["noise"] = noise
    return result


def render(runtime, data):
    wave, durations = (part[0] for part in runtime.run(["audio", "durations"], data))
    if not np.isfinite(wave).all() or len(wave) != int(durations.sum()) * HOP:
        raise ValueError("waveform or duration invariant")
    return wave, durations


def frames(wave, width=1024):
    padded = np.pad(wave, width // 2)
    return np.lib.stride_tricks.sliding_window_view(padded, width)[::HOP]


def hz_to_mel(value):
    return 2595 * np.log10(1 + value / 700)


def mel_bank():
    f = hz_to_mel(np.fft.rfftfreq(1024, 1 / RATE))
    m = np.linspace(hz_to_mel(35), f[-1], 82)
    rise = (f[:, None] - m[:-2]) / (m[1:-1] - m[:-2])
    fall = (m[2:] - f[:, None]) / (m[2:] - m[1:-1])
    return np.maximum(0, np.minimum(rise, fall)).astype(np.float32)


WINDOW = np.hanning(1024).astype(np.float32)
MEL_BANK = mel_bank()


def features(wave):
    x = frames(wave)
    mag = np.abs(rfft(x * WINDOW, axis=-1))
    mel = 10 * np.log10(np.maximum(mag**2 @ MEL_BANK, 1e-7))
    # Normalized autocorrelation for each lag. The earliest near-best local
    # maximum avoids regularly reporting subharmonics as F0.
    xx = x - x.mean(axis=1, keepdims=True)
    scores = irfft(np.abs(rfft(xx, 2048, axis=1)) ** 2, axis=1)[:, F0_LAGS]
    cumulative = np.pad(np.cumsum(xx**2, axis=1), ((0, 0), (1, 0)))
    left = cumulative[:, 1024 - F0_LAGS]
    right = cumulative[:, -1:] - cumulative[:, F0_LAGS]
    scores = scores / np.sqrt(np.maximum(left * right, 1e-12))
    confidence = scores.max(axis=1)
    near_peak = (scores == maximum_filter1d(scores, 3, axis=1)) & (
        scores >= 0.9 * confidence[:, None]
    )
    f0 = RATE / F0_LAGS[near_peak.argmax(axis=1)]
    rms = np.sqrt(np.mean(x**2, axis=1))
    voiced = (confidence >= 0.65) & (rms >= max(float(rms.max()) * 0.06, 1e-4))
    return mel, f0, voiced


def relative_frames(total, length):
    return np.rint(np.linspace(0, length - 1, total)).astype(np.int64)


def parity(baseline, candidate):
    if baseline.shape != candidate.shape:
        return {"snr_db": None, "normalized_peak": None}
    error = baseline.astype(np.float64) - candidate
    peak = float(np.abs(baseline).max())
    energy = np.dot(error, error)
    signal = np.sum(baseline.astype(np.float64) ** 2)
    return {
        "snr_db": float(10 * np.log10(signal / energy)) if energy else None,
        "normalized_peak": float(np.abs(error).max() / max(peak, 1e-12)),
        "sample_exact": not bool(energy),
    }


def summary(baseline, baseline_durations, baseline_features, trial, durations, tokens):
    mel, p, v = baseline_features
    other_mel, q, voiced = features(trial)
    index = relative_frames(len(mel), len(other_mel))
    both = v & voiced[index]
    error = np.abs(12 * np.log2(p[both] / q[index][both]))
    pauses = [i for i, token in enumerate(tokens) if token not in PHONE_IDS]
    return {
        **parity(baseline, trial),
        "duration_equal": bool(np.array_equal(baseline_durations, durations)),
        "duration_mae_frames": float(np.mean(np.abs(baseline_durations - durations))),
        "pause_mae_frames": float(
            np.mean(np.abs(baseline_durations[pauses] - durations[pauses]))
        )
        if pauses
        else 0,
        "original_samples": len(baseline),
        "samples": len(trial),
        "mel_rms_db": float(np.sqrt(np.mean((mel - other_mel[index]) ** 2))),
        "f0_mae_semitones": float(error.mean()) if error.size else None,
        "f0_p95_semitones": float(np.percentile(error, 95)) if error.size else None,
        "voicing_disagreement": float(np.mean(v != voiced[index])),
        "voiced_common_frames": int(both.sum()),
    }


def peak_limit(wave):
    peak = float(np.abs(wave).max())
    return wave * (0.99 / peak) if peak > 0.99 else wave


def independent_metrics(asr_path, speaker_path):
    import torch
    import torchaudio.compliance.kaldi as kaldi
    from faster_whisper import WhisperModel
    from faster_whisper.tokenizer import Tokenizer

    if asr_path and sha(asr_path / "model.bin") != WHISPER_SHA:
        raise ValueError("independent ASR model hash mismatch")
    if speaker_path and sha(speaker_path) != SPEAKER_SHA:
        raise ValueError("speaker embedding model hash mismatch")
    torch.set_num_threads(1)
    whisper = (
        WhisperModel(str(asr_path), device="cpu", compute_type="int8", cpu_threads=2)
        if asr_path
        else None
    )
    speaker = session(speaker_path) if speaker_path else None
    if whisper:
        tokenizer = Tokenizer(
            whisper.hf_tokenizer, whisper.model.is_multilingual, "transcribe", "en"
        )
        prompt = whisper.get_prompt(tokenizer, [], without_timestamps=True)

    def compare(wave):
        audio = resample_poly(peak_limit(wave), 2, 3)
        transcript = None
        if whisper:
            # Independent ASR on each utterance. Preserve end silence and
            # avoid encoding a full 30 s of zeros for a subsecond clip.
            seconds = len(audio) / 16000
            pad_to = max(8, int(np.ceil(seconds)) + 3) * 16000
            padded = np.pad(audio, (0, pad_to - len(audio)))
            features = whisper.feature_extractor(padded, padding=0)
            encoded = whisper.encode(np.ascontiguousarray(features, dtype=np.float32))
            result = whisper.model.generate(
                encoded,
                [prompt],
                beam_size=1,
                max_length=max(40, int(10 * seconds)),
                no_repeat_ngram_size=4,
            )
            transcript = tokenizer.decode(result[0].sequences_ids[0]).strip()
        embed = None
        if speaker:
            features = kaldi.fbank(
                torch.from_numpy(audio.copy())[None] * 32768,
                sample_frequency=16000,
                num_mel_bins=80,
                dither=0.0,
            )
            centered = features - features.mean(dim=0)
            embed = speaker.run(
                None, {speaker.get_inputs()[0].name: centered[None].numpy()}
            )[0].reshape(-1)
            embed = embed / max(float(np.linalg.norm(embed)), 1e-12)
        return transcript, embed

    return compare


def evaluate(args):
    samples = (
        json.loads(args.fixture.read_text())
        if args.fixture
        else list(cases(args.split))
    )
    if args.cases:
        args.cases.write_text(json.dumps(samples, indent=2) + "\n")
    original = session(args.original, args.threads)
    trials = {
        p.stem.removesuffix("-fixed"): session(p, args.threads) for p in args.trials
    }
    report = {
        "split": args.fixture.stem if args.fixture else args.split,
        "baseline_sha256": sha(args.original),
        "candidates": {p.stem.removesuffix("-fixed"): sha(p) for p in args.trials},
        "cases": [],
    }
    secondary = (
        independent_metrics(args.asr, args.speaker)
        if args.asr or args.speaker
        else None
    )
    if args.wavs:
        args.wavs.mkdir(parents=True, exist_ok=True)
    for i, case in enumerate(samples):
        noise = np.random.default_rng(1000 + i).standard_normal(
            (1, 786432), dtype=np.float32
        )
        data = feeds(case, noise=noise)
        reference, base_durations = render(original, data)
        source = features(reference)
        transcript, embedding = secondary(reference) if secondary else (None, None)
        result = {"case": case, "reference_transcript": transcript, "metrics": {}}
        save_wavs = args.wavs and (args.fixture or i // 2 in (0, 6, 25, 44, 63, 82))
        pair = {"original": reference} if save_wavs else None
        for name, runtime in trials.items():
            audio, durations = render(runtime, data)
            row = summary(
                reference, base_durations, source, audio, durations, case["tokens"]
            )
            if secondary and (not args.independent or name in args.independent):
                if row.get("sample_exact"):
                    trial_transcript, trial_embedding = transcript, embedding
                else:
                    trial_transcript, trial_embedding = secondary(audio)
                row["transcript"] = trial_transcript
                if embedding is not None:
                    row["speaker_cosine"] = float(np.dot(embedding, trial_embedding))
            result["metrics"][name] = row
            if pair is not None:
                pair[name] = audio
        if pair is not None:
            # A single gain across candidates avoids accidental loudness cues
            # from independent peak normalization in the listening fixture.
            largest = max(float(np.abs(audio).max()) for audio in pair.values())
            gain = 0.99 / largest if largest > 0.99 else 1.0
            for name, audio in pair.items():
                sf.write(
                    args.wavs / f"{name}-{case['id']}.wav",
                    audio * gain,
                    RATE,
                    subtype="PCM_16",
                )
        report["cases"].append(result)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(samples)}", flush=True)
    args.output.write_text(json.dumps(report, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public",
        type=Path,
        help="create the fixed-noise version of the released graph",
    )
    parser.add_argument("--original", type=Path)
    parser.add_argument(
        "--split", choices=("calibration", "holdout"), default="calibration"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--fixture", type=Path, help="Use pre-encoded cases and save every WAV"
    )
    parser.add_argument(
        "--independent",
        nargs="+",
        help="Variants to score with external ASR and speaker verification",
    )
    parser.add_argument("--output", type=Path, required=True)
    for key in ("asr", "speaker", "wavs", "cases"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("trials", nargs="*", type=Path)
    arguments = parser.parse_args()
    if arguments.public:
        controlled_public(arguments.public, arguments.output)
    else:
        evaluate(arguments)
