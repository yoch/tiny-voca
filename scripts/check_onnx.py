"""Compare an ONNX export with the PyTorch model and time it on the CPU.

    python check_onnx.py --checkpoint inference.pt --onnx test.onnx [--release tiny-voca.onnx]

``--onnx`` is an export made with ``--external-noise`` so the audio can be
compared sample by sample. ``--release`` is the normal export; it is timed and
its audio is written for listening.
"""

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch

from tiny_voca.export import Exportable
from tiny_voca.infer import load_model, match_frontend
from tiny_voca.text import phonemize

SENTENCES = (
    "Bonjour ! Voici tiny voca en français.",
    "Il fait très beau aujourd'hui, et je me sens vraiment détendu.",
    "As-tu vraiment payé quarante-deux euros et cinquante centimes pour cela ?",
    "Les chaussettes de l'archiduchesse sont-elles sèches, archi-sèches ?",
    "Après un long et difficile voyage à travers les montagnes, les voyageurs atteignirent enfin le petit village, "
    "où ils furent accueillis avec un repas chaud, un endroit pour dormir et mille questions sur le chemin parcouru.",
)


def session(path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def feeds(phones, voice, speed=1.0):
    one = lambda value: np.array([value], dtype=np.float32)
    return {"tokens": np.array([phones.ids], dtype=np.int64), "stress": np.array([phones.stress], dtype=np.int64),
            "voice": np.array([voice], dtype=np.int64), "speed": one(speed), "pitch_shift": one(0.0), "pitch_range": one(1.0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--output", type=Path, default=Path("onnx-check"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    match_frontend(args.checkpoint)
    model = load_model(args.checkpoint, "cpu")
    wrapper = Exportable(model, external_noise=True).eval()
    exact = session(args.onnx, args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    rate, hop = model.config.sample_rate, model.config.hop
    rows = []
    for number, text in enumerate(SENTENCES):
        phones = phonemize(text)
        for voice, name in enumerate(model.config.speakers):
            tokens, stress = torch.tensor([phones.ids]), torch.tensor([phones.stress])
            reference, lengths, details = model.synthesize(tokens, stress, torch.tensor([len(phones)]), torch.tensor([voice]))
            reference = reference[0, :int(lengths[0])].numpy()
            noise = torch.randn(len(reference), generator=torch.Generator().manual_seed(0))[None]
            one = lambda value: torch.tensor([value])
            with torch.no_grad():
                rewritten, durations = wrapper(tokens, stress, torch.tensor([voice]), one(1.0), one(0.0), one(1.0), noise)
            feed = feeds(phones, voice)
            padded = np.zeros((1, hop * model.config.max_frames), dtype=np.float32)
            padded[0, :noise.shape[1]] = noise[0].numpy()
            exported, exported_durations = exact.run(None, {**feed, "noise": padded})
            peak = float(np.abs(reference).max())
            rows.append({"sentence": number, "voice": name, "tokens": len(phones), "seconds": round(len(reference) / rate, 2),
                         "same_durations": bool((exported_durations == details["durations"].numpy()).all()),
                         "same_length": exported.shape[1] == len(reference),
                         "rewrite_error": float(np.abs(rewritten[0].numpy() - reference).max() / peak),
                         "onnx_error": float(np.abs(exported[0, :len(reference)] - reference).max() / peak) if exported.shape[1] == len(reference) else None})
            sf.write(args.output / f"s{number}-{name}-torch.wav", reference, rate)
            sf.write(args.output / f"s{number}-{name}-onnx-exact.wav", exported[0], rate)
    for row in rows:
        print(json.dumps(row))
    if args.release:
        release = session(args.release, args.threads)
        text = SENTENCES[1]
        phones = phonemize(text)
        for voice, name in enumerate(model.config.speakers):
            feed = feeds(phones, voice)
            for _ in range(5):
                release.run(None, feed)
            times = []
            for _ in range(20):
                start = time.perf_counter()
                audio = release.run(None, feed)[0]
                times.append(time.perf_counter() - start)
            tokens, stress = torch.tensor([phones.ids]), torch.tensor([phones.stress])
            torch_times = []
            for run in range(25):
                start = time.perf_counter()
                model.synthesize(tokens, stress, torch.tensor([len(phones)]), torch.tensor([voice]))
                if run >= 5:
                    torch_times.append(time.perf_counter() - start)
            seconds = audio.shape[1] / rate
            print(json.dumps({"voice": name, "threads": args.threads, "audio_seconds": round(seconds, 2),
                              "onnx_seconds": round(statistics.median(times), 4), "onnx_rtfx": round(seconds / statistics.median(times), 1),
                              "torch_seconds": round(statistics.median(torch_times), 4),
                              "torch_rtfx": round(seconds / statistics.median(torch_times), 1)}))
            for number, sentence in enumerate(SENTENCES):
                audio = release.run(None, feeds(phonemize(sentence), voice))[0]
                sf.write(args.output / f"s{number}-{name}-onnx.wav", audio[0], rate)
                print(json.dumps({"release_sentence": number, "voice": name, "seconds": round(audio.shape[1] / rate, 2),
                                  "peak": round(float(np.abs(audio).max()), 3), "finite": bool(np.isfinite(audio).all())}))


if __name__ == "__main__":
    main()
