"""Synthesize a WAV with either Tiny Voca English ONNX variant."""

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort


def synthesize(model, tokens, stress, voice, threads):
    if not tokens or len(tokens) != len(stress):
        raise ValueError("expected equally long, nonempty token and stress sequences")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    with_shape = ort.InferenceSession(
        str(model), options, providers=["CPUExecutionProvider"]
    )
    inputs = {
        "tokens": np.asarray([tokens], dtype=np.int64),
        "stress": np.asarray([stress], dtype=np.int64),
        "voice": np.asarray([voice], dtype=np.int64),
        "speed": np.asarray([1.0], dtype=np.float32),
        "pitch_shift": np.asarray([0.0], dtype=np.float32),
        "pitch_range": np.asarray([1.0], dtype=np.float32),
    }
    audio, durations = with_shape.run(["audio", "durations"], inputs)
    if (
        durations.shape != (1, len(tokens))
        or np.any(durations < 0)
        or audio.shape != (1, int(durations.sum()) * 256)
        or not audio.size
        or not np.isfinite(audio).all()
    ):
        raise ValueError("invalid model output")
    return audio[0], durations[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--voice", choices=("F2", "M2"), default="F2")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--text", help="Optional. Requires checkpoint-matched dizione and DIZIONE_DATA."
    )
    args = parser.parse_args()
    if args.text is None:
        sample = json.loads(
            Path(__file__).with_name("sample.json").read_text(encoding="utf-8")
        )
        tokens, stress = sample["tokens"], sample["stress"]
    else:
        import dizione

        phones = dizione.phonemize(args.text)
        tokens, stress = list(phones.ids), list(phones.stress)
    audio, durations = synthesize(
        args.model, tokens, stress, ("F2", "M2").index(args.voice), args.threads
    )
    gain = 0.99 / max(float(np.abs(audio).max()), 0.99)
    pcm = np.rint(audio * gain * 32767).astype("<i2")
    with wave.open(str(args.output), "wb") as stream:
        stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        stream.writeframes(pcm.tobytes())
    print(f"Wrote {len(audio)} samples at 24 kHz, {len(durations)} token durations")


if __name__ == "__main__":
    main()
