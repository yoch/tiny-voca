"""Detect recording clock shifts between the published WAV and MP3 archives."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import correlate, correlation_lags

RATE = 24000


def decode(path):
    data = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-ac",
            "1",
            "-ar",
            str(RATE),
            "-f",
            "f32le",
            "-",
        ]
    )
    return np.frombuffer(data, dtype="<f4")


def estimate(reference, candidate):
    # Use an interior window to avoid encoder padding and avoid comparing
    # differing leading/trailing silence lengths.
    count = min(len(reference), len(candidate), RATE * 12)
    x = candidate[RATE:count].astype(np.float64)
    y = reference[RATE:count].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    values = correlate(x, y, method="fft")
    lags = correlation_lags(len(x), len(y))
    choices = np.flatnonzero(np.abs(lags) <= RATE // 5)
    best = choices[np.argmax(values[choices])]
    return dict(
        mp3_delay_ms=float(1000 * lags[best] / RATE),
        similarity=float(values[best] / (np.linalg.norm(x) * np.linalg.norm(y))),
        wav_seconds=len(reference) / RATE,
        mp3_seconds=len(candidate) / RATE,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", type=Path)
    args = p.parse_args()
    rows = []
    for wav in sorted(args.audio.glob("Rhap-*.wav")):
        mp3 = wav.with_suffix(".mp3")
        if mp3.exists():
            rows.append(dict(recording=wav.stem, **estimate(decode(wav), decode(mp3))))
    if not rows:
        raise ValueError("No matching WAV/MP3 files")
    print(json.dumps(rows, indent=2))
