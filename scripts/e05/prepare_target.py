"""Prepare a frozen F1/M1 holdout from original WAVs and adjudicated boundaries."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
from alignment_probe import HOP, RATE, WAV_FORMATS, digest
from scipy.io import wavfile
from scipy.signal import resample_poly

from tiny_voca.text import BOS, EOS, PAD, PHONE_SET, SYMBOL_TO_ID, UNK, normalize_text

VOICES = {"F1", "M1"}


def text_key(text):
    return tuple(re.findall(r"[^\W_]+", normalize_text(text).casefold()))


def excluded_texts(paths):
    keys = {
        text_key(json.loads(line)["text"])
        for path in paths
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    }
    if not keys:
        raise ValueError(
            "Supply the original train and validation text manifests for leakage checks"
        )
    return keys


def verify_labels(row, seconds):
    reference = row["reference"]
    oracle = row["oracle"]
    if (
        not reference
        or not oracle
        or oracle[0] != BOS
        or oracle[-1] != EOS
        or any(s not in SYMBOL_TO_ID or s in (PAD, UNK, BOS, EOS) for s in oracle[1:-1])
        or [s for s in oracle if s in PHONE_SET] != [r["phone"] for r in reference]
    ):
        raise ValueError(f"Bad E01 phone sequence for {row['id']}")
    last = 0
    for phone in reference:
        start, end = phone["start"], phone["end"]
        if not (last <= start < end <= seconds):
            raise ValueError(f"Unordered or out-of-range boundaries for {row['id']}")
        last = end


def prepare(index, source_root, exclusions, output):
    index = index.resolve()
    source_root = source_root.resolve()
    metadata = json.loads(index.read_text(encoding="utf-8"))
    clips = metadata["clips"]
    for key in ("dataset_revision", "voice_revision", "annotation_revision"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Missing {key}")
    if {row["speaker"] for row in clips} != VOICES:
        raise ValueError("Both original E01 voices F1 and M1 are required")
    if len({row["id"] for row in clips}) != len(clips):
        raise ValueError("Duplicate clip identifiers")
    if len({row["sha256"] for row in clips}) != len(clips):
        raise ValueError("Duplicate source WAVs")
    excluded = excluded_texts(exclusions)

    checked = []
    for row in clips:
        name = row["id"]
        if (
            not re.fullmatch(r"[\w-]+", name, flags=re.ASCII)
            or not row["recording"].strip()
        ):
            raise ValueError("Expected a safe clip ID and its original recording group")
        key = text_key(row["text"])
        if not key or key in excluded:
            raise ValueError(f"Empty or train/validation-overlapping text for {name}")
        original = (source_root / row["wav"]).resolve()
        if (
            not original.is_relative_to(source_root)
            or original.suffix.lower() != ".wav"
        ):
            raise ValueError(f"Expected original WAV under the input root for {name}")
        if digest(original) != row["sha256"]:
            raise ValueError(f"Original WAV hash mismatch for {name}")
        info = sf.info(original)
        if info.format not in WAV_FORMATS or not info.frames or not info.samplerate:
            raise ValueError(f"Expected original WAV container for {name}")
        seconds = info.frames / info.samplerate
        verify_labels(row, seconds)
        checked.append((row, original, seconds))

    output.mkdir(parents=True)
    manifest = []
    for row, original, seconds in checked:
        wave, rate = sf.read(original, dtype="float32", always_2d=True)
        wave = wave.mean(axis=1)
        if not np.isfinite(wave).all():
            raise ValueError(f"Non-finite source samples for {row['id']}")
        if rate != RATE:
            divisor = math.gcd(rate, RATE)
            wave = resample_poly(wave, RATE // divisor, rate // divisor)
        wave = np.pad(wave, (0, (-len(wave)) % HOP))
        clip = output / (row["id"] + ".wav")
        wavfile.write(clip, RATE, wave)
        manifest.append(
            {
                "id": row["id"],
                "speaker": row["speaker"],
                "recording": row["speaker"] + "/" + row["recording"],
                "split": "holdout",
                "text": row["text"],
                "file": clip.name,
                "source_format": "original_wav",
                "source_sha256": row["sha256"],
                "fixture_sha256": digest(clip),
                "start": 0,
                "end": seconds,
                "reference": row["reference"],
                "oracle": row["oracle"],
            }
        )
    path = output / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    qualification = {
        "kind": "independent_teacher_wavs",
        "dataset_revision": metadata["dataset_revision"],
        "voice_revision": metadata["voice_revision"],
        "annotation_revision": metadata["annotation_revision"],
        "index_sha256": digest(index),
        "excluded_manifest_sha256": [digest(path) for path in exclusions],
        "manifest_sha256": digest(path),
        "clips_by_voice": dict(Counter(row["speaker"] for row in clips)),
        "phones_by_voice": {
            voice: sum(
                len(row["reference"]) for row in clips if row["speaker"] == voice
            )
            for voice in sorted(VOICES)
        },
    }
    (output / "qualification.json").write_text(
        json.dumps(qualification, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--exclude-texts", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.index, args.sources, args.exclude_texts, args.output)
