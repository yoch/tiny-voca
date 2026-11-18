"""Make a small, recording-disjoint French timing fixture from Rhapsodie."""

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from tiny_voca.text import PHONE_SET

RATE, HOP = 24000, 256
SPLITS = {
    "dev": ("M0004", "M0008", "M2001", "M0013"),
    "holdout": ("M0010", "M0012", "M0021", "M2005"),
}
# The French Tiny Voca inventory folds SAMPA H (IPA ɥ) into y.
SAMPA = dict(
    R="ʁ",
    S="ʃ",
    Z="ʒ",
    J="ɲ",
    N="ŋ",
    E="ɛ",
    O="ɔ",
    H="y",
    g="ɡ",
    **{"@": "ə", "2": "ø", "9": "œ", "a~": "ɑ̃", "o~": "ɔ̃", "e~": "ɛ̃", "9~": "œ̃"},
)
INTERVAL = re.compile(
    r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "((?:[^"]|"")*)"'
)


def tier(grid, name):
    for section in re.split(r"^\s*item \[\d+\]:\s*$", grid, flags=re.M):
        if f'name = "{name}"' in section:
            intervals = [
                (float(a), float(b), text.replace('""', '"'))
                for a, b, text in INTERVAL.findall(section)
            ]
            if not intervals or any(a >= b for a, b, _ in intervals):
                raise ValueError(f"Malformed {name} tier")
            return intervals
    raise ValueError(f"Missing {name} tier")


def windows(words, phones):
    """Build non-overlapping windows at whole-word cuts, independent of G2P."""
    spoken = [
        (i, a, b, label)
        for i, (a, b, label) in enumerate(words)
        if label not in ("_", "")
    ]
    candidates = []
    start = 0
    while start < len(spoken):
        first, a, _, _ = spoken[start]
        end = start
        while end + 1 < len(spoken) and spoken[end][2] - a < 3.8:
            end += 1
        last, _, b, _ = spoken[end]
        start = end + 1
        if not 2 <= b - a <= 8:
            continue
        block = [
            (start_time, end_time, label)
            for start_time, end_time, label in phones
            if end_time > a + 0.001 and start_time < b - 0.001
        ]
        canonical = [SAMPA.get(p, p) for _, _, p in block]
        if (
            not block
            or abs(block[0][0] - a) > 0.005
            or abs(block[-1][1] - b) > 0.005
            or any(p not in PHONE_SET and p != "_" for p in canonical)
            or sum(p != "_" for p in canonical) < 20
        ):
            continue
        label = " ".join(
            word for _, _, word in words[first : last + 1] if word not in ("_", "")
        )
        label = re.sub(r"(?<=') +(?=\w)", "", label).strip()
        if not label or any(mark in label for mark in ("%", "?", "~", "$", "<", ">")):
            continue
        # Include the available reference pause at either edge, up to 120 ms.
        left = a
        for start_time, end_time, phone in phones:
            if phone == "_" and abs(end_time - a) < 0.005:
                left = max(a - 0.12, start_time)
                break
        right = b
        for start_time, end_time, phone in phones:
            if phone == "_" and abs(start_time - b) < 0.005:
                right = min(b + 0.12, end_time)
                break
        reference = []
        oracle = ["<bos>"]
        prev_word = None
        for (start_time, end_time, raw), phone in zip(block, canonical):
            if phone == "_":
                continue
            middle = (start_time + end_time) / 2
            word = next((i for i, lo, hi, _ in spoken if lo <= middle < hi), None)
            if prev_word is not None and word != prev_word:
                oracle.append("<sp>")
            oracle.append(phone)
            prev_word = word
            reference.append(
                dict(
                    phone=phone,
                    raw=raw,
                    start=round(start_time - left, 6),
                    end=round(end_time - left, 6),
                )
            )
        oracle.append("<eos>")
        candidates.append((left, right, label, oracle, reference))
    if len(candidates) > 3:
        return [
            candidates[i] for i in np.linspace(0, len(candidates) - 1, 3, dtype=int)
        ]
    return candidates


def prepare(grids, audio, output, extension):
    manifests = []
    with zipfile.ZipFile(grids) as z:
        for split, files in SPLITS.items():
            for file in files:
                members = [
                    name
                    for name in z.namelist()
                    if name.endswith("/Rhap-" + file + "-Pro.TextGrid")
                ]
                if len(members) != 1:
                    raise ValueError(f"Missing grid for {file}")
                grid = z.read(members[0]).decode("utf-8-sig")
                source = audio / f"Rhap-{file}.{extension}"
                process = subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-i",
                        str(source),
                        "-ac",
                        "1",
                        "-ar",
                        str(RATE),
                        "-f",
                        "f32le",
                        "-",
                    ],
                    capture_output=True,
                    check=True,
                )
                wave = np.frombuffer(process.stdout, dtype="<f4")
                phones = tier(grid, "phone")
                if abs(len(wave) / RATE - phones[-1][1]) > 0.15:
                    raise ValueError(f"Audio and tier length disagree for {file}")
                chosen = windows(tier(grid, "word"), phones)
                if not chosen:
                    raise ValueError(f"No usable segment in {file}")
                sha = hashlib.sha256(source.read_bytes()).hexdigest()
                for index, (left, right, text, oracle, reference) in enumerate(
                    chosen, 1
                ):
                    name = f"{file}-{index}"
                    start, end = round(left * RATE), round(right * RATE)
                    clip = wave[start:end]
                    clip = np.pad(clip, (0, (-len(clip)) % HOP))
                    sf.write(output / (name + ".wav"), clip, RATE, subtype="FLOAT")
                    manifests.append(
                        dict(
                            id=name,
                            split=split,
                            recording=file,
                            file=name + ".wav",
                            text=text,
                            source=source.name,
                            source_sha256=sha,
                            start=left,
                            end=right,
                            oracle=oracle,
                            reference=reference,
                        )
                    )
    (output / "manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2) + "\n"
    )
    print(Counter(x["split"] for x in manifests))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("grids", type=Path)
    p.add_argument("audio", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--audio-extension", choices=("wav", "mp3"), default="wav")
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    prepare(args.grids, args.audio, args.output, args.audio_extension)
