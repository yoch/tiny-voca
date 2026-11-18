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
from scipy.io import wavfile

from tiny_voca.text import PHONE_SET

RATE, HOP = 24000, 256
SPLITS = {
    "dev": ("M0004", "M0008", "M2001", "M0013"),
    "holdout": ("M0010", "M0012", "M0021", "M2005"),
}
WAV_SHA256 = {
    "M0004": "79831c5f45dc308d1ab3b1d31b6ac904fc7aa3a8075ee9d983c839d0af3a074d",
    "M0008": "655337536ad9ddf5196bb7339c01984a6116416e286b2dc727b0176ddfbee452",
    "M0010": "77e53f889368f0263b38c375178cda5004f6e9d7aec36db3dfaf18897e0174e5",
    "M0012": "5d20aa666effef6d17acbf091500a53baa6e163e7067612eb5237f24d2ae45bc",
    "M0013": "ec23d59c5a6da063dedc73efc95ed4a9e086828c50ee8665ab1f1451a24a7f65",
    "M0021": "66096cfa02b3519bd5e2c5492c919f02091a3a0de1bda0a580839633513ae5ab",
    "M2001": "1cd137b3c162bd84c2bf203a61e38f62deb96a35b515daa831f89f6fdbc41659",
    "M2005": "116817a2ef9f202819015cd1fe66dd338fe9d83016c6a5026bbae8060d13f686",
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
                    word=words[word][2] if word is not None else None,
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


def apply_reviewed_text(text, identifier, text_edits):
    edit = text_edits.get(identifier)
    if edit is None:
        return text
    if (
        hashlib.sha256(text.encode()).hexdigest() != edit["text_sha256"]
        or text.count(edit["before"]) != 1
    ):
        raise ValueError(f"Stale reviewed text for {identifier}")
    return text.replace(edit["before"], edit["after"], 1)


def prepare(grids, audio, output, reviewed=None):
    manifests = []
    text_edits = reviewed["edits"] if reviewed else {}
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
                source = audio / f"Rhap-{file}.wav"
                sha = hashlib.sha256(source.read_bytes()).hexdigest()
                if sha != WAV_SHA256[file]:
                    raise ValueError(f"Expected original WAV for {file}")
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
                for index, (left, right, text, oracle, reference) in enumerate(
                    chosen, 1
                ):
                    name = f"{file}-{index}"
                    text = apply_reviewed_text(text, name, text_edits)
                    start, end = round(left * RATE), round(right * RATE)
                    clip = wave[start:end]
                    clip = np.pad(clip, (0, (-len(clip)) % HOP))
                    wavfile.write(output / (name + ".wav"), RATE, clip)
                    manifests.append(
                        dict(
                            id=name,
                            split=split,
                            recording=file,
                            file=name + ".wav",
                            text=text,
                            source=source.name,
                            source_format="original_wav",
                            source_sha256=sha,
                            text_revision=reviewed["revision"]
                            if reviewed
                            else "rhapsodie-word-tier",
                            fixture_sha256=hashlib.sha256(
                                (output / (name + ".wav")).read_bytes()
                            ).hexdigest(),
                            start=left,
                            end=right,
                            oracle=oracle,
                            reference=reference,
                        )
                    )
    if text_edits.keys() - {item["id"] for item in manifests}:
        raise ValueError("A reviewed text edit does not belong to the WAV fixture")
    (output / "manifest.json").write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2) + "\n"
    )
    print(Counter(x["split"] for x in manifests))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("grids", type=Path)
    p.add_argument("audio", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--reviewed-text",
        type=Path,
        help="Apply a versioned correction to the fixture text",
    )
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    reviewed = (
        json.loads(args.reviewed_text.read_text(encoding="utf-8"))
        if args.reviewed_text
        else None
    )
    prepare(args.grids, args.audio, args.output, reviewed)
