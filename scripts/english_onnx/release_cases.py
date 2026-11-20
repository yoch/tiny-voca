"""Encode the frozen independent release prompts and choose the timing cases."""

import argparse
import json
from pathlib import Path

from evaluate import DOCS, RATE, cases, feeds, render, session

from tiny_voca.text import phonemize


def prepare(public, output):
    previous = {
        case["text"].casefold()
        for split in ("calibration", "holdout")
        for case in cases(split)
    }
    prompts = json.loads((DOCS / "release-texts.json").read_text())
    if len(prompts) != len({row["text"].casefold() for row in prompts}):
        raise ValueError("Duplicate release prompts")
    reference = session(public)
    out = []
    for item in prompts:
        if item["text"].casefold() in previous:
            raise ValueError("Release prompt overlaps an existing set")
        phones = phonemize(item["text"])
        for voice, name in enumerate(("F2", "M2")):
            case = {
                "text": item["text"],
                "tokens": list(phones.ids),
                "stress": list(phones.stress),
                "voice": voice,
                "id": f"{item['id']}-{name}",
                "group": item["group"],
                "speed": item.get("speed", 1.0),
                "pitch_shift": item.get("pitch_shift", 0.0),
                "pitch_range": item.get("pitch_range", 1.0),
            }
            audio, _ = render(reference, feeds(case))
            out.append(
                {**case, "label": case["id"], "reference_seconds": len(audio) / RATE}
            )
    output.mkdir(parents=True, exist_ok=True)
    sets = {
        "release": out,
        "short": [r for r in out if r["group"] == "short"],
        "ola": [r for r in out if r["id"].split("-")[0] in ("cons2", "long2")],
    }
    for name, rows in sets.items():
        (output / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("public", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    prepare(args.public, args.output)
