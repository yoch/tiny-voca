"""Reconstruct and check the 38 reviewed phone edits on the original WAV fixture."""

import argparse
import gzip
import json
from pathlib import Path

from alignment_probe import digest, edit_path

from tiny_voca.text import PHONE_SET, UNK, frontend_fingerprint, phonemize


def cases(corpus, historical):
    rows = json.loads((corpus / "manifest.json").read_text())
    with gzip.open(historical, "rt", encoding="utf-8") as stream:
        old = {
            row["id"]: row
            for row in json.load(stream)["g2p_edits"]
            if row["split"] == "holdout"
        }
    if {row["id"] for row in rows if row["split"] == "holdout"} != old.keys():
        raise ValueError("Holdout IDs differ from the historical result")
    result = []
    for row in rows:
        if row["split"] != "holdout":
            continue
        clip = row["id"]
        if (
            row.get("source_format") != "original_wav"
            or row["source_sha256"] != old[clip]["source_sha256"]
            or digest(corpus / row["file"]) != row["fixture_sha256"]
        ):
            raise ValueError(f"Historical WAV provenance mismatch for {clip}")
        reference = row["reference"]
        gold = [p["phone"] for p in reference]
        predicted = [
            s for s in phonemize(row["text"]).symbols if s in PHONE_SET or s == UNK
        ]
        previous = 0
        edits = []
        for op, i, j in edit_path(gold, predicted):
            if op != "equal":
                left = reference[previous - 1] if previous else None
                right = reference[previous] if previous < len(reference) else None
                a = gold[i] if i is not None else "∅"
                b = predicted[j] if j is not None else "∅"
                edits.append([op, a, b])
                result.append(
                    {
                        "id": f"{clip}:{len(edits)}",
                        "edit": [op, a, b],
                        "words": [
                            left["word"] if left else None,
                            right["word"] if right else None,
                        ],
                        "wav_anchor_ms": round(
                            1000 * (row["start"] + (right or left)["start"]), 3
                        ),
                        "left_gap_ms": round(1000 * (right["start"] - left["end"]), 3)
                        if left and right
                        else None,
                    }
                )
            previous += i is not None
        if edits != old[clip]["edits"]:
            raise ValueError(
                f"E01 phone edits differ from the historical result for {clip}"
            )
    return result


def validate_review(observed, review):
    indexed = {row["id"]: row for row in review}
    if len(indexed) != len(review) or indexed.keys() != {row["id"] for row in observed}:
        raise ValueError("Review must cover every edit once")
    for case in observed:
        record = indexed[case["id"]]
        if any(record[key] != case[key] for key in case):
            raise ValueError(f"Stale review for {case['id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    observed = cases(args.corpus, args.historical)
    if args.review:
        validate_review(observed, json.loads(args.review.read_text(encoding="utf-8")))
    print(
        json.dumps(
            {"frontend": frontend_fingerprint(), "cases": observed},
            ensure_ascii=False,
            indent=2,
        )
    )
