"""Package WAV pairs for blinded listening. Keep the decoding key separate."""

import argparse
import html
import json
import random
import secrets
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from evaluate import cases


def prepare(wavs, output, key, variants, fixture=None):
    if fixture:
        data = json.loads(fixture.read_text())
        if len(variants) != 1 or len(data) != 24:
            raise ValueError(
                "The release listen uses exactly 24 pairs and one candidate"
            )
        rng = secrets.SystemRandom()
        study = secrets.token_hex(12)
        pool = []
        for voice in (0, 1):
            group = [case for case in data if case["voice"] == voice]
            if len(group) != 12 or len({case["id"] for case in group}) != 12:
                raise ValueError("Expected 12 unique cases for each voice")
            rng.shuffle(group)
            pool.extend(
                (variants[0], case["id"], i % 2) for i, case in enumerate(group)
            )
    else:
        data = list(cases("holdout"))
        rng = random.Random(563)
        study = None
        ids = sorted(
            path.stem.removeprefix("original-") for path in wavs.glob("original-*.wav")
        )
        pool = [(variant, case, None) for variant in variants for case in ids]
    samples = {case["id"]: case for case in data}
    rng.shuffle(pool)
    answers, cards = {}, []
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for index, (variant, case, order) in enumerate(pool, 1):
            label = f"{index:02d}"
            assigned = ["original", variant]
            if order is None:
                rng.shuffle(assigned)
            elif order:
                assigned.reverse()
            item = samples[case]
            answers[label] = dict(zip(("A", "B"), assigned), case=case)
            if study:
                answers[label].update(voice=item["voice"], group=item["group"])
            for side, model in zip(("A", "B"), assigned):
                archive.write(wavs / f"{model}-{case}.wav", f"audio/{label}-{side}.wav")
            defect = ""
            if study:
                defect = (
                    "<fieldset><legend>Major audible issue</legend>"
                    + "".join(
                        f'<label><input required type="radio" name="defect-{label}" value="{side}">{description}</label>'
                        for side, description in (
                            ("A", "A only"),
                            ("B", "B only"),
                            ("both", "Both"),
                            ("none", "Neither"),
                        )
                    )
                    + "</fieldset>"
                )
            cards.append(
                f"<section><h2>{label} · {html.escape(item['text'])}</h2>"
                + "".join(
                    f'<label>{side}<audio controls preload="none" src="audio/{label}-{side}.wav"></audio></label>'
                    for side in ("A", "B")
                )
                + "".join(
                    f'<label><input {"required " if study else ""}type="radio" name="{label}" value="{v}">{s}</label>'
                    for v, s in (
                        ("A", "Prefer A"),
                        ("B", "Prefer B"),
                        ("same", "No preference"),
                    )
                )
                + defect
                + f'<textarea name="note-{label}" placeholder="Pronunciation, rhythm, timbre or artifacts"></textarea></section>'
            )
        page = (
            """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Tiny Voca listening test</title>
<style>body{max-width:820px;margin:35px auto;padding:0 16px;font:16px system-ui}
section{border-top:1px solid #bbb;padding:12px 0}h2{font-size:1rem}
label{display:inline-flex;align-items:center;gap:6px;margin:4px 10px}
textarea{width:100%;height:2.6em;margin:6px 0}button{margin:16px 0;padding:8px 12px}</style>
<h1>Tiny Voca listening test</h1>
<p>Listen to both clips with the same headphones and volume. Mark major differences in pronunciation, rhythm, timbre or audio artifacts, then record a preference.</p>
<form id="rating">"""
            + (f'<input type="hidden" name="study" value="{study}">' if study else "")
            + "".join(cards)
            + """<button type="submit">Download responses</button></form>
<script>rating.onsubmit=(event)=>{event.preventDefault();const data=new Blob([JSON.stringify(Object.fromEntries(new FormData(rating)),null,2)],{type:'application/json'});
const link=document.createElement('a');link.href=URL.createObjectURL(data);link.download='responses.json';link.click();}</script>"""
        )
        archive.writestr("index.html", page)
    if study:
        # Earlier listeners may still be scoring a kit with this key.
        with key.open("x", encoding="utf-8") as stream:
            json.dump({"study": study, "pairs": answers}, stream, indent=2)
            stream.write("\n")
    else:
        key.write_text(json.dumps(answers, indent=2) + "\n")


def score_key(key, responses):
    if key["study"] != responses.get("study"):
        raise ValueError("Response belongs to a different listening kit")
    totals = Counter()
    by_voice = {0: Counter(), 1: Counter()}
    issues = []
    for label, record in key["pairs"].items():
        preference = responses.get(label)
        defect = responses.get(f"defect-{label}")
        if preference not in ("A", "B", "same") or defect not in (
            "A",
            "B",
            "both",
            "none",
        ):
            raise ValueError(f"Invalid or missing rating for pair {label}")
        name = record[preference] if preference != "same" else "same"
        prefix = f"preference:{name}"
        problem = record[defect] if defect in ("A", "B") else defect
        for counter in (totals, by_voice[record["voice"]]):
            counter[prefix] += 1
            counter[f"major_issue:{problem}"] += 1
        if defect != "none":
            issues.append(
                {
                    "case": record["case"],
                    "major_issue": problem,
                    "note": responses.get(f"note-{label}", ""),
                }
            )
    return {
        "pairs": len(key["pairs"]),
        "totals": totals,
        "by_voice": by_voice,
        "issues": issues,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("wavs", "output", "fixture", "score"):
        parser.add_argument("--" + field, type=Path)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("variants", nargs="*")
    args = parser.parse_args()
    if args.score:
        print(
            json.dumps(
                score_key(
                    json.loads(args.key.read_text()), json.loads(args.score.read_text())
                ),
                indent=2,
            )
        )
    elif args.wavs and args.output and args.variants:
        if args.fixture and (args.key.exists() or args.output.exists()):
            parser.error("Choose new paths for the kit and key")
        prepare(args.wavs, args.output, args.key, args.variants, args.fixture)
    else:
        parser.error("Provide --score or --wavs, --output and candidate variants")
