"""Package WAV pairs for blinded listening. Keep the decoding key separate."""

import argparse
import html
import json
import random
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from evaluate import cases


def prepare(wavs, output, key, variants):
    samples = {case["id"]: case["text"] for case in cases("holdout")}
    ids = sorted(
        path.stem.removeprefix("original-") for path in wavs.glob("original-*.wav")
    )
    pool = [(variant, case) for variant in variants for case in ids]
    rng = random.Random(563)
    rng.shuffle(pool)
    answers, cards = {}, []
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for index, (variant, case) in enumerate(pool, 1):
            label = f"{index:02d}"
            assigned = ["original", variant]
            rng.shuffle(assigned)
            answers[label] = dict(zip(("A", "B"), assigned), case=case)
            for side, model in zip(("A", "B"), assigned):
                archive.write(wavs / f"{model}-{case}.wav", f"audio/{label}-{side}.wav")
            cards.append(
                f"<section><h2>{label} · {html.escape(samples[case])}</h2>"
                + "".join(
                    f'<label>{side}<audio controls preload="none" src="audio/{label}-{side}.wav"></audio></label>'
                    for side in ("A", "B")
                )
                + "".join(
                    f'<label><input type="radio" name="{label}" value="{v}">{s}</label>'
                    for v, s in (
                        ("A", "Prefer A"),
                        ("B", "Prefer B"),
                        ("same", "Indistinguishable"),
                    )
                )
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
<p>Listen to both clips at the same volume. Mark any audible difference in pronunciation, rhythm or timbre and record a preference if you have one.</p>
<form id="rating">"""
            + "".join(cards)
            + """<button type="submit">Download responses</button></form>
<script>rating.onsubmit=(event)=>{event.preventDefault();const data=new Blob([JSON.stringify(Object.fromEntries(new FormData(rating)),null,2)],{type:'application/json'});
const link=document.createElement('a');link.href=URL.createObjectURL(data);link.download='responses.json';link.click();}</script>"""
        )
        archive.writestr("index.html", page)
    key.write_text(json.dumps(answers, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("wavs", "output", "key"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("variants", nargs="+")
    args = parser.parse_args()
    prepare(args.wavs, args.output, args.key, args.variants)
