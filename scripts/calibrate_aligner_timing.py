"""Measure where the recogniser's CTC emissions sit inside a phone. Cloud only.

For phone transitions that must flip the pitch tracker's voicing (vowel ->
voiceless consonant, voiceless consonant -> vowel), compare the time of the
flip with the START of the next phone's first emission frame. On this corpus
the flip precedes that frame start by about 10 ms (vowel -> fricative) and
27 ms (fricative -> vowel); the pitch tracker's analysis window widens voiced
regions by roughly one frame on each side, so the two cases bracket the true
onset: about 19 ms before the frame start. That constant is
``AlignmentSettings.onset_offset_seconds``. Re-run this script before reusing
the aligner on a different recogniser, hop size or pitch tracker.
"""

import argparse
import collections
import json
from pathlib import Path
import numpy as np
import torch

from tiny_voca.align import ALIGNER_RATE, ALIGNER_STRIDE, PhoneRecognizer, token_spans
from tiny_voca.config import Config
from tiny_voca.text import PHONE_SET, VOWEL_SET, phonemize

FRICATIVES, STOPS = {"f", "s", "ʃ"}, {"p", "t", "k"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    c = Config()
    recognizer = PhoneRecognizer(args.device)
    rows = [json.loads(line) for line in (args.features / (args.split + ".jsonl")).read_text(encoding="utf-8").split("\n") if line.strip()][:args.count]
    offsets, runs = collections.defaultdict(list), collections.Counter()
    for row in rows:
        stored = torch.load(args.features / row["path"], map_location="cpu", weights_only=True)
        voiced = stored["prosody"][:, 1].numpy() > 0.5
        symbols = phonemize(row["text"]).symbols
        position = [i for i, s in enumerate(symbols) if s in PHONE_SET]
        names = [symbols[i] for i in position]
        log_probs = recognizer.log_probs(stored["wave"].numpy(), c.sample_rate)
        spans = token_spans(log_probs, [recognizer.label(s) for s in names], recognizer.blank)
        runs.update(last - first + 1 for first, last in spans)
        flips = [(f, bool(voiced[f - 1])) for f in range(1, len(voiced)) if voiced[f - 1] != voiced[f]]
        for k in range(len(names) - 1):
            if position[k + 1] != position[k] + 1:
                continue                                    # a pause-capable token lies between them
            a, b = names[k], names[k + 1]
            if a in VOWEL_SET and b in FRICATIVES:
                kind, before = "vowel>fricative", True
            elif a in VOWEL_SET and b in STOPS:
                kind, before = "vowel>stop", True
            elif a in FRICATIVES and b in VOWEL_SET:
                kind, before = "fricative>vowel", False
            elif a in STOPS and b in VOWEL_SET:
                kind, before = "stop>vowel", False
            else:
                continue
            emission = spans[k + 1][0] * ALIGNER_STRIDE / ALIGNER_RATE
            times = [(f - 0.5) * c.hop / c.sample_rate - emission for f, was_voiced in flips if was_voiced == before]
            near = [t for t in times if abs(t) < 0.15]
            offsets[(row["speaker"], kind)].append(min(near, key=abs) if near else None)
    report = {"utterances": len(rows), "emission_run_lengths": dict(sorted(runs.items())), "transitions": {}}
    for (voice, kind), values in sorted(offsets.items()):
        found = np.array([v for v in values if v is not None]) * 1000
        report["transitions"][f"{voice} {kind}"] = {
            "count": len(values), "flip_found": round(len(found) / len(values), 3), "median_ms": round(float(np.median(found)), 1),
            "mean_ms": round(float(found.mean()), 1), "std_ms": round(float(found.std()), 1),
            "within_11ms_of_median": round(float((np.abs(found - np.median(found)) <= 11).mean()), 3),
            "within_22ms_of_median": round(float((np.abs(found - np.median(found)) <= 22).mean()), 3)}
    clean = [report["transitions"][f"{v} {k}"]["median_ms"] for v in c.speakers for k in ("vowel>fricative", "fricative>vowel")
             if f"{v} {k}" in report["transitions"]]
    report["suggested_onset_offset_seconds"] = round(float(np.mean(clean)) / 1000, 4) if clean else None
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
