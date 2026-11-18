"""Summarize paired English quality and counterbalanced performance records."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from jiwer import process_characters, process_words
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer


def errors(normalize, text, result):
    expected, prediction = normalize(text), normalize(result or "")
    stats = []
    for metric in (process_words, process_characters):
        outcome = metric(expected, prediction)
        stats.append(
            [
                outcome.substitutions + outcome.insertions + outcome.deletions,
                outcome.hits + outcome.substitutions + outcome.deletions,
            ]
        )
    return np.asarray(stats)


def uncertainty(rows):
    """Bootstrap paired errors by authored sentence (both voices together)."""
    groups = defaultdict(list)
    for sentence, a, b in rows:
        groups[sentence].append((b[:, 0] - a[:, 0], a[:, 1]))
    counts = np.array([np.sum(values, axis=0) for values in groups.values()])
    mean = 100 * counts[:, 0].sum(axis=0) / counts[:, 1].sum(axis=0)
    draws = np.random.default_rng(9).integers(len(counts), size=(10000, len(counts)))
    replicas = counts[draws].sum(axis=1)
    ratios = 100 * replicas[:, 0] / replicas[:, 1]
    return mean.tolist(), np.percentile(ratios, [2.5, 97.5], axis=0).T.tolist()


def quality(report):
    normalize = EnglishTextNormalizer({})
    exact = defaultdict(list)
    paired = defaultdict(list)
    for row in report["cases"]:
        sample = row["case"]
        ref = row["reference_transcript"]
        baseline = errors(normalize, sample["text"], ref) if ref is not None else None
        for name, candidate in row["metrics"].items():
            for label in ("all", sample["id"].split("-")[-1]):
                exact[name, label].append(candidate)
                if baseline is not None and "transcript" in candidate:
                    trial = errors(normalize, sample["text"], candidate["transcript"])
                    paired[name, label].append((sample["text"], baseline, trial))
    result = {}
    for (name, label), rows in exact.items():
        summary = dict(
            cases=len(rows),
            duration_equal=sum(r["duration_equal"] for r in rows),
            length_equal=sum(r["original_samples"] == r["samples"] for r in rows),
            bit_exact=sum(r.get("sample_exact", False) for r in rows),
        )
        for metric in (
            "mel_rms_db",
            "f0_mae_semitones",
            "voicing_disagreement",
            "speaker_cosine",
            "snr_db",
            "normalized_peak",
            "duration_mae_frames",
            "pause_mae_frames",
        ):
            values = [r.get(metric) for r in rows if r.get(metric) is not None]
            if values:
                summary[metric] = dict(
                    zip(
                        ("min", "p05", "median", "p95", "max"),
                        np.percentile(values, [0, 5, 50, 95, 100]).tolist(),
                    )
                )
        if (name, label) in paired:
            items = paired[name, label]
            base, trial = (np.sum([row[i] for row in items], axis=0) for i in (1, 2))
            delta, ci = uncertainty(items)
            for j, metric in enumerate(("wer", "cer")):
                summary[metric] = {
                    "baseline": float(100 * base[j, 0] / base[j, 1]),
                    "candidate": float(100 * trial[j, 0] / trial[j, 1]),
                    "delta_points": delta[j],
                    "delta_ci95_points": ci[j],
                    "baseline_errors": int(base[j, 0]),
                    "candidate_errors": int(trial[j, 0]),
                    "units": int(base[j, 1]),
                }
        result.setdefault(name, {})[label] = summary
    return result


def speed(records):
    result = {}
    for record in records:
        name = record.get("graph", record.get("model"))
        row = result.setdefault(name, {}).setdefault(str(record["threads"]), {})
        if "error" in record:
            row["error"] = record["error"]
            continue
        order = record.get("order", "forward")
        measurements = record.get("cases", record.get("models"))
        cases = {}
        for sample in measurements:
            values = [r["total_ms"] for r in sample["times"]]
            audio = sample["times"][0]["samples"] / 24000
            p50, p95 = np.percentile(values, [50, 95]).tolist()
            cases[sample["case"]] = dict(
                median_ms=p50,
                p95_ms=p95,
                rtf=p50 / (1000 * audio),
                first_ms=sample["first"]["total_ms"],
                audio_seconds=audio,
            )
        row[order] = dict(
            cases=cases,
            load_ms=record["load_ms"],
            rss_peak_bytes=record["rss_peak_bytes"],
        )
    return result


def collect_cpu(directory):
    data = []
    for path in sorted(directory.glob("*.json")):
        row = json.loads(path.read_text())
        row["order"] = path.stem.split("-", 1)[0]
        data.append(row)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quality", type=Path, nargs="+")
    parser.add_argument("--cpu", type=Path)
    parser.add_argument("--web", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for path in args.quality or ():
        result[path.stem] = quality(json.loads(path.read_text()))
    if args.cpu:
        result["cpu"] = speed(collect_cpu(args.cpu))
    if args.web:
        result["web"] = speed(json.loads(args.web.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
