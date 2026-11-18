"""Evaluate French G2P and CTC timing independently, without training."""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
from scipy.special import logsumexp

from tiny_voca.align import (
    ALIGNER_CENTER,
    ALIGNER_RATE,
    ALIGNER_STRIDE,
    AlignmentSettings,
    assemble,
    ctc_viterbi,
    frame_energy_db,
    silence_mask,
    spans_from_states,
)
from tiny_voca.text import (
    PHONE_SET,
    UNK,
    VOWEL_SET,
    espeak_version,
    frontend_fingerprint,
    frontend_identity,
    frontend_rules_fingerprint,
    phonemize,
)

RATE, HOP = 24000, 256
OFFSETS = (-0.050, -0.040, -0.030, -0.019, -0.010, 0.0, 0.010)
POLICIES = ("legacy", "contiguous", "bridge_one")
BASELINE = ("legacy", -0.019)
FIXED_CANDIDATE = ("legacy", -0.030)
WAV_FORMATS = {"WAV", "WAVEX", "RF64"}
FRICATIVES = frozenset(("f", "v", "s", "z", "ʃ", "ʒ"))
STOPS = frozenset(("p", "b", "t", "d", "k", "ɡ"))
NASAL_VOWELS = frozenset(("ɑ̃", "ɔ̃", "ɛ̃", "œ̃"))


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def edit_path(expected, actual):
    """Deterministic phone edit distance, with exact correspondences."""
    matrix = np.zeros((len(expected) + 1, len(actual) + 1), dtype=np.int32)
    matrix[0] = np.arange(len(actual) + 1)
    matrix[:, 0] = np.arange(len(expected) + 1)
    for i, phone in enumerate(expected, 1):
        for j, predicted in enumerate(actual, 1):
            matrix[i, j] = min(
                matrix[i - 1, j] + 1,
                matrix[i, j - 1] + 1,
                matrix[i - 1, j - 1] + (phone != predicted),
            )
    path = []
    i, j = len(expected), len(actual)
    while i or j:
        if (
            i
            and j
            and matrix[i, j]
            == matrix[i - 1, j - 1] + (expected[i - 1] != actual[j - 1])
        ):
            op = "equal" if expected[i - 1] == actual[j - 1] else "substitute"
            path.append((op, i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i and matrix[i, j] == matrix[i - 1, j] + 1:
            path.append(("delete", i - 1, None))
            i -= 1
        else:
            path.append(("insert", None, j - 1))
            j -= 1
    return path[::-1]


def protected_spans(spans, frames):
    first = np.asarray([a for a, _ in spans], dtype=np.float64)
    anchors = np.clip(
        np.round((first * ALIGNER_STRIDE + ALIGNER_CENTER) / ALIGNER_RATE * RATE / HOP),
        0,
        frames - 1,
    ).astype(np.int64)
    for i in range(1, len(anchors)):
        anchors[i] = max(anchors[i], anchors[i - 1] + 1)
    return anchors


def runs(mask):
    changes = np.flatnonzero(np.diff(np.pad(np.asarray(mask, dtype=np.int8), (1, 1))))
    return list(zip(changes[::2], changes[1::2]))


def pause_mask(silent, spans, policy, min_pause):
    """Keep the longest qualifying run per anchored gap, and edge-connected silence."""
    if policy == "legacy":
        return silent
    anchors = protected_spans(spans, len(silent))
    result = np.zeros_like(silent)
    first = runs(silent[: anchors[0]])
    if first and first[0][0] == 0:
        result[: first[0][1]] = True
    last_start = anchors[-1] + 1
    last = runs(silent[last_start:])
    if last and last[-1][1] == len(silent) - last_start:
        result[last_start + last[-1][0] :] = True
    for left, right in zip(anchors[:-1] + 1, anchors[1:]):
        mask = silent[left:right].copy()
        if policy == "bridge_one":
            mask[1:-1] |= mask[:-2] & mask[2:]
        choices = runs(mask)
        if choices:
            a, b = max(choices, key=lambda pair: pair[1] - pair[0])
            if b - a >= min_pause:
                result[left + a : left + b] = True
    return result


def align(logits, symbols, vocabulary, blank):
    phones = [s for s in symbols if s in PHONE_SET or s == UNK]
    labels = np.asarray([vocabulary[s] for s in phones], np.int64)
    repeats = sum(a == b for a, b in zip(labels[:-1], labels[1:]))
    if not len(labels) or len(logits) < len(labels) + repeats:
        raise ValueError("CTC sequence cannot fit")
    spans = spans_from_states(ctc_viterbi(logits, labels, blank), len(labels))
    probability = [
        float(np.exp(logits[a : b + 1, lab]).mean())
        for (a, b), lab in zip(spans, labels)
    ]
    return phones, spans, probability


def phone_group(phone):
    if phone in FRICATIVES:
        return "fricatives"
    if phone in STOPS:
        return "stops"
    if phone in NASAL_VOWELS:
        return "nasal_vowels"
    if phone == "ə":
        return "schwa"
    if phone in VOWEL_SET:
        return "oral_vowels"
    if phone in ("m", "n", "ɲ", "ŋ"):
        return "nasal_consonants"
    if phone in ("l", "ʁ", "j", "w"):
        return "liquids_glides"
    if phone in ("tʃ", "dʒ"):
        return "affricates"
    return "other"


def compare(symbols, durations, reference, correspondence):
    frames = int(sum(durations))
    ends = np.cumsum(durations)
    starts = ends - durations
    only_phones = [i for i, s in enumerate(symbols) if s in PHONE_SET or s == UNK]
    assert len(symbols) == len(durations) and all(durations[i] > 0 for i in only_phones)
    seconds = np.arange(frames) * HOP / RATE
    speech = np.zeros(frames, dtype=bool)
    for item in reference:
        speech |= (seconds >= item["start"]) & (seconds < item["end"])
    predicted = np.ones(frames, dtype=bool)
    for i in only_phones:
        predicted[starts[i] : ends[i]] = False
    reference_edge = (seconds < reference[0]["start"]) | (
        seconds >= reference[-1]["end"]
    )
    predicted_edge = (np.arange(frames) < starts[only_phones[0]]) | (
        np.arange(frames) >= ends[only_phones[-1]]
    )

    def pause_counts(guess, gold):
        return dict(
            tp=int(np.sum(guess & gold)),
            fp=int(np.sum(guess & ~gold)),
            fn=int(np.sum(~guess & gold)),
        )

    pause = pause_counts(predicted, ~speech)
    boundary = []
    for gold, guess in correspondence:
        i = only_phones[guess]
        item = reference[gold]
        a, b = starts[i] * HOP / RATE, ends[i] * HOP / RATE
        boundary.append(
            dict(
                phone=item["phone"],
                onset_ms=round(1000 * (a - item["start"]), 3),
                end_ms=round(1000 * (b - item["end"]), 3),
                duration_ms=round(1000 * ((b - a) - (item["end"] - item["start"])), 3),
                one_frame=bool(durations[i] == 1),
            )
        )
    return dict(
        boundaries=boundary,
        frames=frames,
        pause_tp=pause["tp"],
        pause_fp=pause["fp"],
        pause_fn=pause["fn"],
        pause_edge=pause_counts(predicted_edge, reference_edge),
        pause_internal=pause_counts(
            predicted & ~predicted_edge, ~speech & ~reference_edge
        ),
        phones_one_frame=int(np.sum(durations[only_phones] == 1)),
    )


def pause_stats(tp, fp, fn):
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        f1=(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else None,
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
    )


def overview(rows):
    boundaries = [q for r in rows for q in r["boundaries"]]

    def boundary_stats(values):
        x = np.asarray(values)
        if not x.size:
            return dict(count=0, mae_ms=None, p95_ms=None)
        return dict(
            count=int(x.size),
            bias_ms=float(x.mean()),
            mae_ms=float(np.abs(x).mean()),
            p95_ms=float(np.percentile(np.abs(x), 95)),
            within20=float(np.mean(np.abs(x) <= 20)),
            within40=float(np.mean(np.abs(x) <= 40)),
        )

    tp, fp, fn = (sum(x[k] for x in rows) for k in ("pause_tp", "pause_fp", "pause_fn"))
    out = dict(
        onset=boundary_stats([x["onset_ms"] for x in boundaries]),
        end=boundary_stats([x["end_ms"] for x in boundaries]),
        duration=boundary_stats([x["duration_ms"] for x in boundaries]),
        pause=pause_stats(tp, fp, fn),
        one_frame=sum(row["phones_one_frame"] for row in rows),
    )
    for scope in ("edge", "internal"):
        out["pause_" + scope] = pause_stats(
            *(
                sum(row["pause_" + scope][key] for row in rows)
                for key in ("tp", "fp", "fn")
            )
        )
    out["by_group"] = {
        name: {
            metric: boundary_stats(
                [
                    x[metric + "_ms"]
                    for x in boundaries
                    if phone_group(x["phone"]) == name
                ]
            )
            for metric in ("onset", "end", "duration")
        }
        for name in sorted(set(phone_group(x["phone"]) for x in boundaries))
    }
    return out


def paired_timing(baseline, candidate):
    """Candidate minus baseline MAE, resampling whole recordings."""
    by_recording = defaultdict(list)
    for first, second in zip(baseline, candidate, strict=True):
        if first["id"] != second["id"]:
            raise ValueError("Unpaired timing scores")
        for a, b in zip(first["boundaries"], second["boundaries"], strict=True):
            by_recording[first["recording"]].append(
                [abs(b[key]) - abs(a[key]) for key in ("onset_ms", "end_ms")]
            )
    groups = [
        (key, np.sum(rows, axis=0), len(rows))
        for key, rows in sorted(by_recording.items())
        if rows
    ]
    result = {
        name: dict(
            onset_ms=float(sums[0] / count), end_ms=float(sums[1] / count), phones=count
        )
        for name, sums, count in groups
    }
    ci = {}
    if len(groups) >= 2:
        sums = np.stack([values for _, values, _ in groups])
        counts = np.array([count for _, _, count in groups])
        rng = np.random.default_rng(0)
        indices = rng.integers(len(groups), size=(5000, len(groups)))
        sampled = sums[indices].sum(axis=1) / counts[indices].sum(axis=1)[:, None]
        ci = dict(
            zip(
                ("onset_ms", "end_ms"),
                np.percentile(sampled, [2.5, 97.5], axis=0).T.tolist(),
            )
        )
    return dict(by_recording=result, bootstrap95_ms=ci)


def probe(args):
    if frontend_identity()["voice"] not in ("fr", "fr-fr"):
        raise ValueError("This probe needs a French frontend")
    started = time.perf_counter()
    manifest_path = args.corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = digest(manifest_path)
    target = (
        json.loads((args.corpus / "qualification.json").read_text())
        if args.target
        else None
    )
    if args.target and (
        target.get("kind") != "independent_teacher_wavs"
        or {row.get("speaker") for row in manifest} != {"F1", "M1"}
        or {row["split"] for row in manifest} != {"holdout"}
        or target.get("manifest_sha256") != manifest_sha
    ):
        raise ValueError("Target comparison requires the prepared F1/M1 holdout")
    variants = (
        (BASELINE, FIXED_CANDIDATE)
        if target
        else tuple((policy, offset) for policy in POLICIES for offset in OFFSETS)
    )
    vocabulary = json.loads(args.vocab.read_text())
    if not PHONE_SET <= vocabulary.keys():
        raise ValueError(
            f"Unsupported aligner vocabulary: {PHONE_SET - vocabulary.keys()}"
        )
    model_sha = digest(args.model)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    runtime = None
    corpus = defaultdict(list)
    edit_counts = defaultdict(Counter)
    edit_pairs = defaultdict(Counter)
    raw_edits = []
    inference = []
    args.cache.mkdir(exist_ok=True, parents=True)
    for row in manifest:
        wave_path = args.corpus / row["file"]
        wave_sha = digest(wave_path)
        if (
            row.get("source_format") != "original_wav"
            or row.get("fixture_sha256") != wave_sha
            or sf.info(wave_path).format not in WAV_FORMATS
        ):
            raise ValueError(f"Invalid WAV provenance for {row['id']}")
        wave, rate = sf.read(wave_path, dtype="float32")
        if (
            rate != RATE
            or wave.ndim != 1
            or len(wave) % HOP
            or not np.isfinite(wave).all()
        ):
            raise ValueError(f"Invalid audio fixture {row['id']}")
        reference = row["reference"]
        expected = [x["phone"] for x in reference]
        if list(filter(PHONE_SET.__contains__, row["oracle"])) != expected:
            raise ValueError("Oracle sequence differs from reference intervals")
        cached = args.cache / (wave_sha + "-" + model_sha + ".npz")
        cache_hit = cached.exists()
        elapsed = None
        if cache_hit:
            with np.load(cached) as saved:
                logits = saved["log_probs"]
        else:
            if runtime is None:
                runtime = ort.InferenceSession(
                    str(args.model), options, providers=["CPUExecutionProvider"]
                )
            samples = resample_poly(wave.astype(np.float64), 2, 3)
            audio = ((samples - samples.mean()) / np.sqrt(samples.var() + 1e-7)).astype(
                np.float32
            )[None]
            mark = time.perf_counter()
            logits = runtime.run(None, {runtime.get_inputs()[0].name: audio})[0][0]
            elapsed = time.perf_counter() - mark
            logits -= logsumexp(logits, axis=1, keepdims=True)
            np.savez_compressed(cached, log_probs=logits)
        if not np.isfinite(logits).all() or logits.shape[1] <= max(vocabulary.values()):
            raise ValueError("Unexpected CTC output")
        inference.append(dict(id=row["id"], seconds=elapsed, cache_hit=cache_hit))
        g2p = phonemize(row["text"])
        predicted = [s for s in g2p.symbols if s in PHONE_SET or s == UNK]
        path = edit_path(expected, predicted)
        split = row["split"]
        equal = [(i, j) for op, i, j in path if op == "equal"]
        edits = [
            (
                op,
                expected[i] if i is not None else "∅",
                predicted[j] if j is not None else "∅",
            )
            for op, i, j in path
            if op != "equal"
        ]
        edit_counts[split].update(op for op, _, _ in path)
        edit_pairs[split].update((a, b) for _, a, b in edits)
        edit_counts[split]["reference_phones"] += len(expected)
        edit_counts[split]["unknown"] += len(g2p.unknown)
        raw_edits.append(
            dict(
                id=row["id"],
                split=split,
                recording=row["recording"],
                reference_phones=len(expected),
                edits=edits,
                unknown=list(g2p.unknown),
                starts_at=row["start"],
                ends_at=row["end"],
                source_sha256=row["source_sha256"],
                fixture_sha256=wave_sha,
                text_revision=row.get("text_revision"),
            )
        )
        level = frame_energy_db(wave, HOP)
        silent, _ = silence_mask(level, AlignmentSettings())
        for mode, symbols, matches in (
            (
                "oracle",
                row["oracle"],
                list(zip(range(len(expected)), range(len(expected)))),
            ),
            ("g2p", g2p.symbols, equal),
        ):
            _, spans, prob = align(logits, symbols, vocabulary, vocabulary["<pad>"])
            masks = {
                policy: pause_mask(
                    silent, spans, policy, AlignmentSettings.min_pause_frames
                )
                for policy in dict.fromkeys(policy for policy, _ in variants)
            }
            for policy, offset in variants:
                settings = replace(AlignmentSettings(), onset_offset_seconds=offset)
                durations = assemble(
                    symbols, spans, len(level), masks[policy], HOP, RATE, settings
                )
                scored = compare(symbols, durations, reference, matches)
                corpus[split, mode, policy, offset].append(
                    dict(
                        id=row["id"],
                        recording=row["recording"],
                        speaker=row.get("speaker"),
                        ctc_mean=float(np.mean(prob)),
                        ctc_weak=float(np.mean(np.asarray(prob) < 0.1)),
                        **scored,
                    )
                )
        timing = "cached" if cache_hit else f"{elapsed:.2f}s model"
        print(row["id"], split, len(expected), len(edits), timing, flush=True)
    summaries = {key: overview(rows) for key, rows in corpus.items()}
    candidate = None
    chosen = (BASELINE, FIXED_CANDIDATE)
    if not target:
        best_offset = min(
            OFFSETS,
            key=lambda off: summaries["dev", "oracle", "legacy", off]["onset"][
                "mae_ms"
            ],
        )
        best_policy = max(
            POLICIES,
            key=lambda pol: (
                summaries["dev", "oracle", pol, BASELINE[1]]["pause"]["f1"] or 0.0
            ),
        )
        candidate = (best_policy, best_offset)
        chosen = (
            *chosen,
            ("contiguous", BASELINE[1]),
            ("bridge_one", BASELINE[1]),
            ("legacy", best_offset),
            candidate,
        )

    timing = {}
    for mode in ("oracle", "g2p"):
        a = corpus[("holdout", mode, *BASELINE)]
        b = corpus[("holdout", mode, *FIXED_CANDIDATE)]
        timing[f"holdout/{mode}/all"] = paired_timing(a, b)
        for speaker in sorted({row["speaker"] for row in a if row["speaker"]}):
            before = [row for row in a if row["speaker"] == speaker]
            after = [row for row in b if row["speaker"] == speaker]
            timing[f"holdout/{mode}/{speaker}"] = {
                "baseline": overview(before),
                "candidate": overview(after),
                "paired": paired_timing(before, after),
            }

    def present(key):
        return "/".join([*key[:-1], f"{key[-1]:+.3f}"])

    report = dict(
        model_sha256=model_sha,
        manifest_sha256=manifest_sha,
        vocab_sha256=digest(args.vocab),
        ort=ort.__version__,
        threads=args.threads,
        text_frontend=frontend_fingerprint(),
        frontend_rules=frontend_rules_fingerprint(),
        espeak=espeak_version(),
        wall_seconds=time.perf_counter() - started,
        best_dev=list(candidate) if candidate else None,
        source=target or "Rhapsodie, published phone/word tiers",
        fixed_comparison=timing,
        edit_counts=dict(edit_counts),
        edit_pairs={
            split: [
                dict(reference=a, predicted=b, count=n)
                for (a, b), n in pairs.most_common()
            ]
            for split, pairs in edit_pairs.items()
        },
        g2p_edits=raw_edits,
        inference=inference,
        summaries={present(key): v for key, v in summaries.items()},
        clips={present(key): rows for key, rows in corpus.items() if key[2:] in chosen},
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("corpus", "model", "vocab", "cache", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--target",
        action="store_true",
        help="Only compare the two fixed offsets on the F1/M1 holdout",
    )
    probe(p.parse_args())
