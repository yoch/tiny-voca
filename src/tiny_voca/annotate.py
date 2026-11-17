"""Versioned annotation layer: phoneme tokens and validated durations.

Cloud only. Reads the existing acoustic feature files (wave, mel, pitch); their
byte tokens are stale and ignored. Writes a NEW directory and never modifies the
feature files or the source dataset:

    <output>/annotations.json           identity, completion flag, QC summary
    <output>/phonemes.jsonl             front-end output per distinct text
    <output>/raw/<split>.jsonl          resumable journal of CTC alignments
    <output>/refined-<n>/<split>.jsonl  optional boundary refinement passes
    <output>/<split>.jsonl              accepted rows used by training
    <output>/<split>.rejected.jsonl     rejected rows with reasons

Re-running with ``--resume`` skips finished work. Any change of front end,
aligner, settings or source manifests changes the identity and requires a new
output directory, which is how stale tokens/durations are invalidated.
"""

import argparse
from collections import Counter, defaultdict
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from .align import ALIGNER_MODEL, AlignmentSettings, PhoneRecognizer, align_utterance
from .config import ANNOTATION_VERSION, Config
from .journal import append, atomic_json, file_hash, recover
from .text import (PHONE_SET, SYMBOL_TO_ID, UNK, VOICELESS, VOWEL_SET, Phonemized, frontend_fingerprint,
                   frontend_identity, frontend_runtime, phonemize)

SPLITS = ("train", "validation", "test")
FRICATIVES = frozenset(("f", "s", "ʃ"))


def load_acoustic_rows(features, split, config):
    metadata = features / (split + ".preparation.json")
    if not metadata.exists() or not json.loads(metadata.read_text()).get("complete"):
        raise ValueError(f"acoustic preparation for {split} is missing or incomplete")
    rows = [json.loads(line) for line in (features / (split + ".jsonl")).read_text(encoding="utf-8").split("\n") if line.strip()]
    for row in rows:
        # Current acoustic-only files, or files from the byte-token preparation:
        # same audio/DSP settings, stale tokens that are ignored here.
        if row.get("acoustic") != config.acoustic_fingerprint and row.get("features") != config.legacy_feature_fingerprint:
            raise ValueError("feature files were not produced with compatible audio/DSP settings; rerun acoustic preparation")
        if row["speaker"] not in config.speakers:
            raise ValueError(f"unknown speaker {row['speaker']!r}")
    return rows


def text_key(text):
    return hashlib.sha256(text.strip().encode()).hexdigest()


def phonemize_texts(rows, cache, workers=16, log=print):
    known = {}
    for record in recover(cache):
        known[record["text_hash"]] = Phonemized(tuple(record["symbols"]), tuple(SYMBOL_TO_ID[s] for s in record["symbols"]),
                                                tuple(record["stress"]), tuple(record["unknown"]))
    todo = {}
    for row in rows:
        if row["text_hash"] != text_key(row["text"]):
            raise ValueError(f"text hash mismatch for {row['id']}")
        if row["text_hash"] not in known:
            todo[row["text_hash"]] = row["text"]
    if todo:
        log(f"phonemizing {len(todo)} distinct texts")
        with cache.open("a", encoding="utf-8") as stream, concurrent.futures.ThreadPoolExecutor(workers) as pool:
            for key, result in zip(todo, pool.map(phonemize, todo.values())):
                known[key] = result
                append(stream, {"text_hash": key, "symbols": list(result.symbols), "stress": list(result.stress),
                                "unknown": list(result.unknown)}, sync=False)
    return known


def boundary_offsets(symbols, durations, voiced, reach=6):
    """Label-free timing probe. At vowel -> voiceless-consonant and voiceless
    fricative -> vowel boundaries the pitch tracker's voicing must flip; return
    the signed distance (frames) from the aligned boundary to the nearest flip."""
    found = {"vowel_to_voiceless": [], "fricative_to_vowel": []}
    ends = np.cumsum(durations)
    active = [i for i, d in enumerate(durations) if d > 0]
    for a, b in zip(active[:-1], active[1:]):
        if symbols[a] in VOWEL_SET and symbols[b] in VOICELESS:
            kind, before = "vowel_to_voiceless", True
        elif symbols[a] in FRICATIVES and symbols[b] in VOWEL_SET:
            kind, before = "fricative_to_vowel", False
        else:
            continue
        boundary = int(ends[a])
        best = None
        for shift in range(-reach, reach + 1):
            f = boundary + shift
            if 0 < f < len(voiced) and bool(voiced[f - 1]) == before and bool(voiced[f]) != before:
                if best is None or abs(shift) < abs(best):
                    best = shift
        found[kind].append(best)
    return found


def summarize_offsets(collected):
    summary = {}
    for kind, values in collected.items():
        hits = np.array([v for v in values if v is not None], dtype=np.float64)
        summary[kind] = {"boundaries": len(values), "flip_found": round(len(hits) / max(len(values), 1), 4)}
        if len(hits):
            summary[kind].update(mean_offset_frames=round(float(hits.mean()), 3), mean_abs_frames=round(float(np.abs(hits).mean()), 3),
                                 within_1_frame=round(float((np.abs(hits) <= 1).mean()), 4),
                                 within_2_frames=round(float((np.abs(hits) <= 2).mean()), 4))
    return summary


def gate(row, limits):
    qc, reasons = row["qc"], []
    if row["unknown"]:
        reasons.append("unknown_phoneme")
    if qc["weak_phone_fraction"] > limits["max_weak_phone_fraction"]:
        reasons.append("weak_alignment")
    if qc["token_probability_mean"] < limits["min_token_probability"]:
        reasons.append("low_confidence")
    if qc["max_phone_frames"] > limits["max_phone_frames"]:
        reasons.append("overlong_phone")
    if qc["vowel_voiced"] is not None and qc["vowel_voiced"] < limits["min_vowel_voiced"]:
        reasons.append("vowels_not_voiced")
    if not limits["min_phones_per_second"] <= qc["phones_per_second"] <= limits["max_phones_per_second"]:
        reasons.append("implausible_rate")
    return reasons


def quantiles(values):
    values = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if not len(values):
        return None
    return {str(q): round(float(np.percentile(values, q)), 4) for q in (0, 1, 5, 25, 50, 75, 95, 99, 100)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", type=Path, required=True, help="Directory with <split>.jsonl acoustic manifests")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--limit", type=int, help="Annotate only the first N rows per split (smoke test; never complete)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--refine", type=int, default=0, help="Boundary refinement passes with the from-scratch frame classifier")
    parser.add_argument("--refine-band", type=int, default=4)
    parser.add_argument("--stage", default="ctc", help="Durations used for the final manifests: ctc or refined-<n>")
    parser.add_argument("--min-pause-frames", type=int, default=AlignmentSettings.min_pause_frames)
    parser.add_argument("--max-weak-phone-fraction", type=float, default=0.25)
    parser.add_argument("--min-token-probability", type=float, default=0.5)
    parser.add_argument("--max-phone-frames", type=int, default=70)
    parser.add_argument("--min-vowel-voiced", type=float, default=0.5)
    parser.add_argument("--min-phones-per-second", type=float, default=4.0)
    parser.add_argument("--max-phones-per-second", type=float, default=30.0)
    args = parser.parse_args()
    config = Config.from_dict(json.loads(args.config.read_text())) if args.config else Config()
    settings = AlignmentSettings(min_pause_frames=args.min_pause_frames)
    limits = {key: getattr(args, key) for key in ("max_weak_phone_fraction", "min_token_probability", "max_phone_frames",
                                                    "min_vowel_voiced", "min_phones_per_second", "max_phones_per_second")}
    started = time.time()

    def log(message):
        print(json.dumps({"elapsed": round(time.time() - started, 1), "message": message}), flush=True)

    rows = {split: load_acoustic_rows(args.features, split, config) for split in args.splits}
    if args.limit:
        rows = {split: value[:args.limit] for split, value in rows.items()}
    recognizer = PhoneRecognizer(args.device)
    identity = {
        "annotation_version": ANNOTATION_VERSION, "acoustic": config.acoustic_fingerprint, "frontend": frontend_fingerprint(),
        **frontend_runtime(), "aligner": {"model": ALIGNER_MODEL, "revision": recognizer.revision},
        "alignment": settings.to_dict(), "dataset_revision": args.dataset_revision, "limit": args.limit,
        "sources": {split: file_hash(args.features / (split + ".jsonl")) for split in args.splits},
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "annotations.json"
    if metadata_path.exists():
        if not args.resume:
            raise FileExistsError("annotation directory exists; use --resume or a new directory")
        if json.loads(metadata_path.read_text())["identity"] != identity:
            raise ValueError("front end, aligner, settings or sources changed: these annotations are stale; use a new directory")
    atomic_json(metadata_path, {"identity": identity, "fingerprint": fingerprint, "complete": False})
    (args.output / "raw").mkdir(exist_ok=True)

    every = [row for split in args.splits for row in rows[split]]
    phonemes = phonemize_texts(every, args.output / "phonemes.jsonl", log=log)
    hashes = {split: {row["text_hash"] for row in rows[split]} for split in args.splits}
    for i, a in enumerate(args.splits):
        for b in args.splits[i + 1:]:
            if hashes[a] & hashes[b]:
                raise ValueError(f"sentence leakage between {a} and {b}")

    # 1. CTC alignment, journalled per utterance.
    raw = {}
    voicing = {}
    for split in args.splits:
        journal = args.output / "raw" / (split + ".jsonl")
        done = {record["id"]: record for record in recover(journal)}
        pending = [row for row in rows[split] if row["id"] not in done]
        log(f"aligning {split}: {len(pending)} pending, {len(done)} done")
        with journal.open("a", encoding="utf-8") as stream:
            for count, row in enumerate(pending, 1):
                item = torch.load(args.features / row["path"], map_location="cpu", weights_only=True)
                voicing[row["id"]] = item["prosody"][:, 1].numpy() > 0.5
                try:
                    if item["mel"].shape[0] != row["frames"]:
                        raise ValueError("frame count differs from the manifest")
                    durations, qc = align_utterance(recognizer, phonemes[row["text_hash"]], item["wave"].numpy(),
                                                    item["prosody"][:, 1].numpy(), config, settings)
                    record = {"id": row["id"], "ok": True, "durations": durations, "qc": qc}
                except (ValueError, RuntimeError) as error:
                    record = {"id": row["id"], "ok": False, "reason": f"{type(error).__name__}: {error}"}
                append(stream, record, sync=count % 200 == 0 or count == len(pending))
                done[row["id"]] = record
                if count % 500 == 0:
                    log(f"aligned {split} {count}/{len(pending)}")
        raw[split] = done
    del recognizer
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # 2. Optional boundary refinement over all splits together.
    stages = {"ctc": {split: {key: value["durations"] for key, value in raw[split].items() if value["ok"]} for split in args.splits}}
    if args.refine:
        from .refine import refine
        directories = [args.output / f"refined-{n + 1}" for n in range(args.refine)]
        if all((d / (split + ".jsonl")).exists() for d in directories for split in args.splits):
            for n, directory in enumerate(directories):
                stages[f"refined-{n + 1}"] = {split: {r["id"]: r["durations"] for r in recover(directory / (split + ".jsonl"))} for split in args.splits}
        else:
            items, owners = [], []
            for split in args.splits:
                for row in rows[split]:
                    if raw[split][row["id"]]["ok"]:
                        item = torch.load(args.features / row["path"], map_location="cpu", weights_only=True)
                        items.append({"symbols": phonemes[row["text_hash"]].symbols, "durations": raw[split][row["id"]]["durations"],
                                      "mel": item["mel"], "speaker": config.speakers.index(row["speaker"]), "group": row["text_hash"]})
                        owners.append((split, row["id"]))
            log(f"refining {len(items)} alignments")
            for n, durations in enumerate(refine(items, config, args.device, args.refine, args.refine_band, log=log)):
                directory = directories[n]
                directory.mkdir(exist_ok=True)
                stage = {split: {} for split in args.splits}
                for (split, identifier), value in zip(owners, durations):
                    stage[split][identifier] = value
                for split in args.splits:
                    temporary = directory / (split + ".jsonl.tmp")
                    temporary.write_text("".join(json.dumps({"id": k, "durations": v}) + "\n" for k, v in stage[split].items()))
                    temporary.replace(directory / (split + ".jsonl"))
                stages[f"refined-{n + 1}"] = stage
            del items
    if args.stage not in stages:
        raise ValueError(f"unknown stage {args.stage!r}; available: {sorted(stages)}")

    # 3. Label-free timing probe for every stage, then gate and write manifests.
    probe = {}
    for name, stage in stages.items():
        collected = {voice: defaultdict(list) for voice in config.speakers}
        for split in args.splits:
            for row in rows[split]:
                if row["id"] not in stage[split]:
                    continue
                if row["id"] not in voicing:
                    voicing[row["id"]] = torch.load(args.features / row["path"], map_location="cpu", weights_only=True)["prosody"][:, 1].numpy() > 0.5
                found = boundary_offsets(phonemes[row["text_hash"]].symbols, stage[split][row["id"]], voicing[row["id"]])
                for kind, values in found.items():
                    collected[row["speaker"]][kind].extend(values)
        probe[name] = {voice: summarize_offsets(collected[voice]) for voice in config.speakers}
        log({"timing_probe": name, **probe[name]})

    summary = {"limits": limits, "stage": args.stage, "timing_probe": probe, "splits": {}}
    chosen = stages[args.stage]
    phone_frames = {voice: defaultdict(list) for voice in config.speakers}
    pause_tokens = Counter()
    pause_nonzero = Counter()
    for split in args.splits:
        accepted, rejected = [], []
        for row in rows[split]:
            record = raw[split][row["id"]]
            result = phonemes[row["text_hash"]]
            if not record["ok"]:
                rejected.append({"id": row["id"], "speaker": row["speaker"], "text": row["text"], "reasons": [record["reason"]]})
                continue
            durations = chosen[split][row["id"]]
            entry = {"id": row["id"], "split": split, "speaker": row["speaker"], "text": row["text"], "text_hash": row["text_hash"],
                     "path": row["path"], "frames": row["frames"], "symbols": " ".join(result.symbols), "tokens": list(result.ids),
                     "stress": list(result.stress), "durations": durations, "unknown": list(result.unknown), "qc": record["qc"],
                     "provenance": {**row.get("provenance", {}), "dataset_revision": args.dataset_revision},
                     "acoustic": config.acoustic_fingerprint, "frontend": frontend_fingerprint(), "annotation": fingerprint}
            if sum(durations) != row["frames"] or len(durations) != len(result.ids):
                raise RuntimeError(f"durations of {row['id']} do not match its frames/tokens")
            reasons = gate(entry, limits)
            if reasons:
                rejected.append({"id": row["id"], "speaker": row["speaker"], "text": row["text"], "reasons": reasons, "qc": record["qc"]})
                continue
            accepted.append(entry)
            for symbol, frames in zip(result.symbols, durations):
                if symbol in PHONE_SET or symbol == UNK:
                    phone_frames[row["speaker"]][symbol].append(frames)
                else:
                    pause_tokens[symbol] += 1
                    pause_nonzero[symbol] += frames > 0
        for name, payload in ((split + ".jsonl", accepted), (split + ".rejected.jsonl", rejected)):
            temporary = args.output / (name + ".tmp")
            temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in payload), encoding="utf-8")
            temporary.replace(args.output / name)
        by_voice = {voice: sum(r["speaker"] == voice for r in accepted) for voice in config.speakers}
        summary["splits"][split] = {
            "accepted": len(accepted), "rejected": len(rejected), "accepted_by_voice": by_voice,
            "accepted_hours": round(sum(r["frames"] for r in accepted) * config.hop / config.sample_rate / 3600, 3),
            "rejection_reasons": dict(Counter(reason.split(":")[0] for r in rejected for reason in r["reasons"])),
            "qc_quantiles": {key: quantiles([raw[split][r["id"]]["qc"].get(key) for r in rows[split] if raw[split][r["id"]]["ok"]])
                             for key in ("token_probability_mean", "weak_phone_fraction", "max_phone_frames", "one_frame_phone_fraction",
                                         "phones_per_second", "vowel_voiced", "voiceless_voiced", "pause_voiced", "leading_frames",
                                         "trailing_frames", "pause_level_db")},
        }
    summary["phone_duration_frames"] = {
        voice: {symbol: {"count": len(values), "mean": round(float(np.mean(values)), 2), "median": float(np.median(values)),
                         "one_frame": round(float(np.mean(np.asarray(values) == 1)), 3)}
                for symbol, values in sorted(table.items(), key=lambda kv: -len(kv[1]))}
        for voice, table in phone_frames.items()}
    summary["pause_tokens"] = {symbol: {"count": pause_tokens[symbol], "with_frames": round(pause_nonzero[symbol] / pause_tokens[symbol], 4)}
                               for symbol in pause_tokens}
    complete = not args.limit and set(args.splits) == set(SPLITS)
    atomic_json(args.output / "summary.json", summary)
    atomic_json(metadata_path, {"identity": identity, "fingerprint": fingerprint, "complete": complete, "stage": args.stage,
                                "acoustic_root": str(args.features.resolve()), "limits": limits,
                                "counts": {split: summary["splits"][split]["accepted"] for split in args.splits}})
    log({"complete": complete, "splits": {s: {k: v for k, v in summary["splits"][s].items() if k != "qc_quantiles"} for s in args.splits}})


if __name__ == "__main__":
    main()
