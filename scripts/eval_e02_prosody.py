"""Reproducible E02 prosody calibration evaluation harness (0 training GPU).

Evaluates orthogonal single-factor sweeps before any combination on unseen French sentences:
  1. pitch_range  in (0.9, 1.0, 1.15, 1.3)
  2. pitch_jitter in (0.0, 0.1, 0.2) semitones
  3. voiced_vowels in (True, False)

Refuses to tune automatically on an under-trained checkpoint unless --allow-unready is set
(intended only for smoke-testing the pipeline when the experiment is still formally blocked).
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import soundfile as sf
import torch

from tiny_voca.infer import limit_peak, load_model, match_frontend
from tiny_voca.text import PHONE_SET, phonemize

DEFAULT_SENTENCES = ROOT / "tests" / "french_e02_prosody_sentences.json"
DEFAULT_CHECKPOINT = ROOT / ".local-runs" / "colab" / "opt1-1117" / "out" / "run_info" / "inference.pt"

ORTHOGONAL_VARIANTS = (
    ("baseline", 1.0, 0.0, True),
    ("range_0.9", 0.9, 0.0, True),
    ("range_1.15", 1.15, 0.0, True),
    ("range_1.3", 1.3, 0.0, True),
    ("jitter_0.1", 1.0, 0.1, True),
    ("jitter_0.2", 1.0, 0.2, True),
    ("voiced_vowels_false", 1.0, 0.0, False),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sentences(path: Path = DEFAULT_SENTENCES) -> list[dict]:
    items = json.loads(path.read_text(encoding="utf-8"))
    seen_ids, seen_texts = set(), set()
    for row in items:
        if row["id"] in seen_ids:
            raise ValueError(f"duplicate id {row['id']!r}")
        norm = row["text"].strip().casefold()
        if norm in seen_texts:
            raise ValueError(f"duplicate text {row['text']!r}")
        seen_ids.add(row["id"])
        seen_texts.add(norm)
    return items


def check_corpus_isolation(items: list[dict], exclude_paths: list[Path]) -> dict:
    """Verify 0 unknown phonemes and 0 textual overlap against reference corpora."""
    known = set()
    for p in exclude_paths:
        if not p.exists():
            continue
        if p.suffix == ".jsonl":
            with p.open("r", encoding="utf-8") as fh:
                for line in fh:
                    known.add(json.loads(line)["text"].strip().casefold())
        elif p.suffix == ".json":
            for row in json.loads(p.read_text(encoding="utf-8")):
                if "text" in row:
                    known.add(row["text"].strip().casefold())
    overlaps = [r["id"] for r in items if r["text"].strip().casefold() in known]
    unknowns = []
    token_counts = []
    for r in items:
        res = phonemize(r["text"])
        token_counts.append(len(res))
        if res.unknown:
            unknowns.append((r["id"], list(res.unknown)))
    return {
        "sentence_count": len(items),
        "categories": sorted({r["category"] for r in items}),
        "excluded_corpus_sentences": len(known),
        "overlap_ids": overlaps,
        "unknowns": unknowns,
        "tokens_min": min(token_counts),
        "tokens_median": round(statistics.median(token_counts), 1),
        "tokens_max": max(token_counts),
    }


def _utterance_prosody_metrics(model, phones, wave_np: np.ndarray, meta: dict, pitch_jitter: float, seed: int) -> dict:
    prosody = meta["prosody"][0].cpu().numpy()
    durations = meta["durations"][0].cpu().numpy()
    log_f0 = prosody[:, 0]
    if pitch_jitter > 0:
        wobble = model.pitch_wobble((1, len(log_f0)), pitch_jitter, seed, "cpu")[0].cpu().numpy()
        log_f0 = log_f0 + wobble
    voiced = prosody[:, 1] > 0
    st = log_f0 * (12.0 / math.log(2.0))
    voiced_st = st[voiced]
    f0_std_st = float(np.std(voiced_st)) if voiced_st.size >= 2 else 0.0
    f0_iqr_st = float(np.percentile(voiced_st, 75) - np.percentile(voiced_st, 25)) if voiced_st.size >= 2 else 0.0
    voiced_frac = float(np.mean(voiced))

    syms = phones.symbols
    phone_mask = np.array([s in PHONE_SET for s in syms], dtype=bool)
    vowel_mask = np.array([bool(model.vowel_tokens[i]) for i in phones.ids], dtype=bool)
    pause_mask = ~phone_mask

    # Map token-level durations cleanly onto frame-level voicing for vowels.
    ends = np.cumsum(durations)
    frame_owners = np.searchsorted(ends, np.arange(len(voiced)), side="right").clip(max=len(syms) - 1)
    vowel_frames = vowel_mask[frame_owners]
    vowel_voiced_frac = float(np.mean(voiced[vowel_frames])) if vowel_frames.any() else 1.0

    hop_sec = model.config.hop / model.config.sample_rate
    total_frames = int(durations.sum())
    pause_frames = int(durations[pause_mask].sum())
    internal_pause_frames = int(durations[1:-1][pause_mask[1:-1]].sum()) if len(syms) > 2 else 0
    speech_frames = max(total_frames - pause_frames, 1)
    phone_count = int(phone_mask.sum())

    finite = bool(np.isfinite(wave_np).all())
    peak = float(np.max(np.abs(wave_np))) if wave_np.size else 0.0
    return {
        "finite": finite,
        "peak": peak,
        "seconds": total_frames * hop_sec,
        "f0_std_st": f0_std_st,
        "f0_iqr_st": f0_iqr_st,
        "voiced_frac": voiced_frac,
        "vowel_voiced_frac": vowel_voiced_frac,
        "pause_ratio": pause_frames / max(total_frames, 1),
        "internal_pause_ms": internal_pause_frames * hop_sec * 1000.0,
        "phones_per_sec": phone_count / (speech_frames * hop_sec),
    }


def evaluate_variants(checkpoint: Path, items: list[dict], out_dir: Path, seed: int = 0, limit: int | None = None,
                      save_wavs_for: set[str] | None = None) -> dict:
    match_frontend(checkpoint)
    model = load_model(checkpoint, "cpu")
    rate = model.config.sample_rate
    sample_items = items[:limit] if limit else items
    wav_dir = out_dir / "wavs"
    if save_wavs_for:
        wav_dir.mkdir(parents=True, exist_ok=True)

    tokenized = [(row, phonemize(row["text"])) for row in sample_items]
    summary = {}
    for vid, p_range, p_jitter, v_vowels in ORTHOGONAL_VARIANTS:
        rows = []
        latencies = []
        for voice_idx, voice_name in enumerate(model.config.speakers):
            voice_t = torch.tensor([voice_idx], dtype=torch.long)
            for row, phones in tokenized:
                tok_t = torch.tensor([phones.ids], dtype=torch.long)
                str_t = torch.tensor([phones.stress], dtype=torch.long)
                len_t = torch.tensor([len(phones)], dtype=torch.long)
                t0 = time.perf_counter()
                wave_t, length_t, meta = model.synthesize(
                    tok_t, str_t, len_t, voice_t,
                    speed=1.0, pitch_shift=0.0, seed=seed,
                    pitch_range=p_range, pitch_jitter=p_jitter, voiced_vowels=v_vowels,
                )
                latencies.append((time.perf_counter() - t0) * 1000.0)
                wave_np = limit_peak(wave_t[0, :int(length_t[0])]).cpu().numpy()
                m = _utterance_prosody_metrics(model, phones, wave_np, meta, p_jitter, seed)
                m["voice"] = voice_name
                m["id"] = row["id"]
                rows.append(m)
                if save_wavs_for and row["id"] in save_wavs_for:
                    sf.write(str(wav_dir / f"{row['id']}_{voice_name}_{vid}.wav"), wave_np, rate)

        by_voice = {}
        for voice_name in ("all", *model.config.speakers):
            sub = rows if voice_name == "all" else [r for r in rows if r["voice"] == voice_name]
            total_audio_s = sum(r["seconds"] for r in sub)
            by_voice[voice_name] = {
                "utterances": len(sub),
                "all_finite": all(r["finite"] for r in sub),
                "f0_std_st_mean": round(statistics.mean(r["f0_std_st"] for r in sub), 3),
                "f0_iqr_st_mean": round(statistics.mean(r["f0_iqr_st"] for r in sub), 3),
                "voiced_frac_mean": round(statistics.mean(r["voiced_frac"] for r in sub), 4),
                "vowel_voiced_frac_mean": round(statistics.mean(r["vowel_voiced_frac"] for r in sub), 4),
                "pause_ratio_mean": round(statistics.mean(r["pause_ratio"] for r in sub), 4),
                "internal_pause_ms_mean": round(statistics.mean(r["internal_pause_ms"] for r in sub), 1),
                "phones_per_sec_mean": round(statistics.mean(r["phones_per_sec"] for r in sub), 2),
                "total_audio_seconds": round(total_audio_s, 2),
            }
        summary[vid] = {
            "params": {"pitch_range": p_range, "pitch_jitter": p_jitter, "voiced_vowels": v_vowels},
            "median_latency_ms": round(statistics.median(latencies), 2),
            "metrics": by_voice,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, help="Optional eval report.json to enforce readiness gate")
    parser.add_argument("--max-novel-wer", type=float, default=0.15)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Verify dataset isolation & phonemization only")
    parser.add_argument("--allow-unready", action="store_true", help="Run synthesis diagnostics even when checkpoint fails WER gate")
    parser.add_argument("--limit", type=int, help="Optional sentence cap for quick smoke test")
    args = parser.parse_args()

    items = load_sentences(args.sentences)
    excludes = [
        ROOT / "data" / "scripts_fr_4k" / "train.jsonl",
        ROOT / "data" / "scripts_fr_4k" / "validation.jsonl",
        ROOT / "data" / "scripts_fr_4k" / "test.jsonl",
        ROOT / "tests" / "french_phoneme_benchmark.json",
    ]
    isolation = check_corpus_isolation(items, excludes)
    if isolation["overlap_ids"] or isolation["unknowns"]:
        raise RuntimeError(f"dataset validation failed: {isolation}")

    if args.dry_run:
        print(json.dumps({"status": "dry_run_ok", "isolation": isolation}, indent=2, ensure_ascii=False))
        return

    if args.report and args.report.exists():
        rep = json.loads(args.report.read_text(encoding="utf-8"))
        novel_wer = rep.get("summary", {}).get("novel", {}).get("all", {}).get("generated_wer")
        if novel_wer is not None and novel_wer > args.max_novel_wer and not args.allow_unready:
            raise SystemExit(
                f"E02 BLOCKED: checkpoint baseline novel WER={novel_wer:.4f} > {args.max_novel_wer:.2f}; "
                "train fr4k checkpoint first or pass --allow-unready for diagnostic smoke run."
            )

    if args.out_dir is None:
        raise ValueError("--out-dir required unless --dry-run")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_ids = {"qt_01", "ex_01", "en_01", "sl_01"}
    results = evaluate_variants(args.checkpoint, items, args.out_dir, limit=args.limit, save_wavs_for=save_ids)
    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "isolation": isolation,
        "results": results,
    }
    (args.out_dir / "e02_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
