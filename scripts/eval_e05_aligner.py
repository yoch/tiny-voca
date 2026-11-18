"""Reproducible E05 French forced-aligner qualification and controlled boundary benchmark.

Performs 0-GPU diagnostic evaluation against unbiased physical waveform transitions:
  1. Compares old (pre-E01) alignment QC metrics on E01-changed vs unchanged utterances.
  2. Measures boundary timing accuracy on acoustic validation/test recordings via:
     - Symmetric 24 kHz LF (80-700 Hz) / HF (>3.2 kHz) spectral crossing for F>V and V>F.
     - High-frequency stop burst onset vs LF vowel dominance (>2.5 kHz vs 80-600 Hz) for Stop>V.
     - Short-window RMS energy transitions for <bos> and <eos> silences.
     - Pitch-tracker (pyin) voicing flips (to isolate pyin creak/lag from true CTC onset).
  3. Builds a curated controlled French boundary reference sample across F1 & M1 (optional).
"""

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from tiny_voca.config import Config
from tiny_voca.text import PHONE_SET, VOWEL_SET, phonemize

FRICATIVES = frozenset(("f", "s", "ʃ"))
STOPS = frozenset(("p", "t", "k"))


def lf_hf_band_ratio_db(wave: np.ndarray, sr: int = 24000, win: int = 256, hop: int = 24,
                        lf_band=(80.0, 700.0), hf_min: float = 3200.0) -> np.ndarray:
    """Symmetric Hann short-time LF/HF log energy ratio on a 1 ms (default) grid."""
    pad = win // 2
    padded = np.pad(np.asarray(wave, dtype=np.float64), (pad, pad))
    hann = np.hanning(win)
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    lf_mask = (freqs >= lf_band[0]) & (freqs <= lf_band[1])
    hf_mask = freqs >= hf_min
    starts = np.arange(0, len(wave), hop)
    frames = np.lib.stride_tricks.sliding_window_view(padded, win)[starts] * hann
    spec = np.abs(np.fft.rfft(frames, axis=-1)) ** 2
    lf = spec[:, lf_mask].sum(axis=1) + 1e-12
    hf = spec[:, hf_mask].sum(axis=1) + 1e-12
    return 10.0 * np.log10(lf / hf)


def rms_db_profile(wave: np.ndarray, hop: int = 64, win: int = 256) -> np.ndarray:
    """Short-window RMS energy in dB centered on multiples of ``hop`` samples."""
    pad = win // 2
    padded = np.pad(np.asarray(wave, dtype=np.float64), (pad, pad))
    starts = np.arange(0, len(wave), hop)
    frames = np.lib.stride_tricks.sliding_window_view(padded, win)[starts]
    mean_sq = np.mean(frames ** 2, axis=1)
    return 10.0 * np.log10(np.maximum(mean_sq, 1e-12))


def summarize_ms(values: list[float]) -> dict:
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "median_ms": round(float(np.median(arr)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "within_1frame_10_7ms": round(float(np.mean(np.abs(arr) <= 10.67)), 4),
        "within_2frames_21_3ms": round(float(np.mean(np.abs(arr) <= 21.33)), 4),
    }


def _classify_e01_change(text: str, old_syms: tuple[str, ...], new_syms: tuple[str, ...]) -> list[str]:
    old_s, new_s = " ".join(old_syms), " ".join(new_syms)
    tags = []
    if re.search(r"(?i)\bet\b", text):
        tags.append("et_liaison")
    if re.search(r"([tz]) \1", old_s) and not re.search(r"([tz]) \1", new_s):
        tags.append("geminate_tz")
    if re.search(r"\b\w+ou(?:es|ent)\b", text.lower()):
        tags.append("ouent")
    if re.search(r"\d|€|\$|£|°|\b(?:km|min|m|g|s)\b|\b(?:M|MM|Mgr|fig|vol|chap|av|ap)\.", text):
        tags.append("num_abbr_unit")
    if re.search(r"[dlnsmtjc]'|qu'|jusqu'|lorsqu'|puisqu'|quoiqu'|\bdos\b|\bcoud\b|\bbruns\b|\bvécut\b|enthousiasm|euthanas|zénith|rythme|Annecy|Nancy", text.lower()):
        tags.append("en_elision_or_hint")
    return tags or ["other_rule"]


def audit_changed_vs_unchanged(meta_root: Path, ann_root: Path) -> dict:
    old_ph = {
        r["text_hash"]: tuple(r["symbols"])
        for r in (json.loads(l) for l in (ann_root / "phonemes.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())
    }
    groups = {"changed": [], "unchanged": []}
    rejected_counts = Counter()
    tags = Counter()
    by_split = {}

    for sp in ("train", "validation", "test"):
        raw = {r["id"]: r for r in (json.loads(l) for l in (ann_root / "raw" / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
        rej = {json.loads(l)["id"] for l in (ann_root / f"{sp}.rejected.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
        sp_changed = 0
        rows = [json.loads(l) for l in (meta_root / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            th = hashlib.sha256(r["text"].strip().encode()).hexdigest()
            new_syms = phonemize(r["text"]).symbols
            is_changed = new_syms != old_ph[th]
            bucket = "changed" if is_changed else "unchanged"
            if is_changed:
                sp_changed += 1
                for t in _classify_e01_change(r["text"], old_ph[th], new_syms):
                    tags[t] += 1
            rec = raw[r["id"]]
            if rec.get("ok"):
                groups[bucket].append(rec["qc"])
            if r["id"] in rej:
                rejected_counts[bucket] += 1
        by_split[sp] = {"total": len(rows), "changed": sp_changed}

    summary = {"by_split": by_split, "change_tags": dict(tags), "groups": {}}
    for k, qcs in groups.items():
        probs = [q["token_probability_mean"] for q in qcs]
        weak = [q["weak_phone_fraction"] for q in qcs]
        onef = [q["one_frame_phone_fraction"] for q in qcs]
        summary["groups"][k] = {
            "raw_ok_count": len(qcs),
            "rejected": rejected_counts[k],
            "token_probability_mean": round(statistics.mean(probs), 4),
            "token_probability_min": round(min(probs), 4),
            "weak_phone_fraction_mean": round(statistics.mean(weak), 4),
            "weak_phone_fraction_max": round(max(weak), 4),
            "one_frame_phone_fraction_mean": round(statistics.mean(onef), 4),
        }
    return summary


def evaluate_physical_boundaries(ann_manifest: Path, features_root: Path, config: Config, limit: int | None = None) -> dict:
    rows = [json.loads(l) for l in ann_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]

    spectral = {("F1", "F>V"): [], ("F1", "V>F"): [], ("M1", "F>V"): [], ("M1", "V>F"): [],
                ("F1", "S>V"): [], ("M1", "S>V"): []}
    pyin = {("F1", "F>V"): [], ("F1", "V>F"): [], ("M1", "F>V"): [], ("M1", "V>F"): []}
    silence_bos, silence_eos = [], []

    for r in rows:
        item = torch.load(features_root / r["path"], map_location="cpu", weights_only=True)
        wave = item["wave"].numpy()
        voiced = item["prosody"][:, 1].numpy() > 0.5
        syms = r["symbols"].split()
        durs = r["durations"]
        ends = np.cumsum(durs)
        starts = ends - durs
        spk = r["speaker"]

        ratio_fric = lf_hf_band_ratio_db(wave, config.sample_rate, win=256, hop=24, lf_band=(80.0, 700.0), hf_min=3200.0)
        ratio_stop = lf_hf_band_ratio_db(wave, config.sample_rate, win=128, hop=24, lf_band=(80.0, 600.0), hf_min=2500.0)

        rms = rms_db_profile(wave, hop=64, win=256)
        loud, floor = np.percentile(rms, 95), np.percentile(rms, 2)
        thresh = min(loud - 30.0, max(loud - 45.0, floor + 8.0))
        speech_idx = np.flatnonzero(rms >= thresh)
        if speech_idx.size:
            first_ms = float(speech_idx[0] * 64 / config.sample_rate * 1000.0)
            last_ms = float(speech_idx[-1] * 64 / config.sample_rate * 1000.0)
            if durs[0] > 0:
                silence_bos.append(first_ms - float(ends[0] * config.hop / config.sample_rate * 1000.0))
            if durs[-1] > 0:
                silence_eos.append(last_ms - float(starts[-1] * config.hop / config.sample_rate * 1000.0))

        active = [i for i, d in enumerate(durs) if d > 0]
        for a, b in zip(active[:-1], active[1:]):
            if b != a + 1:
                continue
            sa, sb = syms[a], syms[b]
            bnd_frame = int(ends[a])
            bnd_ms = bnd_frame * config.hop / config.sample_rate * 1000.0
            center = int(round(bnd_ms))

            if (sa in FRICATIVES and sb in VOWEL_SET) or (sa in VOWEL_SET and sb in FRICATIVES):
                if min(durs[a], durs[b]) < 4:
                    continue
                kind = "F>V" if sa in FRICATIVES else "V>F"
                want_pos_to_neg = (kind == "V>F")
                lo, hi = max(1, center - 60), min(len(ratio_fric) - 1, center + 61)
                cross = [
                    t - bnd_ms for t in range(lo, hi)
                    if (want_pos_to_neg and ratio_fric[t - 1] > 0.0 >= ratio_fric[t])
                    or (not want_pos_to_neg and ratio_fric[t - 1] < 0.0 <= ratio_fric[t])
                ]
                if cross:
                    spectral[(spk, kind)].append(min(cross, key=abs))

                best_p = None
                for sh in range(-6, 7):
                    f = bnd_frame + sh
                    if 0 < f < len(voiced) and bool(voiced[f - 1]) == want_pos_to_neg and bool(voiced[f]) != want_pos_to_neg:
                        if best_p is None or abs(sh) < abs(best_p):
                            best_p = sh
                if best_p is not None:
                    pyin[(spk, kind)].append(best_p * config.hop / config.sample_rate * 1000.0)

            elif sa in STOPS and sb in VOWEL_SET and min(durs[a], durs[b]) >= 3:
                lo, hi = max(1, center - 50), min(len(ratio_stop) - 1, center + 50)
                cross = [t - bnd_ms for t in range(lo, hi) if ratio_stop[t - 1] < 0.0 <= ratio_stop[t]]
                if cross:
                    spectral[(spk, "S>V")].append(min(cross, key=abs))

    return {
        "utterances_evaluated": len(rows),
        "spectral_lf_hf": {f"{spk} {kind}": summarize_ms(vals) for (spk, kind), vals in sorted(spectral.items())},
        "pyin_voicing_flip": {f"{spk} {kind}": summarize_ms(vals) for (spk, kind), vals in sorted(pyin.items())},
        "silence_rms": {"bos_end": summarize_ms(silence_bos), "eos_start": summarize_ms(silence_eos)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta-root", type=Path, default=ROOT / ".local-runs" / "colab" / "fr4k" / "manifests" / "_meta")
    parser.add_argument("--ann-root", type=Path, default=ROOT / ".local-runs" / "colab" / "fr4k" / "annotations" / "annotations")
    parser.add_argument("--features-root", type=Path, default=ROOT / ".local-runs" / "colab" / "fr4k" / "features")
    parser.add_argument("--splits", nargs="+", default=["validation", "test"])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    c = Config()
    payload = {"qc_audit": audit_changed_vs_unchanged(args.meta_root, args.ann_root), "physical_probes": {}}
    for sp in args.splits:
        manifest = args.ann_root / f"{sp}.jsonl"
        if manifest.is_file() and (args.features_root / sp).is_dir():
            payload["physical_probes"][sp] = evaluate_physical_boundaries(manifest, args.features_root, c)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
