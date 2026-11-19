"""Independent P4 ablation evaluator (Étape 4): strict dev vs held-out final splits,
paired bootstrap 95% CIs against Control C and baseline 35k, per-voice breakdowns,
acoustics/timbre/artifact checks, optional WER+CER (faster-whisper + jiwer),
and blind A/B WAV exports.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys
import wave as wave_io
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tiny_voca.config import Config
from tiny_voca.data import AnnotatedDataset
from tiny_voca.evaluate import NOVEL_SENTENCES, dtw_distance
from tiny_voca.infer import load_model
from tiny_voca.losses import SpectralLoss
from tiny_voca.text import normalize_for_wer, phonemize
from tiny_voca.train import balanced_indices

from p0_expressivity import (
    _sentence_category,
    compute_intonation_metrics,
    compute_rhythm_metrics,
    extract_unbiased_f0,
    oracle_synthesize,
)

# Extra unseen French sentences (disjoint from NOVEL_SENTENCES & corpus) so ASR covers >= 32 novel clips.
EXTRA_NOVEL_SENTENCES = (
    "Pourquoi as-tu refermé la fenêtre avant la fin de l'orage ?",
    "Écoutez bien cette histoire, elle vous surprendra certainement !",
    "Dès l'aube, le boulanger préparait des pains chauds et des croissants dorés.",
    "Est-ce vraiment la seule solution que vous ayez trouvée ce matin ?",
    "Sans hésiter une seconde, elle traversa la place sous une pluie battante.",
    "Non, je ne pense pas qu'il soit déjà rentré de son voyage à Marseille.",
    "Quel silence étrange régnait dans les couloirs de l'ancienne bibliothèque !",
    "Les enfants couraient dans le jardin, riaient aux éclats, puis s'arrêtaient net.",
)

KEY_METRICS = (
    # Lower is better
    ("f0_mae_st", "lower"),
    ("f0_rmse_st", "lower"),
    ("f0_slope_mae_st_s", "lower"),
    ("end_delta_abs_err_st", "lower"),
    ("mel_dtw", "lower"),
    ("oracle_dur_stft", "lower"),
    ("phone_mae_ms", "lower"),
    ("phone_rel_dur_mae", "lower"),
    ("gen_f0_break_rate", "lower"),
    # Higher is better
    ("f0_corr", "higher"),
    ("energy_corr", "higher"),
    ("voicing_agreement", "higher"),
    ("gen_vowel_voiced", "higher"),
    ("voice_mel_cosine", "higher"),
    ("gen_accent_prom_st", "target"),
    ("gen_f0_std_st", "target"),
    ("gen_f0_local_std_st", "target"),
    ("pre_pausal_ratio", "target"),
    ("local_rate_cv", "target"),
)


def _write_wav(path: Path, audio: np.ndarray, sr: int = 24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    pcm = np.clip(np.round(audio * 32767.0), -32768, 32767).astype(np.int16)
    with wave_io.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _select_disjoint_pairs(
    manifest_path: Path,
    exclude_ids: set[str],
    per_category_pairs: int,
    seed: int,
) -> list[dict]:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_text: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r["id"] in exclude_ids:
            continue
        by_text[r["text_hash"]][r["speaker"]] = r

    cats: dict[str, list[tuple[str, dict[str, dict]]]] = defaultdict(list)
    for th, spks in sorted(by_text.items()):
        if "F1" in spks and "M1" in spks:
            cats[_sentence_category(spks["F1"]["text"])].append((th, spks))

    rng = random.Random(seed)
    selected: list[dict] = []
    for cat in ("plain_assert", "question", "exclamation", "comma_pause"):
        pool = list(cats.get(cat, []))
        rng.shuffle(pool)
        # Prefer medium-to-rich sentences (between 20 and 220 characters)
        pool.sort(key=lambda item: (not (20 <= len(item[1]["F1"]["text"]) <= 220), item[0]))
        chosen = pool[:per_category_pairs]
        for _, spks in chosen:
            for spk in ("F1", "M1"):
                row = dict(spks[spk])
                row["category"] = cat
                selected.append(row)
    return selected


def _mean_opt(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return round(float(np.mean(xs)), 5) if xs else None


def _paired_bootstrap_ci(
    a_by_id: dict[str, float | None],
    b_by_id: dict[str, float | None],
    ids: list[str],
    n_boot: int = 2000,
    seed: int = 20261119,
) -> dict | None:
    diffs = []
    for uid in ids:
        va, vb = a_by_id.get(uid), b_by_id.get(uid)
        if va is not None and vb is not None and math.isfinite(va) and math.isfinite(vb):
            diffs.append(float(va - vb))
    if not diffs:
        return None
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(len(arr)),
        "mean_delta": round(float(arr.mean()), 5),
        "ci95_low": round(float(lo), 5),
        "ci95_high": round(float(hi), 5),
        "significant": bool(lo > 0.0 or hi < 0.0),
    }


def _summarize_rows(rows: list[dict], filter_fn=None) -> dict:
    sub = [r for r in rows if (filter_fn(r) if filter_fn else True)]
    if not sub:
        return {"n": 0}
    out = {"n": len(sub)}
    for key in (
        "mel_dtw", "oracle_dur_stft", "voice_mel_cosine",
        "f0_mae_st", "f0_rmse_st", "f0_corr", "f0_slope_mae_st_s",
        "ref_f0_std_st", "gen_f0_std_st", "ref_f0_local_std_st", "gen_f0_local_std_st",
        "ref_accent_prom_st", "gen_accent_prom_st", "end_delta_abs_err_st",
        "energy_corr", "ref_vowel_voiced", "gen_vowel_voiced", "voicing_agreement",
        "ref_f0_break_rate", "gen_f0_break_rate",
        "total_dur_ratio", "total_dur_abs_err_pct", "phone_mae_ms", "phone_rel_dur_mae",
        "phone_dur_corr", "pre_pausal_ratio_ref", "pre_pausal_ratio", "local_rate_cv_ref", "local_rate_cv",
    ):
        out[key] = _mean_opt([r.get(key) for r in sub])

    tp = sum(int(r.get("pause_tp", 0)) for r in sub)
    fp = sum(int(r.get("pause_fp", 0)) for r in sub)
    fn = sum(int(r.get("pause_fn", 0)) for r in sub)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    tp_weighted = [(float(r["pause_tp_mae_ms"]), int(r.get("pause_tp", 0))) for r in sub
                   if r.get("pause_tp_mae_ms") is not None and int(r.get("pause_tp", 0)) > 0]
    out["pause_tp"] = tp
    out["pause_fp"] = fp
    out["pause_fn"] = fn
    out["pause_precision"] = round(prec, 4)
    out["pause_recall"] = round(rec, 4)
    out["pause_f1"] = round(f1, 4)
    if tp_weighted:
        out["pause_tp_mae_ms"] = round(sum(m * c for m, c in tp_weighted) / sum(c for _, c in tp_weighted), 2)
    else:
        out["pause_tp_mae_ms"] = None
    return out


def _mel_profile_cosine(ref_mel: np.ndarray, gen_mel: np.ndarray) -> float:
    # Mean log-mel spectral profile cosine similarity over [mel, frames] (proxy for voice timbre consistency).
    a = ref_mel.mean(axis=1).astype(np.float64)
    b = gen_mel.mean(axis=1).astype(np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / max(denom, 1e-12))


def _run_asr(audio_items: list[tuple[str, str, str, np.ndarray]], sr: int = 24000, model_size: str = "small") -> dict:
    from faster_whisper import WhisperModel
    import jiwer
    import librosa

    whisper = WhisperModel(model_size, device="cpu", compute_type="int8")
    by_split_spk: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for split, spk, ref_text, wav in audio_items:
        wav16 = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        segs, _ = whisper.transcribe(wav16, language="fr", beam_size=5, vad_filter=False)
        hyp = " ".join(s.text.strip() for s in segs).strip()
        r_norm = normalize_for_wer(ref_text, language="fr")
        h_norm = normalize_for_wer(hyp, language="fr")
        by_split_spk[split][spk].append((r_norm, h_norm))
        by_split_spk[split]["all"].append((r_norm, h_norm))

    out = {}
    for split, groups in by_split_spk.items():
        out[split] = {}
        for grp, pairs in groups.items():
            refs = [p[0] for p in pairs]
            hyps = [p[1] for p in pairs]
            out[split][grp] = {
                "samples": len(pairs),
                "wer": round(float(jiwer.wer(refs, hyps)), 4),
                "cer": round(float(jiwer.cer(refs, hyps)), 4),
            }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME=PATH_TO_INFERENCE_PT (repeatable; include base_35k and C for paired deltas)",
    )
    parser.add_argument("--partition", choices=("dev", "final"), required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path(".local-runs/colab/fr4k/annotations_e01"),
    )
    parser.add_argument(
        "--p0-report",
        type=Path,
        default=Path(".local-runs/p0_expressivity/p0_report.json"),
        help="P0 report whose 40 test IDs are strictly excluded from final partition",
    )
    parser.add_argument("--per-category-pairs", type=int, default=0,
                        help="Override pairs per category (defaults: 5 for dev=40 utt, 10 for final=80 utt)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--with-asr", action="store_true", help="Compute faster-whisper WER and CER on partition + 32 novel clips")
    parser.add_argument("--export-blind-wavs", action="store_true", help="Export randomized blind A/B listening WAVs")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    variants: list[tuple[str, Path]] = []
    for spec in args.variant:
        if "=" not in spec:
            parser.error(f"invalid --variant {spec!r}, expected NAME=PATH")
        name, path_s = spec.split("=", 1)
        variants.append((name.strip(), Path(path_s.strip())))

    cfg = Config()
    if args.partition == "dev":
        manifest = args.annotations / "validation.jsonl"
        ds = AnnotatedDataset(manifest, cfg, cache=False)
        exclude_ids = {ds.rows[i]["id"] for i in balanced_indices(ds, 32)}
        per_cat = args.per_category_pairs or 5
    else:
        manifest = args.annotations / "test.jsonl"
        ds = AnnotatedDataset(manifest, cfg, cache=False)
        exclude_ids = set(json.loads(args.p0_report.read_text(encoding="utf-8"))["selected_ids"]) if args.p0_report.exists() else set()
        per_cat = args.per_category_pairs or 10

    selected = _select_disjoint_pairs(manifest, exclude_ids, per_category_pairs=per_cat, seed=args.seed)
    id_to_idx = {r["id"]: i for i, r in enumerate(ds.rows)}
    items = {r["id"]: ds[id_to_idx[r["id"]]] for r in selected}

    # Pre-extract reference F0 and log-mel ONCE (shared across all variants!)
    spectral = SpectralLoss(cfg.sample_rate).to(args.device)
    ref_cache = {}
    for row in selected:
        uid = row["id"]
        it = items[uid]
        ref_wave = it["wave"].numpy()
        ref_mel = spectral.log_mel(it["wave"][None].to(args.device))[0].cpu().numpy()
        ref_f0 = extract_unbiased_f0(ref_wave, cfg, int(it["durations"].sum()))
        ref_cache[uid] = {"wave": ref_wave, "mel": ref_mel, "f0": ref_f0}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_variant_rows: dict[str, list[dict]] = {}
    per_variant_summary: dict[str, dict] = {}
    asr_summary: dict[str, dict] = {}
    blind_wav_cache: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    # Pick 6 representative blind listening sentences (question, exclamation, comma_pause x F1/M1)
    blind_ids = []
    for cat in ("question", "exclamation", "comma_pause"):
        for spk in ("F1", "M1"):
            cand = next((r["id"] for r in selected if r["category"] == cat and r["speaker"] == spk), None)
            if cand:
                blind_ids.append(cand)

    for vname, ckpt_path in variants:
        model = load_model(ckpt_path, device=args.device, check_frontend=True)
        rows_out: list[dict] = []
        asr_clips: list[tuple[str, str, str, np.ndarray]] = []
        for idx, row in enumerate(selected):
            uid = row["id"]
            it = items[uid]
            rc = ref_cache[uid]
            ref_durs = it["durations"].numpy()
            toks = it["tokens"].numpy()

            # Free-running A_free
            gen_wave, meta = oracle_synthesize(model, row, it, "A_free", seed=idx)
            gen_durs = meta["durations"]
            gen_mel = spectral.log_mel(torch.from_numpy(gen_wave)[None].to(args.device))[0].cpu().numpy()
            dtw = float(dtw_distance(gen_mel, rc["mel"]))
            timbre_cos = _mel_profile_cosine(rc["mel"], gen_mel)

            # Oracle-duration STFT check (measures pure spectral/vocoder artifact non-regression)
            b_wave, _ = oracle_synthesize(model, row, it, "B_oracle_dur", seed=idx)
            min_len = min(len(b_wave), len(rc["wave"]))
            with torch.no_grad():
                stft_b, _ = spectral(
                    torch.from_numpy(b_wave[:min_len])[None].to(args.device),
                    torch.from_numpy(rc["wave"][:min_len])[None].to(args.device),
                    torch.tensor([min_len], device=args.device),
                )
            inton = compute_intonation_metrics(rc["wave"], gen_wave, ref_durs, gen_durs, toks, cfg, ref_f0_info=rc["f0"])
            rhythm = compute_rhythm_metrics(ref_durs, gen_durs, toks, cfg)

            rec = {
                "id": uid,
                "speaker": row["speaker"],
                "category": row["category"],
                "text": row["text"],
                "mel_dtw": round(dtw, 5),
                "oracle_dur_stft": round(float(stft_b.item()), 5),
                "voice_mel_cosine": round(timbre_cos, 5),
                **{k: v for k, v in inton.items() if isinstance(v, (int, float)) or v is None},
                "total_dur_ratio": rhythm["total_dur_ratio"],
                "total_dur_abs_err_pct": rhythm["total_dur_abs_err_pct"],
                "phone_mae_ms": rhythm["phone_mae_ms"],
                "phone_rel_dur_mae": rhythm["phone_rel_dur_mae"],
                "phone_dur_corr": rhythm["phone_dur_corr"],
                "pre_pausal_ratio_ref": rhythm["pre_pausal_ratio_ref"],
                "pre_pausal_ratio": rhythm["pre_pausal_ratio_gen"],
                "local_rate_cv_ref": rhythm["local_rate_cv_ref"],
                "local_rate_cv": rhythm["local_rate_cv_gen"],
                "pause_tp": rhythm["pause_tp"],
                "pause_fp": rhythm["pause_fp"],
                "pause_fn": rhythm["pause_fn"],
                "pause_tp_mae_ms": rhythm["pause_tp_mae_ms"],
            }
            rows_out.append(rec)
            if args.with_asr:
                asr_clips.append((args.partition, row["speaker"], row["text"], gen_wave))
            if args.export_blind_wavs and uid in blind_ids:
                blind_wav_cache[uid][vname] = gen_wave

        # Expanded novel ASR (16 unseen French texts x 2 voices = 32 clips)
        if args.with_asr:
            all_novel = list(NOVEL_SENTENCES) + list(EXTRA_NOVEL_SENTENCES)
            for n_idx, text in enumerate(all_novel):
                ph = phonemize(text)
                t_ids = torch.tensor([ph.ids], dtype=torch.long, device=args.device)
                t_str = torch.tensor([ph.stress], dtype=torch.long, device=args.device)
                t_len = torch.tensor([len(ph.ids)], dtype=torch.long, device=args.device)
                for spk_i, spk_name in enumerate(cfg.speakers):
                    with torch.inference_mode():
                        w_t, l_t, _ = model.synthesize(
                            t_ids, t_str, t_len, torch.tensor([spk_i], dtype=torch.long, device=args.device),
                            speed=1.0, seed=1000 + n_idx * 2 + spk_i, pitch_range=1.0, pitch_jitter=0.0, voiced_vowels=True,
                        )
                    asr_clips.append(("novel", spk_name, text, w_t[0, :int(l_t[0])].cpu().numpy()))
            asr_summary[vname] = _run_asr(asr_clips, sr=cfg.sample_rate)

        per_variant_rows[vname] = rows_out
        per_variant_summary[vname] = {
            "all": _summarize_rows(rows_out),
            "F1": _summarize_rows(rows_out, lambda r: r["speaker"] == "F1"),
            "M1": _summarize_rows(rows_out, lambda r: r["speaker"] == "M1"),
            "question": _summarize_rows(rows_out, lambda r: r["category"] == "question"),
            "exclamation": _summarize_rows(rows_out, lambda r: r["category"] == "exclamation"),
            "comma_pause": _summarize_rows(rows_out, lambda r: r["category"] == "comma_pause"),
            "plain_assert": _summarize_rows(rows_out, lambda r: r["category"] == "plain_assert"),
        }

    # Reference ASR baseline on held-out audio if --with-asr
    if args.with_asr:
        ref_clips = [(args.partition, r["speaker"], r["text"], ref_cache[r["id"]]["wave"]) for r in selected]
        asr_summary["reference_audio"] = _run_asr(ref_clips, sr=cfg.sample_rate)

    # Paired bootstrap 95% CIs vs Control C and vs base_35k
    paired_deltas: dict[str, dict] = {}
    for ref_name in ("C", "base_35k"):
        if ref_name not in per_variant_rows:
            continue
        paired_deltas[f"vs_{ref_name}"] = {}
        ref_map = {r["id"]: r for r in per_variant_rows[ref_name]}
        for vname, rows_v in per_variant_rows.items():
            if vname == ref_name:
                continue
            v_map = {r["id"]: r for r in rows_v}
            comp = {}
            for grp, subset_ids in (
                ("all", [r["id"] for r in selected]),
                ("F1", [r["id"] for r in selected if r["speaker"] == "F1"]),
                ("M1", [r["id"] for r in selected if r["speaker"] == "M1"]),
                ("question", [r["id"] for r in selected if r["category"] == "question"]),
            ):
                comp[grp] = {}
                for mkey, _ in KEY_METRICS:
                    ci = _paired_bootstrap_ci(
                        {uid: v_map[uid].get(mkey) for uid in subset_ids},
                        {uid: ref_map[uid].get(mkey) for uid in subset_ids},
                        subset_ids,
                        seed=args.seed,
                    )
                    if ci is not None:
                        comp[grp][mkey] = ci
            paired_deltas[f"vs_{ref_name}"][vname] = comp

    # Export randomized blind A/B listening files if requested
    blind_manifest = {}
    if args.export_blind_wavs:
        blind_dir = args.out_dir / "blind_listening"
        blind_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(args.seed + 77)
        for uid in blind_ids:
            row = next(r for r in selected if r["id"] == uid)
            _write_wav(blind_dir / f"{uid}_REF.wav", ref_cache[uid]["wave"], cfg.sample_rate)
            vnames = list(blind_wav_cache[uid].keys())
            rng.shuffle(vnames)
            mapping = {}
            for slot_i, vn in enumerate(vnames):
                slot = chr(ord("A") + slot_i)
                fname = f"{uid}_sample_{slot}.wav"
                _write_wav(blind_dir / fname, blind_wav_cache[uid][vn], cfg.sample_rate)
                mapping[slot] = vn
            blind_manifest[uid] = {"speaker": row["speaker"], "category": row["category"], "text": row["text"], "key": mapping}
        (blind_dir / "blind_key.json").write_text(json.dumps(blind_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = {
        "partition": args.partition,
        "manifest": str(manifest),
        "utterances": len(selected),
        "excluded_count": len(exclude_ids),
        "selected_ids": [r["id"] for r in selected],
        "variants": {k: str(v) for k, v in variants},
        "summary": per_variant_summary,
        "paired_bootstrap_95ci": paired_deltas,
        "asr": asr_summary,
        "blind_listening_ids": blind_ids if args.export_blind_wavs else [],
    }
    (args.out_dir / f"p4_{args.partition}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / f"p4_{args.partition}_per_utterance.json").write_text(
        json.dumps(per_variant_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Concise tabular stdout summary
    print(f"=== P4 {args.partition.upper()} SUMMARY (n={len(selected)}, excluded={len(exclude_ids)}) ===")
    for vname in [v[0] for v in variants]:
        s = per_variant_summary[vname]["all"]
        q = per_variant_summary[vname]["question"]
        m1 = per_variant_summary[vname]["M1"]
        print(
            f"  {vname:12s} | f0_mae={s['f0_mae_st']:.4f} f0_r={s['f0_corr']:.4f} slope={s['f0_slope_mae_st_s']:.2f} "
            f"end={s['end_delta_abs_err_st']:.3f} Q_f0={q['f0_mae_st']:.3f}(r={q['f0_corr']:.3f},end={q['end_delta_abs_err_st']:.2f}) "
            f"prom={s['gen_accent_prom_st']:.3f} M1_vow={m1['gen_vowel_voiced']:.3f} "
            f"dtw={s['mel_dtw']:.4f} stftB={s['oracle_dur_stft']:.4f} ph_rel={s['phone_rel_dur_mae']:.4f} pF1={s['pause_f1']:.4f}"
        )
    if "vs_C" in paired_deltas:
        print("\n=== PAIRED DELTAS vs Control C (all, 95% bootstrap CI) ===")
        for vname, grp_map in paired_deltas["vs_C"].items():
            parts = []
            for mkey in ("f0_mae_st", "f0_corr", "f0_slope_mae_st_s", "end_delta_abs_err_st", "mel_dtw", "oracle_dur_stft"):
                ci = grp_map["all"].get(mkey)
                if ci:
                    star = "*" if ci["significant"] else ""
                    parts.append(f"{mkey}={ci['mean_delta']:+.4f}[{ci['ci95_low']:+.4f},{ci['ci95_high']:+.4f}]{star}")
            print(f"  {vname:12s} - C: " + " | ".join(parts))
    if asr_summary:
        print("\n=== ASR WER / CER ===")
        for vname, sp_map in asr_summary.items():
            print(f"  {vname:16s}: " + json.dumps(sp_map))


if __name__ == "__main__":
    main()
