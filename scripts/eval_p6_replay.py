"""Offline CPU evaluator for P6 paired E06 replay (arms C, D, P at 12k or 11k).

Implements all pre-registered P6 Item 3 & Item 5 metrics on CPU (0 GPU):
  1. Primary criterion (host fr_ref_35k F0 injection on identical 35k predicted durations,
     identical energy, voicing, and noise):
       - common_mask_f0_corr (PRIMARY)
       - common_mask_f0_mae_st, common_mask_slope_mae_st_s, common_mask_coverage
       - Stratified text-level paired 95% bootstrap CIs (F1/M1 grouped per text) for
         D - C and P - C (All, F1, M1).
  2. Native WAV checks for C, D, P:
       - Mel-DTW (free-running), STFT (oracle durations), WER & CER (faster-whisper small)
       - Terminal inflection of French questions (totales vs partielles, by F1 & M1)
         on validation.jsonl paired questions.
  3. Automated verdict against pre-registered win & non-regression margins (determining
     whether the 121 untouched test.jsonl pairs may be touched for single confirmation).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tiny_voca.config import Config
from tiny_voca.data import AnnotatedDataset
from tiny_voca.evaluate import dtw_distance, normalize_for_wer
from tiny_voca.losses import SpectralLoss
from tiny_voca.text import PHONE_SET, SYMBOLS

from eval_p4_ablation import stratified_text_bootstrap_ci
from eval_p5_f0_causal import (
    _apply_common_mask,
    _direction_label,
    _heard_f0_metrics,
    _peak_syllable_stats,
    _summarize,
    classify_french_question,
    load_any_model,
    predict_donor_log_f0,
    sha256_file,
    synthesize_decoupled_f0,
)
from p0_expressivity import (
    SEMITONES,
    _terminal_inflection_st,
    compute_intonation_metrics,
    compute_rhythm_metrics,
    extract_unbiased_f0,
    oracle_synthesize,
    warped_token_alignment,
)

# Pre-registered margins (frozen in docs/experiments/P6-preregistration.md before GPU).
PREREG_MARGINS = {
    "primary_corr_ci95_low_min": 0.0,      # P - C common_mask_f0_corr 95% CI low > 0
    "primary_mae_max_delta_st": 0.05,      # P - C common_mask_f0_mae_st <= +0.05 st
    "native_mel_dtw_max_delta": 0.015,     # P - C native Mel-DTW <= +0.015
    "native_mel_dtw_ci95_hi_max": 0.025,   # P - C native Mel-DTW 95% CI high <= +0.025
    "native_stft_max_delta": 0.05,         # P - C native oracle-dur STFT <= +0.05
    "native_wer_max_delta": 0.020,         # P - C native WER <= +2.0 pp
    "native_cer_max_delta": 0.010,         # P - C native CER <= +1.0 pp
    "question_totale_dir_acc_max_drop": 0.05,  # P - C totale dirAcc >= -5 pp
}


def _load_p4_dev_rows(val_manifest: Path, p4_dev_report: Path) -> list[dict]:
    dev_meta = json.loads(p4_dev_report.read_text(encoding="utf-8"))
    selected_ids = list(dev_meta["selected_ids"])
    all_val = {r["id"]: r for r in (json.loads(l) for l in val_manifest.read_text(encoding="utf-8").splitlines() if l.strip())}
    rows = []
    from p0_expressivity import _sentence_category
    for uid in selected_ids:
        r = dict(all_val[uid])
        r["category"] = _sentence_category(r["text"])
        rows.append(r)
    return rows


def _load_reserve121_rows(test_manifest: Path, reserve_json: Path) -> list[dict]:
    reserve = json.loads(reserve_json.read_text(encoding="utf-8"))
    allowed_hashes = set(reserve["untouched_text_hashes"])
    all_test = [json.loads(l) for l in test_manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    from p0_expressivity import _sentence_category
    rows = []
    for r in all_test:
        if r["text_hash"] in allowed_hashes:
            rr = dict(r)
            rr["category"] = _sentence_category(rr["text"])
            rows.append(rr)
    rows.sort(key=lambda x: (x["category"], x["text_hash"], x["speaker"]))
    return rows


def _mean_opt(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return round(float(np.mean(xs)), 5) if xs else None


def _eval_questions_native(
    arms: dict[str, torch.nn.Module],
    val_ds: AnnotatedDataset,
    cfg: Config,
) -> dict:
    """Evaluate terminal inflection of French questions (totales vs partielles, by voice) on validation.jsonl."""
    by_hash: dict[str, dict[str, int]] = defaultdict(dict)
    for i, r in enumerate(val_ds.rows):
        if "?" in r["text"]:
            by_hash[r["text_hash"]][r["speaker"]] = i
    paired_indices = []
    for th in sorted(by_hash):
        if "F1" in by_hash[th] and "M1" in by_hash[th]:
            paired_indices.extend([by_hash[th]["F1"], by_hash[th]["M1"]])

    ref_cache = {}
    for idx in paired_indices:
        row = val_ds.rows[idx]
        item = val_ds[idx]
        ref_wave = item["wave"].numpy()
        ref_f0 = extract_unbiased_f0(ref_wave, cfg, frames=int(item["durations"].sum()))
        ref_st = ref_f0["continuous_log_f0"] * SEMITONES
        r_tok = item["tokens"].numpy()
        r_dur = item["durations"].numpy()
        r_phone = np.repeat([SYMBOLS[int(t)] in PHONE_SET for t in r_tok], r_dur)[: len(ref_st)]
        ref_end = _terminal_inflection_st(ref_st, ref_f0["voiced"] & r_phone)
        ref_pk = _peak_syllable_stats(ref_st, ref_f0["voiced"] & r_phone, r_tok, r_dur)
        ref_cache[row["id"]] = {
            "qtype": classify_french_question(row["text"]),
            "speaker": row["speaker"],
            "text": row["text"],
            "ref_end": ref_end,
            "ref_dir": _direction_label(ref_end),
            "ref_pk": ref_pk,
            "item": item,
            "row": row,
        }

    out = {}
    for arm_name, model in arms.items():
        recs = []
        for k_i, idx in enumerate(paired_indices):
            row = val_ds.rows[idx]
            rc = ref_cache[row["id"]]
            gen_wave, meta = oracle_synthesize(model, row, rc["item"], "A_free", seed=2000 + k_i)
            g_dur = meta["durations"]
            g_f0 = extract_unbiased_f0(gen_wave, cfg, frames=int(np.sum(g_dur)))
            g_st = g_f0["continuous_log_f0"] * SEMITONES
            g_tok = rc["item"]["tokens"].numpy()
            g_phone = np.repeat([SYMBOLS[int(t)] in PHONE_SET for t in g_tok], g_dur)[: len(g_st)]
            gen_end = _terminal_inflection_st(g_st, g_f0["voiced"] & g_phone)
            gen_dir = _direction_label(gen_end)
            gen_pk = _peak_syllable_stats(g_st, g_f0["voiced"] & g_phone, g_tok, g_dur)
            recs.append({
                "id": row["id"],
                "speaker": row["speaker"],
                "qtype": rc["qtype"],
                "ref_end": rc["ref_end"],
                "gen_end": gen_end,
                "end_abs_err_st": abs(gen_end - rc["ref_end"]) if (gen_end is not None and rc["ref_end"] is not None) else None,
                "dir_match": (1.0 if gen_dir == rc["ref_dir"] else 0.0) if (gen_dir and rc["ref_dir"]) else None,
                "ref_dir": rc["ref_dir"],
                "gen_dir": gen_dir,
                "pk_exact": (1.0 if gen_pk["peak_syl_from_end"] == rc["ref_pk"]["peak_syl_from_end"] else 0.0)
                if (gen_pk["peak_syl_from_end"] is not None and rc["ref_pk"]["peak_syl_from_end"] is not None) else None,
            })

        arm_summary = {}
        for qtype in ("totale", "partielle"):
            arm_summary[qtype] = {}
            for spk in ("all", "F1", "M1"):
                sub = [x for x in recs if x["qtype"] == qtype and (spk == "all" or x["speaker"] == spk)]
                arm_summary[qtype][spk] = {
                    "n": len(sub),
                    "ref_end_mean_st": _mean_opt([x["ref_end"] for x in sub]),
                    "gen_end_mean_st": _mean_opt([x["gen_end"] for x in sub]),
                    "end_abs_err_st": _mean_opt([x["end_abs_err_st"] for x in sub]),
                    "dir_match_acc": _mean_opt([x["dir_match"] for x in sub]),
                    "pk_exact_acc": _mean_opt([x["pk_exact"] for x in sub]),
                    "gen_dir_counts": dict(Counter(x["gen_dir"] for x in sub if x["gen_dir"])),
                }
        out[arm_name] = arm_summary
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition", choices=("dev", "reserve121"), default="dev")
    parser.add_argument("--annotations", type=Path, default=Path(".local-runs/colab/fr4k/annotations_e01"))
    parser.add_argument("--p4-dev-report", type=Path, default=Path(".local-runs/p4_ablation/p4_final/dev_screen/p4_dev_report.json"))
    parser.add_argument("--reserve-manifest", type=Path, default=Path("docs/experiments/p5_untouched_test_reserve.json"))
    parser.add_argument("--host-35k", type=Path, default=Path(".local-runs/colab/fr_ref_35k/fr_ref_35k_final/inference.pt"))
    parser.add_argument("--e06-8k", type=Path, default=Path(".local-runs/colab/e06_20k/e06_20k_final/step-00008000.pt"))
    parser.add_argument("--arm-c", type=Path, required=True)
    parser.add_argument("--arm-d", type=Path, required=True)
    parser.add_argument("--arm-p", type=Path, required=True)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-questions", action="store_true")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    host_35k, step_host = load_any_model(args.host_35k, "cpu")
    cfg = host_35k.config
    spectral = SpectralLoss(cfg.sample_rate)

    donors = {}
    ckpt_meta = {
        "host_35k": {"path": str(args.host_35k), "step": step_host, "sha256": sha256_file(args.host_35k)},
    }
    for name, path in (("e06_8k", args.e06_8k), ("C", args.arm_c), ("D", args.arm_d), ("P", args.arm_p)):
        m, st = load_any_model(path, "cpu")
        donors[name] = m
        ckpt_meta[name] = {"path": str(path), "step": st, "sha256": sha256_file(path)}

    manifest_path = args.annotations / ("validation.jsonl" if args.partition == "dev" else "test.jsonl")
    ds = AnnotatedDataset(manifest_path, cfg, voiced_vowels=True)
    id_to_idx = {r["id"]: i for i, r in enumerate(ds.rows)}
    if args.partition == "dev":
        selected = _load_p4_dev_rows(manifest_path, args.p4_dev_report)
    else:
        selected = _load_reserve121_rows(manifest_path, args.reserve_manifest)
    print(f"Evaluating P6 on partition={args.partition} ({len(selected)} recordings / {len(selected)//2} texts)...", flush=True)

    # Pre-cache references & 35k predicted durations.
    ref_cache = {}
    pred_durs_35k = {}
    fps = cfg.sample_rate / cfg.hop
    for row in selected:
        uid = row["id"]
        item = ds[id_to_idx[uid]]
        ref_wave = item["wave"].numpy()
        ref_durs = item["durations"].numpy()
        tokens = item["tokens"].numpy()
        t_ref = int(ref_durs.sum())
        ref_f0 = extract_unbiased_f0(ref_wave, cfg, frames=t_ref)
        ref_mel = spectral.log_mel(torch.from_numpy(ref_wave)[None])[0].numpy()
        ref_phone = np.repeat([SYMBOLS[int(t)] in PHONE_SET for t in tokens], ref_durs)[:t_ref]
        r_rel = ref_f0["voiced"] & ref_f0["trusted"] & ref_phone
        ref_cache[uid] = {
            "row": row,
            "item": item,
            "wave": ref_wave,
            "f0": ref_f0,
            "mel": ref_mel,
            "ref_durs": ref_durs,
            "tokens": tokens,
            "ref_phone": ref_phone,
            "r_rel": r_rel,
            "r_cont_st": ref_f0["continuous_log_f0"] * SEMITONES,
            "target_log_f0": item["prosody"][:, 0].numpy(),
        }
        with torch.inference_mode():
            tok = item["tokens"][None]
            st = item["stress"][None]
            tl = torch.tensor([tok.shape[1]], dtype=torch.long)
            spk = torch.tensor([int(item["speaker"])], dtype=torch.long)
            hid, msk = host_35k.encode(tok, st, tl, spk)
            pred_durs_35k[uid] = host_35k.predict_durations(hid, msk, tok)

    # --- PART 1: Host fr_ref_35k F0 injection on identical 35k predicted durations ---
    inj_names = ("35k_host", "e06_8k", "C", "D", "P")
    inj_rows: dict[str, list[dict]] = {k: [] for k in inj_names}
    inj_masks: dict[str, dict[str, np.ndarray]] = {k: {} for k in inj_names}
    inj_curves: dict[str, dict[str, np.ndarray]] = {k: {} for k in inj_names}

    for i_u, row in enumerate(selected):
        uid = row["id"]
        rc = ref_cache[uid]
        item = rc["item"]
        dur_35k = pred_durs_35k[uid]
        tok, st = item["tokens"][None], item["stress"][None]
        tl = torch.tensor([tok.shape[1]], dtype=torch.long)
        spk = torch.tensor([int(item["speaker"])], dtype=torch.long)
        syn_seed = 7000 + i_u

        for name in inj_names:
            donor_f0 = None if name == "35k_host" else predict_donor_log_f0(donors[name], tok, st, tl, spk, dur_35k)
            wav, meta = synthesize_decoupled_f0(host_35k, item, dur_35k, dec_log_f0=donor_f0, osc_log_f0=donor_f0, seed=syn_seed)
            m, g_rel_w, g_st_w = _heard_f0_metrics(
                rc["wave"], rc["mel"], rc["f0"], rc["target_log_f0"], rc["ref_durs"],
                wav, meta["durations"], meta["osc_log_f0"], rc["tokens"], cfg, spectral, "cpu",
            )
            inj_rows[name].append({
                "id": uid, "text_hash": row["text_hash"], "speaker": row["speaker"], "category": row["category"], **m,
            })
            inj_masks[name][uid] = g_rel_w
            inj_curves[name][uid] = g_st_w

    inj_common_cov = _apply_common_mask(inj_rows, inj_masks, inj_curves, ref_cache, list(inj_names), fps)
    inj_summary = {
        k: {
            "all": _summarize(inj_rows[k]),
            "F1": _summarize(inj_rows[k], lambda r: r["speaker"] == "F1"),
            "M1": _summarize(inj_rows[k], lambda r: r["speaker"] == "M1"),
        }
        for k in inj_names
    }
    inj_contrasts = {}
    inj_metrics = ("common_mask_f0_corr", "common_mask_f0_mae_st", "common_mask_slope_mae_st_s", "f0_mae_st", "mel_dtw")
    for cname, cmap in (
        ("D_minus_C", {"D": 1.0, "C": -1.0}),
        ("P_minus_C", {"P": 1.0, "C": -1.0}),
        ("P_minus_D", {"P": 1.0, "D": -1.0}),
        ("P_minus_e06_8k", {"P": 1.0, "e06_8k": -1.0}),
        ("C_minus_e06_8k", {"C": 1.0, "e06_8k": -1.0}),
        ("P_minus_35k_host", {"P": 1.0, "35k_host": -1.0}),
    ):
        inj_contrasts[cname] = {}
        for spk in ("all", "F1", "M1"):
            sf = None if spk == "all" else spk
            inj_contrasts[cname][spk] = {
                mk: stratified_text_bootstrap_ci(inj_rows, cmap, mk, speaker_filter=sf, n_boot=args.n_boot, seed=args.seed)
                for mk in inj_metrics
            }

    # --- PART 2: Native WAV evaluation for C, D, P (Mel-DTW, STFT, WER/CER) ---
    native_arms = ("C", "D", "P")
    native_rows: dict[str, list[dict]] = {k: [] for k in native_arms}
    asr_batches: dict[str, list[tuple[str, str, str, np.ndarray]]] = {k: [] for k in native_arms}

    for name in native_arms:
        model = donors[name]
        for i_u, row in enumerate(selected):
            uid = row["id"]
            rc = ref_cache[uid]
            item = rc["item"]
            gen_free, meta_free = oracle_synthesize(model, row, item, "A_free", seed=8000 + i_u)
            gen_b, _ = oracle_synthesize(model, row, item, "B_oracle_dur", seed=8000 + i_u)

            gen_mel = spectral.log_mel(torch.from_numpy(gen_free)[None])[0].numpy()
            mel_dtw = float(dtw_distance(rc["mel"], gen_mel))
            stft_b, mel_b = spectral(
                torch.from_numpy(gen_b)[None],
                torch.from_numpy(rc["wave"])[None],
                torch.tensor([len(rc["wave"])], dtype=torch.long),
            )
            int_m = compute_intonation_metrics(rc["wave"], gen_free, rc["ref_durs"], meta_free["durations"], rc["tokens"], cfg, ref_f0_info=rc["f0"])
            rhy_m = compute_rhythm_metrics(rc["ref_durs"], meta_free["durations"], rc["tokens"], cfg)
            native_rows[name].append({
                "id": uid,
                "text_hash": row["text_hash"],
                "speaker": row["speaker"],
                "category": row["category"],
                "mel_dtw": round(mel_dtw, 5),
                "oracle_dur_stft": round(float(stft_b.item()), 5),
                "oracle_dur_mel": round(float(mel_b.item()), 5),
                "f0_mae_st": int_m.get("f0_mae_st"),
                "f0_corr": int_m.get("f0_corr"),
                "f0_slope_mae_st_s": int_m.get("f0_slope_mae_st_s"),
                "phone_mae_ms": rhy_m.get("phone_mae_ms"),
            })
            asr_batches[name].append((uid, row["speaker"], row["text"], gen_free))

    asr_summary = {}
    if not args.skip_asr:
        from faster_whisper import WhisperModel
        import jiwer
        import librosa

        print("Loading faster-whisper small (int8 CPU) once for native C, D, P...", flush=True)
        whisper = WhisperModel("small", device="cpu", compute_type="int8")
        for name in native_arms:
            print(f"Running ASR on native {name} ({len(asr_batches[name])} WAVs)...", flush=True)
            by_spk_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
            per_u = {}
            for uid, spk, ref_text, wav in asr_batches[name]:
                wav16 = librosa.resample(wav.astype(np.float32), orig_sr=cfg.sample_rate, target_sr=16000)
                segs, _ = whisper.transcribe(wav16, language="fr", beam_size=5, vad_filter=False)
                hyp = " ".join(s.text.strip() for s in segs).strip()
                r_norm = normalize_for_wer(ref_text, language="fr")
                h_norm = normalize_for_wer(hyp, language="fr")
                u_wer = float(jiwer.wer([r_norm], [h_norm])) if r_norm else 0.0
                u_cer = float(jiwer.cer([r_norm], [h_norm])) if r_norm else 0.0
                per_u[uid] = {"wer": round(u_wer, 5), "cer": round(u_cer, 5), "hyp": hyp}
                by_spk_pairs[spk].append((r_norm, h_norm))
                by_spk_pairs["all"].append((r_norm, h_norm))
            asr_summary[name] = {
                grp: {
                    "samples": len(pairs),
                    "corpus_wer": round(float(jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs])), 4),
                    "corpus_cer": round(float(jiwer.cer([p[0] for p in pairs], [p[1] for p in pairs])), 4),
                }
                for grp, pairs in by_spk_pairs.items()
            }
            for r in native_rows[name]:
                if r["id"] in per_u:
                    r["wer"] = per_u[r["id"]]["wer"]
                    r["cer"] = per_u[r["id"]]["cer"]

    native_summary = {}
    nat_keys = ("mel_dtw", "oracle_dur_stft", "oracle_dur_mel", "f0_mae_st", "f0_corr", "f0_slope_mae_st_s", "phone_mae_ms", "wer", "cer")
    for name in native_arms:
        native_summary[name] = {}
        for spk in ("all", "F1", "M1"):
            sub = [r for r in native_rows[name] if spk == "all" or r["speaker"] == spk]
            native_summary[name][spk] = {"n": len(sub), **{k: _mean_opt([r.get(k) for r in sub]) for k in nat_keys}}

    native_contrasts = {}
    for cname, cmap in (("D_minus_C", {"D": 1.0, "C": -1.0}), ("P_minus_C", {"P": 1.0, "C": -1.0})):
        native_contrasts[cname] = {}
        for spk in ("all", "F1", "M1"):
            sf = None if spk == "all" else spk
            native_contrasts[cname][spk] = {
                mk: stratified_text_bootstrap_ci(native_rows, cmap, mk, speaker_filter=sf, n_boot=args.n_boot, seed=args.seed)
                for mk in ("mel_dtw", "oracle_dur_stft", "f0_mae_st", "f0_corr", "wer", "cer")
            }

    # --- PART 3: Native questions (totales vs partielles, validation.jsonl) ---
    questions_summary = None
    if not args.skip_questions and args.partition == "dev":
        print("Evaluating native C, D, P on paired validation.jsonl questions (totales vs partielles)...", flush=True)
        val_ds = AnnotatedDataset(args.annotations / "validation.jsonl", cfg, voiced_vowels=True)
        questions_summary = _eval_questions_native({k: donors[k] for k in native_arms}, val_ds, cfg)

    # --- PART 4: Pre-registered gate evaluation (P vs C) ---
    pc_inj = inj_contrasts["P_minus_C"]["all"]
    pc_nat = native_contrasts["P_minus_C"]["all"]
    dc_inj = inj_contrasts["D_minus_C"]["all"]

    gates = {
        "primary_P_corr_ci_positive": bool(pc_inj["common_mask_f0_corr"] and pc_inj["common_mask_f0_corr"]["ci95_low"] > PREREG_MARGINS["primary_corr_ci95_low_min"]),
        "primary_P_mae_within_margin": bool(pc_inj["common_mask_f0_mae_st"] and pc_inj["common_mask_f0_mae_st"]["mean_delta"] <= PREREG_MARGINS["primary_mae_max_delta_st"]),
        "native_mel_dtw_within_margin": bool(
            pc_nat["mel_dtw"]
            and pc_nat["mel_dtw"]["mean_delta"] <= PREREG_MARGINS["native_mel_dtw_max_delta"]
            and pc_nat["mel_dtw"]["ci95_high"] <= PREREG_MARGINS["native_mel_dtw_ci95_hi_max"]
        ),
        "native_stft_within_margin": bool(pc_nat["oracle_dur_stft"] and pc_nat["oracle_dur_stft"]["mean_delta"] <= PREREG_MARGINS["native_stft_max_delta"]),
        "native_wer_within_margin": None if args.skip_asr else bool(pc_nat["wer"] and pc_nat["wer"]["mean_delta"] <= PREREG_MARGINS["native_wer_max_delta"]),
        "native_cer_within_margin": None if args.skip_asr else bool(pc_nat["cer"] and pc_nat["cer"]["mean_delta"] <= PREREG_MARGINS["native_cer_max_delta"]),
        "question_totale_dir_acc_within_margin": None if questions_summary is None else bool(
            (questions_summary["P"]["totale"]["all"]["dir_match_acc"] - questions_summary["C"]["totale"]["all"]["dir_match_acc"]) >= -PREREG_MARGINS["question_totale_dir_acc_max_drop"]
        ),
        "D_beats_C_primary_corr": bool(dc_inj["common_mask_f0_corr"] and dc_inj["common_mask_f0_corr"]["ci95_low"] > 0.0),
    }
    required_p_gates = [
        gates["primary_P_corr_ci_positive"],
        gates["primary_P_mae_within_margin"],
        gates["native_mel_dtw_within_margin"],
        gates["native_stft_within_margin"],
    ]
    if gates["native_wer_within_margin"] is not None:
        required_p_gates.extend([gates["native_wer_within_margin"], gates["native_cer_within_margin"]])
    if gates["question_totale_dir_acc_within_margin"] is not None:
        required_p_gates.append(gates["question_totale_dir_acc_within_margin"])
    unlock_reserve121 = all(required_p_gates)

    report = {
        "partition": args.partition,
        "n_recordings": len(selected),
        "n_texts": len(selected) // 2,
        "checkpoints": ckpt_meta,
        "preregistered_margins": PREREG_MARGINS,
        "injected_common_mask_coverage": round(inj_common_cov, 5),
        "injected_summary": inj_summary,
        "injected_contrasts_95ci": inj_contrasts,
        "native_summary": native_summary,
        "native_asr_summary": asr_summary,
        "native_contrasts_95ci": native_contrasts,
        "native_questions_validation": questions_summary,
        "gate_results": gates,
        "unlock_reserve121_confirmation": unlock_reserve121,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"p6_eval_{args.partition}_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / f"p6_eval_{args.partition}_per_recording.json").write_text(
        json.dumps({"injected": inj_rows, "native": native_rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"\n=== P6 EVALUATION SUMMARY ({args.partition}, common_mask_cov={inj_common_cov:.4f}) ===")
    for k in inj_names:
        d = inj_summary[k]["all"]
        print(f"  [INJ->35k] {k:10s}: cm_r={d['common_mask_f0_corr']} cm_mae={d['common_mask_f0_mae_st']} cm_slope={d['common_mask_slope_mae_st_s']}")
    for cname in ("D_minus_C", "P_minus_C"):
        c = inj_contrasts[cname]["all"]
        print(f"  [INJ CONTRAST] {cname:10s}: " + " | ".join(
            f"{mk}={c[mk]['mean_delta']:+.4f}[{c[mk]['ci95_low']:+.4f},{c[mk]['ci95_high']:+.4f}]{'*' if c[mk]['significant'] else ''}"
            for mk in ("common_mask_f0_corr", "common_mask_f0_mae_st", "common_mask_slope_mae_st_s")
        ))
    for k in native_arms:
        d = native_summary[k]["all"]
        print(f"  [NATIVE]   {k:10s}: mel_dtw={d['mel_dtw']} stft={d['oracle_dur_stft']} wer={d['wer']} cer={d['cer']} f0_mae={d['f0_mae_st']} f0_r={d['f0_corr']}")
    for cname in ("D_minus_C", "P_minus_C"):
        c = native_contrasts[cname]["all"]
        parts = []
        for mk in ("mel_dtw", "oracle_dur_stft", "wer", "cer"):
            if c.get(mk):
                parts.append(f"{mk}={c[mk]['mean_delta']:+.4f}[{c[mk]['ci95_low']:+.4f},{c[mk]['ci95_high']:+.4f}]{'*' if c[mk]['significant'] else ''}")
        print(f"  [NAT CONTRAST] {cname:10s}: " + " | ".join(parts))
    print("GATES:", json.dumps(gates, indent=2))
    print("UNLOCK RESERVE 121:", unlock_reserve121)


if __name__ == "__main__":
    main()
