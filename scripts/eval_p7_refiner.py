#!/usr/bin/env python3
"""Offline CPU evaluator & blind A/B kit builder for P7 FrenchProsodyRefiner.

Evaluates 4 cleanly separated systems on top of frozen fr_ref_35k:
  - base_35k:         exact fr_ref_35k baseline (sp_thr=0.50)
  - base_35k_sp_cal:  fr_ref_35k + calibrated <sp> pause threshold ONLY (0 extra params)
  - p7_rhythm_only:   fr_ref_35k + P7 Stage A Rhythm/Pause refiner (host F0/energy)
  - p7_full:          fr_ref_35k + P7 Stage A Rhythm/Pause + Stage B Expressive Prosody

Metrics (100% CPU, 0 GPU, 121 virgin test pairs strictly UNTOUCHED):
  1. Full validation.jsonl (373 utts) rhythm & pause audit (<sp> P/R/FP/FN, punct P/R,
     phone_mae_ms, total_dur_err_pct) + paired French questions (totales vs partielles
     by voice F1/M1).
  2. Held-out P4-dev (40 utts / 20 texts, excluded from val_cal) full WAV evaluation:
     Mel-DTW, oracle-dur STFT, WAV F0 corr / MAE / slope MAE / local std ratio,
     faster-whisper small WER & CER, with stratified text-grouped paired 95% CIs.
  3. Optional blind A/B listening kit builder (randomized 6/6 A/B pairs from
     validation.jsonl — never touching test.jsonl — with key stored outside the
     served directory for scripts/blind_ab_ui.py).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys
import wave

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
from tiny_voca.model import TinyVoca
from tiny_voca.prosody_fr import FrenchProsodyRefiner, synthesize_with_refiner
from tiny_voca.text import PHONE_SET, SYMBOL_TO_ID, SYMBOLS

from eval_p4_ablation import stratified_text_bootstrap_ci
from eval_p5_f0_causal import _direction_label, _peak_syllable_stats, classify_french_question, sha256_file
from p0_expressivity import (
    SEMITONES,
    _sentence_category,
    _terminal_inflection_st,
    compute_intonation_metrics,
    compute_rhythm_metrics,
    extract_unbiased_f0,
)

_SP_ID = SYMBOL_TO_ID["<sp>"]
_BOS_ID = SYMBOL_TO_ID["<bos>"]
_EOS_ID = SYMBOL_TO_ID["<eos>"]


def _mean_opt(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return round(float(np.mean(xs)), 5) if xs else None


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(audio.astype(np.float32), -1.0, 1.0)
    pcm = (x * 32767.0).round().astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


@torch.inference_mode()
def _synth_variant(
    host: TinyVoca,
    refiner: FrenchProsodyRefiner,
    row: dict,
    item: dict,
    variant: str,
    host_sp_thr: float,
    seed: int,
    oracle_dur: bool = False,
) -> tuple[np.ndarray, dict]:
    tok = item["tokens"][None]
    st = item["stress"][None]
    tl = torch.tensor([tok.shape[1]], dtype=torch.long)
    spk = torch.tensor([int(item["speaker"])], dtype=torch.long)
    texts = [row["text"]]

    if oracle_dur:
        # Oracle durations (for fair oracle-dur STFT comparison).
        hidden, mask = host.encode(tok, st, tl, spk)
        durations = item["durations"][None]
        frames, frame_mask, frame_lengths = host.expand(hidden, durations, mask, spk)
        pro_h, host_pro = host.predict_prosody(frames, frame_mask)
        if variant in ("p7_full", "p7_sp_cal_prosody"):
            from tiny_voca.prosody_fr import extract_french_phrasal_features
            ph = extract_french_phrasal_features(item["tokens"], row["text"])[None]
            exp_ph, fpos = FrenchProsodyRefiner.expand_token_features(ph, durations, frame_lengths)
            prosody = refiner.refine_prosody(pro_h.float(), host_pro.float(), exp_ph, fpos, frame_mask, spk).to(host_pro.dtype)
        else:
            prosody = host_pro
        # Match TinyVoca.synthesize voiced_vowels=True
        ends = durations.cumsum(1)
        frame = torch.arange(frames.shape[1])[None]
        owner = torch.searchsorted(ends, frame, right=True).clamp(max=tok.shape[1] - 1)
        vowel = host.vowel_tokens[tok].gather(1, owner) & frame_mask
        prosody = torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None], prosody[..., 2:]), dim=-1)
        acoustic = host.decode(pro_h, host.hard_prosody(prosody), frame_mask)
        noise = torch.randn(1, int(frame_lengths[0]) * host.config.hop, generator=torch.Generator().manual_seed(seed))
        wav = host.vocoder(acoustic, host.frequency(prosody[..., 0]), frame_lengths, noise=noise)
        return wav[0, : int(frame_lengths[0]) * host.config.hop].numpy(), {"durations": durations[0].numpy()}

    if variant == "base_35k":
        wav, slen, meta = synthesize_with_refiner(
            host, None, tok, st, tl, spk, texts=texts, seed=seed,
            use_rhythm=False, use_prosody=False, sp_pause_threshold=0.50,
        )
    elif variant == "base_35k_sp_cal":
        wav, slen, meta = synthesize_with_refiner(
            host, None, tok, st, tl, spk, texts=texts, seed=seed,
            use_rhythm=False, use_prosody=False, sp_pause_threshold=host_sp_thr,
        )
    elif variant == "p7_sp_cal_prosody":
        wav, slen, meta = synthesize_with_refiner(
            host, refiner, tok, st, tl, spk, texts=texts, seed=seed,
            use_rhythm=False, use_prosody=True, sp_pause_threshold=host_sp_thr,
        )
    elif variant == "p7_rhythm_only":
        wav, slen, meta = synthesize_with_refiner(
            host, refiner, tok, st, tl, spk, texts=texts, seed=seed,
            use_rhythm=True, use_prosody=False,
        )
    elif variant == "p7_full":
        wav, slen, meta = synthesize_with_refiner(
            host, refiner, tok, st, tl, spk, texts=texts, seed=seed,
            use_rhythm=True, use_prosody=True,
        )
    else:
        raise ValueError(f"unknown variant {variant}")
    n_s = int(slen[0].item())
    return wav[0, :n_s].numpy(), {"durations": meta["durations"][0].numpy(), "dur_out": meta["dur_out"][0].numpy()}


@torch.inference_mode()
def audit_validation_pauses(
    host: TinyVoca,
    refiner: FrenchProsodyRefiner,
    val_ds: AnnotatedDataset,
    host_sp_thr: float,
    variants: tuple[str, ...],
) -> dict:
    """Fast token-level pause & duration audit across all 373 validation.jsonl utterances."""
    from tiny_voca.prosody_fr import extract_french_phrasal_features
    cfg = host.config
    ms_per_frame = cfg.hop * 1000.0 / cfg.sample_rate
    phone_mask = host.phone_tokens

    stats = {v: {
        "sp_tp": 0, "sp_fp": 0, "sp_fn": 0, "sp_fp_ms": [],
        "pu_tp": 0, "pu_fp": 0, "pu_fn": 0,
        "phone_abs_ms": [], "tot_err_pct": [],
    } for v in variants}

    for idx in range(len(val_ds)):
        row = val_ds.rows[idx]
        item = val_ds[idx]
        tok = item["tokens"][None]
        st = item["stress"][None]
        tl = torch.tensor([tok.shape[1]], dtype=torch.long)
        spk = torch.tensor([int(item["speaker"])], dtype=torch.long)
        gt_d = item["durations"]

        hidden, mask = host.encode(tok, st, tl, spk)
        host_dur_out = host.duration(hidden, mask).float()
        phr = extract_french_phrasal_features(item["tokens"], row["text"])[None]
        p7_dur_out = refiner.refine_durations(hidden.float(), host_dur_out, phr, mask, spk)

        is_phone = phone_mask[tok[0]]
        is_edge = (tok[0] == _BOS_ID) | (tok[0] == _EOS_ID)
        is_sp = tok[0] == _SP_ID
        is_pu = (~is_phone) & (~is_edge) & (~is_sp)
        internal_gt = float(gt_d[~is_edge].sum().item())

        for v in variants:
            if v == "base_35k":
                d_out, sp_thr = host_dur_out, 0.50
            elif v in ("base_35k_sp_cal", "p7_sp_cal_prosody"):
                d_out, sp_thr = host_dur_out, host_sp_thr
            else:
                d_out, sp_thr = p7_dur_out, float(refiner.sp_pause_threshold.item())
            pred_d = FrenchProsodyRefiner.decode_durations(
                d_out, tok, phone_mask, mask, sp_pause_threshold=sp_thr, punct_pause_threshold=0.50
            )[0]

            s = stats[v]
            # <sp> pauses
            sp_pr = pred_d[is_sp] > 0
            sp_gt = gt_d[is_sp] > 0
            s["sp_tp"] += int((sp_pr & sp_gt).sum().item())
            s["sp_fp"] += int((sp_pr & ~sp_gt).sum().item())
            s["sp_fn"] += int((~sp_pr & sp_gt).sum().item())
            fp_lens = pred_d[is_sp][sp_pr & ~sp_gt].float() * ms_per_frame
            s["sp_fp_ms"].extend(fp_lens.tolist())

            # Punctuation pauses
            pu_pr = pred_d[is_pu] > 0
            pu_gt = gt_d[is_pu] > 0
            s["pu_tp"] += int((pu_pr & pu_gt).sum().item())
            s["pu_fp"] += int((pu_pr & ~pu_gt).sum().item())
            s["pu_fn"] += int((~pu_pr & pu_gt).sum().item())

            s["phone_abs_ms"].extend(((pred_d[is_phone] - gt_d[is_phone]).abs().float() * ms_per_frame).tolist())
            internal_pr = float(pred_d[~is_edge].sum().item())
            if internal_gt > 0:
                s["tot_err_pct"].append(abs(internal_pr - internal_gt) / internal_gt * 100.0)

    out = {}
    for v, s in stats.items():
        sp_p = s["sp_tp"] / max(s["sp_tp"] + s["sp_fp"], 1)
        sp_r = s["sp_tp"] / max(s["sp_tp"] + s["sp_fn"], 1)
        sp_f1 = 2 * sp_p * sp_r / max(sp_p + sp_r, 1e-8)
        sp_f05 = 1.25 * sp_p * sp_r / max(0.25 * sp_p + sp_r, 1e-8)
        pu_p = s["pu_tp"] / max(s["pu_tp"] + s["pu_fp"], 1)
        pu_r = s["pu_tp"] / max(s["pu_tp"] + s["pu_fn"], 1)
        out[v] = {
            "sp_tp": s["sp_tp"],
            "sp_fp": s["sp_fp"],
            "sp_fn": s["sp_fn"],
            "sp_prec": round(sp_p, 4),
            "sp_rec": round(sp_r, 4),
            "sp_f1": round(sp_f1, 4),
            "sp_f05": round(sp_f05, 4),
            "sp_fp_mean_ms": round(float(np.mean(s["sp_fp_ms"])), 1) if s["sp_fp_ms"] else 0.0,
            "pu_tp": s["pu_tp"],
            "pu_fp": s["pu_fp"],
            "pu_fn": s["pu_fn"],
            "pu_prec": round(pu_p, 4),
            "pu_rec": round(pu_r, 4),
            "phone_mae_ms": round(float(np.mean(s["phone_abs_ms"])), 3),
            "tot_dur_err_pct": round(float(np.mean(s["tot_err_pct"])), 3),
        }
    return out


def eval_questions_validation(
    host: TinyVoca,
    refiner: FrenchProsodyRefiner,
    val_ds: AnnotatedDataset,
    host_sp_thr: float,
    variants: tuple[str, ...],
) -> dict:
    cfg = host.config
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
            "ref_end": ref_end,
            "ref_dir": _direction_label(ref_end),
            "ref_pk": ref_pk,
            "item": item,
            "row": row,
        }

    out = {}
    for v in variants:
        recs = []
        for k_i, idx in enumerate(paired_indices):
            row = val_ds.rows[idx]
            rc = ref_cache[row["id"]]
            gen_wave, meta = _synth_variant(host, refiner, row, rc["item"], v, host_sp_thr, seed=2000 + k_i)
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
        out[v] = arm_summary
    return out


def build_blind_ab_kit(
    host: TinyVoca,
    refiner: FrenchProsodyRefiner,
    val_ds: AnnotatedDataset,
    host_sp_thr: float,
    candidate_variant: str,
    blind_root: Path,
    seed: int = 20261119,
) -> Path:
    """Build a 12-pair blind A/B kit (candidate vs fr_ref_35k) from validation.jsonl ONLY.

    Key is saved OUTSIDE the served `blind_p7` directory so scripts/blind_ab_ui.py
    remains strictly blind.
    """
    served_dir = blind_root / "blind_p7"
    if served_dir.exists():
        import shutil
        shutil.rmtree(served_dir)
    served_dir.mkdir(parents=True, exist_ok=True)
    id_to_idx = {r["id"]: i for i, r in enumerate(val_ds.rows)}

    # Hand-picked high-signal validation.jsonl utterances covering:
    # - Known false-<sp> pause bugs ("jeunes et premières", "vendredi ou lundi", "Lina, Claire et Jade", "Les regardant tous")
    # - Phrase-final lengthening ("Signes terrifiants vaguement aperçus !")
    # - French questions totales & partielles (both F1 and M1)
    # - Exclamation & comma phrasing (both F1 and M1)
    target_substrings = [
        ("jeunes et premières", "F1", "false_sp_pause_excl"),
        ("vendredi ou lundi", "M1", "false_sp_pause_q_totale"),
        ("Lina, Claire et Jade", "F1", "false_sp_pause_comma"),
        ("Les regardant tous en face", "M1", "false_sp_pause_decl"),
        ("Signes terrifiants", "F1", "final_lengthening_excl"),
        ("Que peut-il me vouloir", "M1", "q_partielle"),
    ]
    chosen_ids: list[tuple[str, str]] = []
    used_hashes: set[str] = set()
    for sub, spk, tag in target_substrings:
        for r in val_ds.rows:
            if sub in r["text"] and r["speaker"] == spk and r["text_hash"] not in used_hashes:
                chosen_ids.append((r["id"], tag))
                used_hashes.add(r["text_hash"])
                break

    # Fill remaining up to 12 balanced across F1/M1 and q_totale / q_partielle / exclamation / comma.
    want_slots = [
        ("totale", "F1"),
        ("totale", "M1"),
        ("partielle", "F1"),
        ("exclamation", "M1"),
        ("comma", "M1"),
        ("declarative", "F1"),
    ]
    for cat, spk in want_slots:
        if len(chosen_ids) >= 12:
            break
        for r in val_ds.rows:
            if r["speaker"] != spk or r["text_hash"] in used_hashes:
                continue
            txt = r["text"]
            if len(txt) < 28 or len(txt) > 105:
                continue
            match = False
            if cat in ("totale", "partielle") and "?" in txt and classify_french_question(txt) == cat:
                match = True
            elif cat == "exclamation" and "!" in txt:
                match = True
            elif cat == "comma" and "," in txt and "?" not in txt and "!" not in txt:
                match = True
            elif cat == "declarative" and "?" not in txt and "!" not in txt and "," not in txt:
                match = True
            if match:
                chosen_ids.append((r["id"], f"{cat}_{spk}"))
                used_hashes.add(r["text_hash"])
                break

    rng = random.Random(seed)
    # Exactly 6 A=candidate and 6 A=base_35k, shuffled.
    n = len(chosen_ids)
    a_is_cand = [True] * (n // 2) + [False] * (n - n // 2)
    rng.shuffle(a_is_cand)

    key_rows = []
    manifest_lines = [
        "# Écoute aveugle P7 — 12 paires (A vs B)",
        "",
        "Chaque paire compare deux versions de la même phrase (A et B mélangés aléatoirement, 6/6).",
        "Écoutez dans l'interface web locale (`scripts/blind_ab_ui.py`) et notez votre préférence.",
        "",
        "| Paire | Voix | Catégorie | Texte |",
        "|---|---|---|---|",
    ]

    for idx_p, ((uid, tag), cand_on_a) in enumerate(zip(chosen_ids, a_is_cand), start=1):
        ds_i = id_to_idx[uid]
        row = val_ds.rows[ds_i]
        item = val_ds[ds_i]
        wav_base, meta_base = _synth_variant(host, refiner, row, item, "base_35k", host_sp_thr, seed=9000 + idx_p)
        wav_cand, meta_cand = _synth_variant(host, refiner, row, item, candidate_variant, host_sp_thr, seed=9000 + idx_p)

        pid = f"P{idx_p:02d}"
        wav_a, wav_b = (wav_cand, wav_base) if cand_on_a else (wav_base, wav_cand)
        _write_wav(served_dir / f"{pid}_A.wav", wav_a, host.config.sample_rate)
        _write_wav(served_dir / f"{pid}_B.wav", wav_b, host.config.sample_rate)
        cat_label = _sentence_category(row["text"])
        safe_txt = row["text"].replace("|", "/")
        manifest_lines.append(f"| `{pid}` | {row['speaker']} | {cat_label} | {safe_txt} |")
        key_rows.append({
            "pair": idx_p,
            "pair_id": pid,
            "id": uid,
            "speaker": row["speaker"],
            "category": cat_label,
            "text": row["text"],
            "tag": tag,
            "A": candidate_variant if cand_on_a else "base_35k",
            "B": "base_35k" if cand_on_a else candidate_variant,
            "base_dur_s": round(len(wav_base) / host.config.sample_rate, 3),
            "cand_dur_s": round(len(wav_cand) / host.config.sample_rate, 3),
        })

    (served_dir / "FICHE.md").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    # Save secret key ONE LEVEL UP (never inside served_dir!)
    secret_path = blind_root / "p7_blind_secret_key.json"
    secret_path.write_text(json.dumps({
        "candidate_variant": candidate_variant,
        "baseline": "base_35k",
        "seed": seed,
        "pairs": key_rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Built {len(key_rows)}-pair blind A/B kit at {served_dir} (secret key: {secret_path})")
    return served_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=Path, default=Path(".local-runs/colab/fr_ref_35k/fr_ref_35k_final/inference.pt"))
    ap.add_argument("--refiner", type=Path, default=Path(".local-runs/p7_refiner/p7_refiner.pt"))
    ap.add_argument("--annotations", type=Path, default=Path(".local-runs/colab/fr4k/annotations_e01"))
    ap.add_argument("--p4-dev-report", type=Path, default=Path(".local-runs/p4_ablation/p4_final/dev_screen/p4_dev_report.json"))
    ap.add_argument("--out-dir", type=Path, default=Path(".local-runs/p7_refiner/eval_dev"))
    ap.add_argument("--skip-asr", action="store_true")
    ap.add_argument("--skip-questions", action="store_true")
    ap.add_argument("--make-blind-variant", default="auto", help="auto | p7_full | p7_rhythm_only | none")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261119)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    host_ckpt = torch.load(args.host, map_location="cpu", weights_only=False)
    cfg = Config(**host_ckpt["config"]) if isinstance(host_ckpt["config"], dict) else host_ckpt["config"]
    host = TinyVoca(cfg).eval()
    host.load_state_dict(host_ckpt["model"])

    ref_ckpt = torch.load(args.refiner, map_location="cpu", weights_only=False)
    refiner = FrenchProsodyRefiner(**ref_ckpt["refiner_config"]).eval()
    refiner.load_state_dict(ref_ckpt["refiner_state"])
    host_sp_thr = float(ref_ckpt.get("host_best_sp_threshold", 0.90))

    val_manifest = args.annotations / "validation.jsonl"
    val_ds = AnnotatedDataset(val_manifest, cfg, voiced_vowels=True)
    id_to_idx = {r["id"]: i for i, r in enumerate(val_ds.rows)}
    dev_meta = json.loads(args.p4_dev_report.read_text(encoding="utf-8"))
    selected_ids = list(dev_meta["selected_ids"])
    selected = []
    for uid in selected_ids:
        r = dict(val_ds.rows[id_to_idx[uid]])
        r["category"] = _sentence_category(r["text"])
        selected.append(r)

    variants = ("base_35k", "base_35k_sp_cal", "p7_sp_cal_prosody", "p7_rhythm_only", "p7_full")
    print(f"=== 1/3 Full validation.jsonl ({len(val_ds)} utts) Pause & Rhythm Audit ===", flush=True)
    pause_audit = audit_validation_pauses(host, refiner, val_ds, host_sp_thr, variants)
    for v in variants:
        pa = pause_audit[v]
        print(
            f"  {v:16s} | <sp> TP={pa['sp_tp']:2d} FP={pa['sp_fp']:2d} FN={pa['sp_fn']:2d} "
            f"(P={pa['sp_prec']:.1%}, R={pa['sp_rec']:.1%}, F0.5={pa['sp_f05']:.3f}, fp_ms={pa['sp_fp_mean_ms']:.0f}) | "
            f"punct P={pa['pu_prec']:.1%} R={pa['pu_rec']:.1%} | phone_mae={pa['phone_mae_ms']:.2f}ms | tot_err={pa['tot_dur_err_pct']:.2f}%",
            flush=True,
        )

    print(f"\n=== 2/3 Held-out P4-dev ({len(selected)} utts / {len(selected)//2} texts) Full WAV Eval ===", flush=True)
    spectral = SpectralLoss(cfg.sample_rate)
    ref_cache = {}
    for row in selected:
        uid = row["id"]
        item = val_ds[id_to_idx[uid]]
        ref_wave = item["wave"].numpy()
        ref_durs = item["durations"].numpy()
        tokens = item["tokens"].numpy()
        t_ref = int(ref_durs.sum())
        ref_f0 = extract_unbiased_f0(ref_wave, cfg, frames=t_ref)
        ref_mel = spectral.log_mel(torch.from_numpy(ref_wave)[None])[0].numpy()
        ref_cache[uid] = {
            "row": row, "item": item, "wave": ref_wave, "f0": ref_f0,
            "mel": ref_mel, "ref_durs": ref_durs, "tokens": tokens,
        }

    dev_rows: dict[str, list[dict]] = {v: [] for v in variants}
    asr_batches: dict[str, list[tuple[str, str, str, np.ndarray]]] = {v: [] for v in variants}

    for v in variants:
        for i_u, row in enumerate(selected):
            uid = row["id"]
            rc = ref_cache[uid]
            gen_free, meta_free = _synth_variant(host, refiner, row, rc["item"], v, host_sp_thr, seed=8000 + i_u, oracle_dur=False)
            gen_ora, _ = _synth_variant(host, refiner, row, rc["item"], v, host_sp_thr, seed=8000 + i_u, oracle_dur=True)

            gen_mel = spectral.log_mel(torch.from_numpy(gen_free)[None])[0].numpy()
            mel_dtw = float(dtw_distance(rc["mel"], gen_mel))
            stft_b, mel_b = spectral(
                torch.from_numpy(gen_ora)[None],
                torch.from_numpy(rc["wave"])[None],
                torch.tensor([len(rc["wave"])], dtype=torch.long),
            )
            int_m = compute_intonation_metrics(rc["wave"], gen_free, rc["ref_durs"], meta_free["durations"], rc["tokens"], cfg, ref_f0_info=rc["f0"])
            rhy_m = compute_rhythm_metrics(rc["ref_durs"], meta_free["durations"], rc["tokens"], cfg)
            dev_rows[v].append({
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
                "f0_std_ratio": int_m.get("f0_std_ratio"),
                "phone_mae_ms": rhy_m.get("phone_mae_ms"),
            })
            asr_batches[v].append((uid, row["speaker"], row["text"], gen_free))

    asr_summary = {}
    if not args.skip_asr:
        from faster_whisper import WhisperModel
        import jiwer
        import librosa

        print("Loading faster-whisper small (int8 CPU) for P4-dev ASR...", flush=True)
        whisper = WhisperModel("small", device="cpu", compute_type="int8")
        for v in variants:
            by_spk_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
            per_u = {}
            for uid, spk, ref_text, wav in asr_batches[v]:
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
            asr_summary[v] = {
                grp: {
                    "samples": len(pairs),
                    "corpus_wer": round(float(jiwer.wer([p[0] for p in pairs], [p[1] for p in pairs])), 4),
                    "corpus_cer": round(float(jiwer.cer([p[0] for p in pairs], [p[1] for p in pairs])), 4),
                }
                for grp, pairs in by_spk_pairs.items()
            }
            for r in dev_rows[v]:
                if r["id"] in per_u:
                    r["wer"] = per_u[r["id"]]["wer"]
                    r["cer"] = per_u[r["id"]]["cer"]

    dev_keys = ("mel_dtw", "oracle_dur_stft", "oracle_dur_mel", "f0_mae_st", "f0_corr", "f0_slope_mae_st_s", "f0_std_ratio", "phone_mae_ms", "wer", "cer")
    dev_summary = {}
    for v in variants:
        dev_summary[v] = {}
        for spk in ("all", "F1", "M1"):
            sub = [r for r in dev_rows[v] if spk == "all" or r["speaker"] == spk]
            dev_summary[v][spk] = {"n": len(sub), **{k: _mean_opt([r.get(k) for r in sub]) for k in dev_keys}}
        s_all = dev_summary[v]["all"]
        wer_s = f"{asr_summary[v]['all']['corpus_wer']*100:.2f}%" if v in asr_summary else "n/a"
        cer_s = f"{asr_summary[v]['all']['corpus_cer']*100:.2f}%" if v in asr_summary else "n/a"
        print(
            f"  {v:16s} | Mel-DTW={s_all['mel_dtw']:.4f} | STFT={s_all['oracle_dur_stft']:.4f} | "
            f"F0_r={s_all['f0_corr']:.4f} MAE={s_all['f0_mae_st']:.3f}st slope={s_all['f0_slope_mae_st_s']:.1f} std_r={s_all['f0_std_ratio']:.3f} | "
            f"ph_mae={s_all['phone_mae_ms']:.2f}ms | WER={wer_s} CER={cer_s}",
            flush=True,
        )

    contrasts = {}
    for cname, cmap in (
        ("p7_sp_cal_prosody_minus_base", {"p7_sp_cal_prosody": 1.0, "base_35k": -1.0}),
        ("p7_sp_cal_prosody_minus_sp_cal", {"p7_sp_cal_prosody": 1.0, "base_35k_sp_cal": -1.0}),
        ("p7_full_minus_base", {"p7_full": 1.0, "base_35k": -1.0}),
        ("p7_rhythm_minus_base", {"p7_rhythm_only": 1.0, "base_35k": -1.0}),
        ("sp_cal_minus_base", {"base_35k_sp_cal": 1.0, "base_35k": -1.0}),
        ("p7_full_minus_rhythm", {"p7_full": 1.0, "p7_rhythm_only": -1.0}),
    ):
        contrasts[cname] = {}
        for spk in ("all", "F1", "M1"):
            sf = None if spk == "all" else spk
            contrasts[cname][spk] = {
                mk: stratified_text_bootstrap_ci(dev_rows, cmap, mk, speaker_filter=sf, n_boot=args.n_boot, seed=args.seed)
                for mk in ("mel_dtw", "oracle_dur_stft", "f0_corr", "f0_mae_st", "f0_slope_mae_st_s", "phone_mae_ms", "wer", "cer")
            }

    questions = None
    if not args.skip_questions:
        print("\n=== 3/3 Paired Questions (totales vs partielles, validation.jsonl) ===", flush=True)
        q_vars = ("base_35k", "base_35k_sp_cal", "p7_sp_cal_prosody", "p7_full")
        questions = eval_questions_validation(host, refiner, val_ds, host_sp_thr, q_vars)
        for v in q_vars:
            t_all = questions[v]["totale"]["all"]
            p_all = questions[v]["partielle"]["all"]
            print(
                f"  {v:18s} | totale (n={t_all['n']}): end={t_all['gen_end_mean_st']:+.2f}st (ref {t_all['ref_end_mean_st']:+.2f}) "
                f"err={t_all['end_abs_err_st']:.2f}st dirAcc={t_all['dir_match_acc']:.1%} | "
                f"partielle (n={p_all['n']}): end={p_all['gen_end_mean_st']:+.2f}st (ref {p_all['ref_end_mean_st']:+.2f}) "
                f"err={p_all['end_abs_err_st']:.2f}st dirAcc={p_all['dir_match_acc']:.1%}",
                flush=True,
            )

    # Pick winner for blind A/B kit (if requested)
    winner = args.make_blind_variant
    if winner == "auto":
        b0 = dev_summary["base_35k"]["all"]
        best_score = -1e9
        winner = "base_35k_sp_cal"
        for cand in ("p7_sp_cal_prosody", "p7_full", "base_35k_sp_cal", "p7_rhythm_only"):
            sc = dev_summary[cand]["all"]
            wer_ok = True
            if cand in asr_summary and "base_35k" in asr_summary:
                wer_ok = (asr_summary[cand]["all"]["corpus_wer"] - asr_summary["base_35k"]["all"]["corpus_wer"]) <= 0.010
            if not wer_ok:
                continue
            if sc["mel_dtw"] > b0["mel_dtw"] + 0.005 or sc["oracle_dur_stft"] > b0["oracle_dur_stft"] + 0.01:
                continue
            # Combined gain: F0 corr gain + F0 MAE gain + Mel-DTW gain
            score = (
                (sc["f0_corr"] - b0["f0_corr"]) * 10.0
                + (b0["f0_mae_st"] - sc["f0_mae_st"]) * 2.0
                + (b0["mel_dtw"] - sc["mel_dtw"]) * 5.0
                + (pause_audit[cand]["sp_f05"] - pause_audit["base_35k"]["sp_f05"]) * 2.0
            )
            if score > best_score:
                best_score = score
                winner = cand

    blind_dir = None
    if winner != "none":
        blind_dir = str(
            build_blind_ab_kit(
                host, refiner, val_ds, host_sp_thr, winner, args.out_dir.parent / "listening_p7", seed=args.seed
            )
        )

    report = {
        "host_path": str(args.host),
        "host_sha256": sha256_file(args.host),
        "refiner_path": str(args.refiner),
        "refiner_sha256": sha256_file(args.refiner),
        "host_params": sum(p.numel() for p in host.parameters()),
        "refiner_params": sum(p.numel() for p in refiner.parameters()),
        "p7_sp_pause_threshold": float(refiner.sp_pause_threshold.item()),
        "host_calibrated_sp_threshold": host_sp_thr,
        "validation_pause_audit": pause_audit,
        "p4_dev_summary": dev_summary,
        "p4_dev_asr": asr_summary,
        "p4_dev_contrasts_95ci": contrasts,
        "validation_questions": questions,
        "selected_blind_candidate": winner,
        "blind_kit_dir": blind_dir,
    }
    out_json = args.out_dir / "p7_eval_dev_summary.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out_dir / "p7_eval_dev_per_utterance.json").write_text(
        json.dumps(dev_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWrote P7 evaluation report to {out_json}", flush=True)


if __name__ == "__main__":
    main()
