"""P5 Frozen-weight F0 causal experiments (Items 2 & 3, 0 GPU):
  - 2A: 2x2 decoupled F0 source matrix (dec_f0 x osc_f0 in {35k, ref}) on fr_ref_35k
        with fixed oracle durations, noise, energy, voicing (verifies bit-exact match
        with P0 B_oracle_dur and C_oracle_dur_f0).
  - 2B: Pre-GAN 8k F0 trajectory (e06 step-8000, plus e06-20k & fr_ref-10k controls)
        injected into mature fr_ref_35k on oracle durations AND on identical 35k
        predicted durations.
  - 3A: Heard F0 audit per voice (pred vs ref, osc vs WAV realization, common-mask
        comparison, reliable coverage, voicing FP/FN).
  - 3B: Questions deep-dive (totales vs partielles, terminal inflection direction,
        peak-carrying syllable) on validation.jsonl (disjoint from test reserve).
  - 3C: French minimal pairs ("Vous venez." / "Vous venez ?" / ...) + blind A/B kit.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import wave as wave_io
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tiny_voca.config import Config
from tiny_voca.data import AnnotatedDataset, VOWEL_IDS
from tiny_voca.evaluate import dtw_distance
from tiny_voca.infer import load_model
from tiny_voca.losses import SpectralLoss
from tiny_voca.model import TinyVoca
from tiny_voca.text import PHONE_IDS, phonemize
from tiny_voca.train import balanced_indices

from eval_p4_ablation import _select_disjoint_pairs, stratified_text_bootstrap_ci
from p0_expressivity import (
    SEMITONES,
    _apply_voiced_vowels,
    _safe_corr,
    _terminal_inflection_st,
    compute_intonation_metrics,
    extract_unbiased_f0,
    oracle_synthesize,
    warped_token_alignment,
)

# Interrogative partielle (wh-question) vs totale (yes/no) regex for French.
_WH_RE = re.compile(
    r"(?:^|[\s,'\"«(\-—])(qui|que|qu'|quoi|où|quand|comment|pourquoi|combien|quel|quels|quelle|quelles|lequel|laquelle|lesquels|lesquelles)(?:$|[\s,'\"»?!.;:\-])",
    re.IGNORECASE,
)
_EST_CE_QUE_RE = re.compile(r"\best-ce\s+qu(?:e\b|')", re.IGNORECASE)

MINIMAL_PAIRS = (
    ("vous_venez_decl", "Vous venez."),
    ("vous_venez_qtot", "Vous venez ?"),
    ("vous_venez_excl", "Vous venez !"),
    ("deja_parti_decl", "Il est déjà parti."),
    ("deja_parti_qtot", "Il est déjà parti ?"),
    ("trouve_cle_decl", "Tu as trouvé la clé."),
    ("trouve_cle_qtot", "Tu as trouvé la clé ?"),
    ("demain_matin_decl", "Elle arrive demain matin."),
    ("demain_matin_qtot", "Elle arrive demain matin ?"),
    ("vraiment_fini_decl", "C'est vraiment fini."),
    ("vraiment_fini_qtot", "C'est vraiment fini ?"),
    ("pourquoi_pars_qpart", "Pourquoi tu pars maintenant ?"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def load_any_model(path: Path, device: str = "cpu") -> tuple[TinyVoca, int]:
    """Load either an inference.pt or a training checkpoint (step-*.pt / last.pt)."""
    st = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in st:
        m = load_model(path, device=device, check_frontend=True)
        return m, int(st.get("step", -1))
    if "system" in st:
        cfg = Config.from_dict(st["config"])
        m = TinyVoca(cfg)
        prefix = "model."
        model_sd = {k[len(prefix):]: v for k, v in st["system"].items() if k.startswith(prefix)}
        m.load_state_dict(model_sd, strict=True)
        return m.to(device).eval(), int(st.get("step", -1))
    raise ValueError(f"Unrecognized checkpoint format: {path}")


@torch.inference_mode()
def predict_donor_log_f0(donor: TinyVoca, tokens: torch.Tensor, stress: torch.Tensor,
                         token_lengths: torch.Tensor, speakers: torch.Tensor,
                         durations: torch.Tensor) -> torch.Tensor:
    """Predict continuous log_f0 [1, T] from a donor model on an exact shared duration grid."""
    hidden, mask = donor.encode(tokens, stress, token_lengths, speakers)
    frames, frame_mask, _ = donor.expand(hidden, durations, mask, speakers)
    _, prosody = donor.predict_prosody(frames, frame_mask)
    return prosody[..., 0] * frame_mask.to(prosody.dtype)


@torch.inference_mode()
def synthesize_decoupled_f0(
    host: TinyVoca,
    item: dict,
    durations: torch.Tensor,
    dec_log_f0: torch.Tensor | None,
    osc_log_f0: torch.Tensor | None,
    seed: int = 0,
    voiced_vowels: bool = True,
) -> tuple[np.ndarray, dict]:
    """Synthesize with host's fixed weights, fixed durations, fixed energy/voicing/noise,
    independently swapping the log_f0 passed to decode() vs vocoder()."""
    device = next(host.parameters()).device
    tokens = item["tokens"][None].to(device)
    stress = item["stress"][None].to(device)
    token_lengths = torch.tensor([tokens.shape[1]], dtype=torch.long, device=device)
    speakers = torch.tensor([int(item["speaker"])], dtype=torch.long, device=device)
    dur = durations.to(device)

    hidden, mask = host.encode(tokens, stress, token_lengths, speakers)
    frames, frame_mask, frame_lengths = host.expand(hidden, dur, mask, speakers)
    prosody_hidden, host_prosody = host.predict_prosody(frames, frame_mask)
    if voiced_vowels:
        host_prosody = _apply_voiced_vowels(host, tokens, dur, host_prosody, frame_mask)

    f0_dec = host_prosody[..., 0] if dec_log_f0 is None else dec_log_f0.to(device, dtype=host_prosody.dtype)
    f0_osc = host_prosody[..., 0] if osc_log_f0 is None else osc_log_f0.to(device, dtype=host_prosody.dtype)

    dec_pros = torch.cat((f0_dec[..., None], host_prosody[..., 1:]), dim=-1)
    condition = host.hard_prosody(dec_pros)
    acoustic = host.decode(prosody_hidden, condition.to(prosody_hidden.dtype), frame_mask)
    samples = int(frame_lengths[0]) * host.config.hop
    noise = torch.randn(1, samples, generator=torch.Generator().manual_seed(seed)).to(device)
    f0_hz = host.frequency(f0_osc)
    wave_t = host.vocoder(acoustic, f0_hz, frame_lengths, noise=noise)
    wave = wave_t[0, :samples].cpu().numpy()
    return wave, {
        "durations": dur[0].cpu().numpy(),
        "dec_log_f0": f0_dec[0].cpu().numpy(),
        "osc_log_f0": f0_osc[0].cpu().numpy(),
        "osc_f0_hz": f0_hz[0].cpu().numpy(),
        "host_voicing_prob": torch.sigmoid(host_prosody[0, :, 1]).cpu().numpy(),
    }


def classify_french_question(text: str) -> str:
    """Distinguish interrogative totale (yes/no) vs partielle (wh-question)."""
    cleaned = _EST_CE_QUE_RE.sub(" ", text)
    if _WH_RE.search(cleaned):
        return "partielle"
    return "totale"


def _syllable_spans(tokens: np.ndarray, durations: np.ndarray) -> list[tuple[int, int]]:
    """Each vowel token defines a French syllable nucleus span [start_frame, end_frame)."""
    spans = []
    pos = 0
    for tok, d in zip(tokens, durations):
        d_i = int(d)
        if int(tok) in VOWEL_IDS and d_i > 0:
            spans.append((pos, pos + d_i))
        pos += d_i
    return spans


def _peak_syllable_stats(cont_st: np.ndarray, reliable: np.ndarray,
                         tokens: np.ndarray, durations: np.ndarray) -> dict:
    spans = _syllable_spans(tokens, durations)
    if not spans:
        return {"n_syllables": 0, "peak_syl_idx": None, "peak_syl_from_end": None, "peak_syl_rel": None, "peak_st": None}
    syl_vals = []
    for idx, (s, e) in enumerate(spans):
        seg = cont_st[s:e]
        rel = reliable[s:e]
        val = float(np.median(seg[rel])) if rel.any() else float(np.median(seg))
        syl_vals.append((idx, val))
    best_idx, best_val = max(syl_vals, key=lambda kv: kv[1])
    n_syl = len(spans)
    return {
        "n_syllables": n_syl,
        "peak_syl_idx": int(best_idx),
        "peak_syl_from_end": int(n_syl - 1 - best_idx),
        "peak_syl_rel": round(best_idx / max(n_syl - 1, 1), 4),
        "peak_st": round(best_val, 3),
    }


def _direction_label(delta_st: float | None, thresh: float = 0.5) -> str | None:
    if delta_st is None or not math.isfinite(delta_st):
        return None
    if delta_st > thresh:
        return "rising"
    if delta_st < -thresh:
        return "falling"
    return "flat"


def _heard_f0_metrics(
    ref_wave: np.ndarray,
    ref_mel: np.ndarray,
    ref_f0: dict,
    ref_target_log_f0: np.ndarray,
    ref_durs: np.ndarray,
    gen_wave: np.ndarray,
    gen_durs: np.ndarray,
    osc_log_f0: np.ndarray,
    tokens: np.ndarray,
    cfg: Config,
    spectral: SpectralLoss,
    device: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Compute predictor error, oscillator->WAV realization error, end-to-end heard error,
    coverage, voicing FP/FN, and return (metrics, gen_reliable_warped, gen_cont_warped_st)."""
    t_ref, t_gen = int(ref_durs.sum()), int(gen_durs.sum())
    gen_f0 = extract_unbiased_f0(gen_wave, cfg, t_gen)
    inton = compute_intonation_metrics(ref_wave, gen_wave, ref_durs, gen_durs, tokens, cfg, ref_f0_info=ref_f0, gen_f0_info=gen_f0)

    gen_mel = spectral.log_mel(torch.from_numpy(gen_wave)[None].to(device))[0].cpu().numpy()
    dtw = float(dtw_distance(gen_mel, ref_mel))
    mel_l1 = None
    stft_loss = None
    if t_ref == t_gen:
        min_l = min(len(gen_wave), len(ref_wave))
        with torch.no_grad():
            stft_t, mel_t = spectral(
                torch.from_numpy(gen_wave[:min_l])[None].to(device),
                torch.from_numpy(ref_wave[:min_l])[None].to(device),
                torch.tensor([min_l], device=device),
            )
            stft_loss = round(float(stft_t.item()), 5)
            mel_l1 = round(float(mel_t.item()), 5)

    phone_tok = np.isin(tokens, list(PHONE_IDS))
    ref_phone = np.repeat(phone_tok, ref_durs)[:t_ref]
    gen_phone = np.repeat(phone_tok, gen_durs)[:t_gen]
    r_rel = ref_f0["trusted"] & ref_phone
    g_rel = gen_f0["trusted"] & gen_phone

    # 1. Oscillator-to-WAV realization gap (on gen's own reliable voiced phone frames!)
    osc_st = osc_log_f0[:t_gen] * SEMITONES
    wav_st = gen_f0["continuous_log_f0"][:t_gen] * SEMITONES
    if g_rel.any():
        osc_gap = wav_st[g_rel] - osc_st[g_rel]
        wav_vs_osc_mae = round(float(np.mean(np.abs(osc_gap))), 5)
        wav_vs_osc_bias = round(float(np.mean(osc_gap)), 5)
        wav_vs_osc_p90 = round(float(np.percentile(np.abs(osc_gap), 90)), 5)
    else:
        wav_vs_osc_mae = wav_vs_osc_bias = wav_vs_osc_p90 = None

    # 2. Predictor (commanded osc) vs Reference YIN & vs dataset target log_f0 (warped onto ref grid)
    osc_warped, mapped = warped_token_alignment(ref_durs, gen_durs, osc_log_f0[:t_gen])
    g_trust_warped, _ = warped_token_alignment(ref_durs, gen_durs, gen_f0["trusted"].astype(np.float64))
    g_cont_warped, _ = warped_token_alignment(ref_durs, gen_durs, gen_f0["continuous_log_f0"])
    g_rel_warped = (g_trust_warped > 0.7) & mapped & ref_phone
    both_rel = r_rel & g_rel_warped

    r_cont_st = ref_f0["continuous_log_f0"] * SEMITONES
    r_tgt_st = ref_target_log_f0[:t_ref] * SEMITONES
    osc_w_st = osc_warped * SEMITONES
    g_cont_w_st = g_cont_warped * SEMITONES

    pred_vs_ref_yin_mae = round(float(np.mean(np.abs(osc_w_st[r_rel] - r_cont_st[r_rel]))), 5) if r_rel.any() else None
    pred_vs_target_mae = round(float(np.mean(np.abs(osc_w_st[r_rel] - r_tgt_st[r_rel]))), 5) if r_rel.any() else None

    # 3. Reliable coverage & voicing FP/FN (warped onto ref speech frames)
    n_ref_phone = max(int(ref_phone.sum()), 1)
    ref_cov = round(float(r_rel.sum() / n_ref_phone), 5)
    gen_cov = round(float(g_rel_warped.sum() / n_ref_phone), 5)
    both_cov = round(float(both_rel.sum() / n_ref_phone), 5)

    ref_unv = ref_phone & ~r_rel
    voicing_fp = round(float((g_rel_warped & ref_unv).sum() / max(int(ref_unv.sum()), 1)), 5)
    voicing_fn = round(float((~g_rel_warped & r_rel).sum() / max(int(r_rel.sum()), 1)), 5)

    rec = {
        "mel_dtw": round(dtw, 5),
        "mel_l1": mel_l1,
        "stft_loss": stft_loss,
        "pred_vs_ref_yin_mae_st": pred_vs_ref_yin_mae,
        "pred_vs_target_mae_st": pred_vs_target_mae,
        "wav_vs_osc_mae_st": wav_vs_osc_mae,
        "wav_vs_osc_bias_st": wav_vs_osc_bias,
        "wav_vs_osc_p90_st": wav_vs_osc_p90,
        "ref_reliable_cov": ref_cov,
        "gen_reliable_cov": gen_cov,
        "both_reliable_cov": both_cov,
        "voicing_fp_rate": voicing_fp,
        "voicing_fn_rate": voicing_fn,
        **{k: v for k, v in inton.items() if isinstance(v, (int, float)) or v is None},
    }
    return rec, g_rel_warped, g_cont_w_st


def _mean_opt(vals: list[float | None]) -> float | None:
    xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    return round(float(np.mean(xs)), 5) if xs else None


def _summarize(rows: list[dict], filter_fn=None) -> dict:
    sub = [r for r in rows if (filter_fn(r) if filter_fn else True)]
    if not sub:
        return {"n": 0}
    keys = [
        "mel_dtw", "mel_l1", "stft_loss",
        "pred_vs_ref_yin_mae_st", "pred_vs_target_mae_st",
        "wav_vs_osc_mae_st", "wav_vs_osc_bias_st", "wav_vs_osc_p90_st",
        "f0_mae_st", "common_mask_f0_mae_st", "common_mask_f0_corr", "common_mask_slope_mae_st_s",
        "f0_corr", "f0_slope_mae_st_s", "end_delta_abs_err_st",
        "gen_f0_std_st", "gen_f0_local_std_st", "gen_accent_prom_st",
        "energy_corr", "gen_vowel_voiced", "voicing_agreement",
        "ref_reliable_cov", "gen_reliable_cov", "both_reliable_cov",
        "voicing_fp_rate", "voicing_fn_rate", "gen_f0_break_rate",
    ]
    out = {"n": len(sub)}
    for k in keys:
        if any(k in r for r in sub):
            out[k] = _mean_opt([r.get(k) for r in sub])
    return out


def _apply_common_mask(
    cond_rows: dict[str, list[dict]],
    cond_masks: dict[str, dict[str, np.ndarray]],
    cond_curves: dict[str, dict[str, np.ndarray]],
    ref_cache: dict[str, dict],
    cond_names: list[str],
    fps: float,
):
    """Add common_mask_f0_mae_st / corr / slope computed on the strict intersection of reliable frames."""
    uids = [r["id"] for r in cond_rows[cond_names[0]]]
    common_covs = []
    for uid in uids:
        rc = ref_cache[uid]
        inter = rc["r_rel"].copy()
        for cn in cond_names:
            inter = inter & cond_masks[cn][uid]
        common_covs.append(float(inter.sum() / max(int(rc["ref_phone"].sum()), 1)))
        r_st = rc["r_cont_st"]
        adj = inter[:-1] & inter[1:] & rc["ref_phone"][:-1] & rc["ref_phone"][1:]
        r_sl = np.diff(r_st)[adj] * fps if adj.any() else None
        for cn in cond_names:
            g_st = cond_curves[cn][uid]
            row = next(r for r in cond_rows[cn] if r["id"] == uid)
            if inter.any():
                row["common_mask_f0_mae_st"] = round(float(np.mean(np.abs(g_st[inter] - r_st[inter]))), 5)
                row["common_mask_f0_corr"] = _safe_corr(r_st[inter], g_st[inter])
            else:
                row["common_mask_f0_mae_st"] = None
                row["common_mask_f0_corr"] = None
            if adj.any() and r_sl is not None:
                g_sl = np.diff(g_st)[adj] * fps
                row["common_mask_slope_mae_st_s"] = round(float(np.mean(np.abs(g_sl - r_sl))), 5)
            else:
                row["common_mask_slope_mae_st_s"] = None
            row["common_mask_cov"] = round(common_covs[-1], 5)
    return round(float(np.mean(common_covs)), 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=Path(".local-runs/colab/fr4k/annotations_e01"))
    parser.add_argument("--host-ckpt", type=Path, default=Path(".local-runs/colab/fr_ref_35k/fr_ref_35k_final/inference.pt"))
    parser.add_argument("--donor-8k", type=Path, default=Path(".local-runs/colab/e06_20k/e06_20k_final/step-00008000.pt"))
    parser.add_argument("--donor-e06-20k", type=Path, default=Path(".local-runs/colab/e06_20k/e06_20k_final/inference.pt"))
    parser.add_argument("--donor-fr10k", type=Path, default=Path(".local-runs/colab/fr_ref_20k_snap10k/snap10k/step-00010000.pt"))
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--out-dir", type=Path, default=Path(".local-runs/p5_audit/f0_causal"))
    args = parser.parse_args()

    device = "cpu"
    host, host_step = load_any_model(args.host_ckpt, device=device)
    donor_8k, step_8k = load_any_model(args.donor_8k, device=device)
    donor_e06_20k, step_e06_20k = load_any_model(args.donor_e06_20k, device=device)
    donor_fr10k, step_fr10k = load_any_model(args.donor_fr10k, device=device)
    cfg = host.config
    fps = cfg.sample_rate / cfg.hop
    spectral = SpectralLoss(cfg.sample_rate).to(device)

    val_manifest = args.annotations / "validation.jsonl"
    val_ds = AnnotatedDataset(val_manifest, cfg, cache=False)
    monitor_ids = {val_ds.rows[i]["id"] for i in balanced_indices(val_ds, 32)}
    dev_selected = _select_disjoint_pairs(val_manifest, monitor_ids, per_category_pairs=5, seed=args.seed)
    id_to_idx = {r["id"]: i for i, r in enumerate(val_ds.rows)}
    items = {r["id"]: val_ds[id_to_idx[r["id"]]] for r in dev_selected}

    # Pre-cache references on P4-dev
    ref_cache = {}
    for row in dev_selected:
        uid = row["id"]
        it = items[uid]
        ref_wave = it["wave"].numpy()
        ref_durs = it["durations"].numpy()
        toks = it["tokens"].numpy()
        t_ref = int(ref_durs.sum())
        ref_mel = spectral.log_mel(it["wave"][None])[0].cpu().numpy()
        ref_f0 = extract_unbiased_f0(ref_wave, cfg, t_ref)
        phone_tok = np.isin(toks, list(PHONE_IDS))
        ref_phone = np.repeat(phone_tok, ref_durs)[:t_ref]
        ref_cache[uid] = {
            "wave": ref_wave,
            "mel": ref_mel,
            "f0": ref_f0,
            "durs": ref_durs,
            "tokens": toks,
            "target_log_f0": it["prosody"][:, 0].numpy(),
            "ref_phone": ref_phone,
            "r_rel": ref_f0["trusted"] & ref_phone,
            "r_cont_st": ref_f0["continuous_log_f0"] * SEMITONES,
        }

    # --- PART 2A + 2B (Oracle durations) ---
    oracle_cond_names = [
        "M00_dec35k_osc35k",
        "M10_decRef_osc35k",
        "M01_dec35k_oscRef",
        "M11_decRef_oscRef",
        "D_e06_8k_into_35k",
        "D_e06_20k_into_35k",
        "D_fr10k_into_35k",
    ]
    oracle_rows: dict[str, list[dict]] = {k: [] for k in oracle_cond_names}
    oracle_masks: dict[str, dict[str, np.ndarray]] = {k: {} for k in oracle_cond_names}
    oracle_curves: dict[str, dict[str, np.ndarray]] = {k: {} for k in oracle_cond_names}

    # --- PART 2B (Identical 35k predicted durations) ---
    pred_cond_names = [
        "P_35k_f0_on_35k_dur",
        "P_e06_8k_f0_on_35k_dur",
        "P_e06_20k_f0_on_35k_dur",
        "P_fr10k_f0_on_35k_dur",
    ]
    pred_rows: dict[str, list[dict]] = {k: [] for k in pred_cond_names}
    pred_masks: dict[str, dict[str, np.ndarray]] = {k: {} for k in pred_cond_names}
    pred_curves: dict[str, dict[str, np.ndarray]] = {k: {} for k in pred_cond_names}

    # Bit-exactness verification against P0 oracle_synthesize (B and C) on first 4 utterances
    bit_exact_diffs = {"B_max_abs_diff": 0.0, "C_max_abs_diff": 0.0}
    blind_wavs_oracle: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    blind_wavs_pred: dict[str, dict[str, np.ndarray]] = defaultdict(dict)

    # Pick 12 representative dev clips for francophone blind listening (3 cats x 2 pairs x F1/M1)
    blind_ids = []
    for cat in ("question", "exclamation", "comma_pause"):
        cat_rows = [r for r in dev_selected if r["category"] == cat]
        for r in cat_rows[:4]:  # 2 pairs = 4 recordings per category
            blind_ids.append(r["id"])

    for idx, row in enumerate(dev_selected):
        uid = row["id"]
        it = items[uid]
        rc = ref_cache[uid]
        tokens_t = it["tokens"][None]
        stress_t = it["stress"][None]
        tlen_t = torch.tensor([tokens_t.shape[1]], dtype=torch.long)
        spk_t = torch.tensor([int(it["speaker"])], dtype=torch.long)
        oracle_dur = it["durations"][None]
        ref_log_f0 = it["prosody"][None, :, 0]

        # Donor F0s on oracle durations
        f0_e8k_o = predict_donor_log_f0(donor_8k, tokens_t, stress_t, tlen_t, spk_t, oracle_dur)
        f0_e20k_o = predict_donor_log_f0(donor_e06_20k, tokens_t, stress_t, tlen_t, spk_t, oracle_dur)
        f0_fr10k_o = predict_donor_log_f0(donor_fr10k, tokens_t, stress_t, tlen_t, spk_t, oracle_dur)

        oracle_specs = [
            ("M00_dec35k_osc35k", None, None),
            ("M10_decRef_osc35k", ref_log_f0, None),
            ("M01_dec35k_oscRef", None, ref_log_f0),
            ("M11_decRef_oscRef", ref_log_f0, ref_log_f0),
            ("D_e06_8k_into_35k", f0_e8k_o, f0_e8k_o),
            ("D_e06_20k_into_35k", f0_e20k_o, f0_e20k_o),
            ("D_fr10k_into_35k", f0_fr10k_o, f0_fr10k_o),
        ]
        for cname, df0, of0 in oracle_specs:
            wav, meta = synthesize_decoupled_f0(host, it, oracle_dur, df0, of0, seed=idx, voiced_vowels=True)
            if idx < 4 and cname == "M00_dec35k_osc35k":
                w_b, _ = oracle_synthesize(host, row, it, "B_oracle_dur", seed=idx)
                bit_exact_diffs["B_max_abs_diff"] = max(bit_exact_diffs["B_max_abs_diff"], float(np.max(np.abs(wav - w_b))))
            if idx < 4 and cname == "M11_decRef_oscRef":
                w_c, _ = oracle_synthesize(host, row, it, "C_oracle_dur_f0", seed=idx)
                bit_exact_diffs["C_max_abs_diff"] = max(bit_exact_diffs["C_max_abs_diff"], float(np.max(np.abs(wav - w_c))))
            rec, g_rel_w, g_cont_w = _heard_f0_metrics(
                rc["wave"], rc["mel"], rc["f0"], rc["target_log_f0"], rc["durs"],
                wav, meta["durations"], meta["osc_log_f0"], rc["tokens"], cfg, spectral, device,
            )
            rec.update(id=uid, text_hash=row["text_hash"], speaker=row["speaker"], category=row["category"])
            oracle_rows[cname].append(rec)
            oracle_masks[cname][uid] = g_rel_w
            oracle_curves[cname][uid] = g_cont_w
            if uid in blind_ids and cname in ("M00_dec35k_osc35k", "M10_decRef_osc35k", "M01_dec35k_oscRef", "M11_decRef_oscRef"):
                blind_wavs_oracle[uid][cname] = wav

        # Shared predicted durations from host (fr_ref_35k)
        with torch.inference_mode():
            h35, m35 = host.encode(tokens_t, stress_t, tlen_t, spk_t)
            pred_dur = host.predict_durations(h35, m35, tokens_t, speed=1.0)
        f0_e8k_p = predict_donor_log_f0(donor_8k, tokens_t, stress_t, tlen_t, spk_t, pred_dur)
        f0_e20k_p = predict_donor_log_f0(donor_e06_20k, tokens_t, stress_t, tlen_t, spk_t, pred_dur)
        f0_fr10k_p = predict_donor_log_f0(donor_fr10k, tokens_t, stress_t, tlen_t, spk_t, pred_dur)

        for cname, df0, of0 in (
            ("P_35k_f0_on_35k_dur", None, None),
            ("P_e06_8k_f0_on_35k_dur", f0_e8k_p, f0_e8k_p),
            ("P_e06_20k_f0_on_35k_dur", f0_e20k_p, f0_e20k_p),
            ("P_fr10k_f0_on_35k_dur", f0_fr10k_p, f0_fr10k_p),
        ):
            wav, meta = synthesize_decoupled_f0(host, it, pred_dur, df0, of0, seed=idx, voiced_vowels=True)
            rec, g_rel_w, g_cont_w = _heard_f0_metrics(
                rc["wave"], rc["mel"], rc["f0"], rc["target_log_f0"], rc["durs"],
                wav, meta["durations"], meta["osc_log_f0"], rc["tokens"], cfg, spectral, device,
            )
            rec.update(id=uid, text_hash=row["text_hash"], speaker=row["speaker"], category=row["category"])
            pred_rows[cname].append(rec)
            pred_masks[cname][uid] = g_rel_w
            pred_curves[cname][uid] = g_cont_w
            if uid in blind_ids and cname in ("P_35k_f0_on_35k_dur", "P_e06_8k_f0_on_35k_dur", "P_e06_20k_f0_on_35k_dur"):
                blind_wavs_pred[uid][cname] = wav

    common_cov_oracle = _apply_common_mask(oracle_rows, oracle_masks, oracle_curves, ref_cache, oracle_cond_names, fps)
    common_cov_pred = _apply_common_mask(pred_rows, pred_masks, pred_curves, ref_cache, pred_cond_names, fps)

    # Stratified text-level CIs for key causal contrasts
    oracle_contrasts = {}
    for cname, spec in (
        ("M10_decRef_only_minus_M00", {"M10_decRef_osc35k": 1.0, "M00_dec35k_osc35k": -1.0}),
        ("M01_oscRef_only_minus_M00", {"M01_dec35k_oscRef": 1.0, "M00_dec35k_osc35k": -1.0}),
        ("M11_bothRef_minus_M01_oscOnly", {"M11_decRef_oscRef": 1.0, "M01_dec35k_oscRef": -1.0}),
        ("D_e06_8k_minus_M00_35k", {"D_e06_8k_into_35k": 1.0, "M00_dec35k_osc35k": -1.0}),
        ("D_e06_8k_minus_D_e06_20k", {"D_e06_8k_into_35k": 1.0, "D_e06_20k_into_35k": -1.0}),
        ("D_fr10k_minus_M00_35k", {"D_fr10k_into_35k": 1.0, "M00_dec35k_osc35k": -1.0}),
    ):
        oracle_contrasts[cname] = {
            grp: {
                mk: stratified_text_bootstrap_ci(oracle_rows, spec, mk, speaker_filter=(None if grp == "all" else grp), n_boot=3000, seed=args.seed)
                for mk in ("mel_dtw", "mel_l1", "stft_loss", "f0_mae_st", "common_mask_f0_mae_st", "common_mask_f0_corr", "f0_slope_mae_st_s", "wav_vs_osc_mae_st")
            }
            for grp in ("all", "F1", "M1")
        }

    pred_contrasts = {}
    for cname, spec in (
        ("P_e06_8k_minus_35k", {"P_e06_8k_f0_on_35k_dur": 1.0, "P_35k_f0_on_35k_dur": -1.0}),
        ("P_e06_8k_minus_e06_20k", {"P_e06_8k_f0_on_35k_dur": 1.0, "P_e06_20k_f0_on_35k_dur": -1.0}),
        ("P_fr10k_minus_35k", {"P_fr10k_f0_on_35k_dur": 1.0, "P_35k_f0_on_35k_dur": -1.0}),
    ):
        pred_contrasts[cname] = {
            grp: {
                mk: stratified_text_bootstrap_ci(pred_rows, spec, mk, speaker_filter=(None if grp == "all" else grp), n_boot=3000, seed=args.seed)
                for mk in ("mel_dtw", "f0_mae_st", "common_mask_f0_mae_st", "common_mask_f0_corr", "f0_slope_mae_st_s", "end_delta_abs_err_st")
            }
            for grp in ("all", "F1", "M1")
        }

    # --- PART 3B: Questions deep-dive on ALL paired questions in validation.jsonl (disjoint from test!) ---
    by_val_th: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in val_ds.rows:
        if "?" in r["text"]:
            by_val_th[r["text_hash"]][r["speaker"]] = r
    q_records = []
    for th, spks in sorted(by_val_th.items()):
        if "F1" not in spks or "M1" not in spks:
            continue
        qtype = classify_french_question(spks["F1"]["text"])
        for spk in ("F1", "M1"):
            row = spks[spk]
            uid = row["id"]
            it = val_ds[id_to_idx[uid]]
            ref_wave = it["wave"].numpy()
            ref_durs = it["durations"].numpy()
            toks = it["tokens"].numpy()
            t_ref = int(ref_durs.sum())
            ref_f0 = extract_unbiased_f0(ref_wave, cfg, t_ref)
            gen_wave, meta = oracle_synthesize(host, row, it, "A_free", seed=0)
            gen_durs = meta["durations"]
            t_gen = int(gen_durs.sum())
            gen_f0 = extract_unbiased_f0(gen_wave, cfg, t_gen)

            phone_tok = np.isin(toks, list(PHONE_IDS))
            r_speech = np.repeat(phone_tok, ref_durs)[:t_ref]
            g_speech = np.repeat(phone_tok, gen_durs)[:t_gen]
            r_rel = ref_f0["trusted"] & r_speech
            g_rel = gen_f0["trusted"] & g_speech
            r_st = ref_f0["continuous_log_f0"] * SEMITONES
            g_st = gen_f0["continuous_log_f0"] * SEMITONES

            r_end = _terminal_inflection_st(r_st, r_rel)
            g_end = _terminal_inflection_st(g_st, g_rel)
            r_dir = _direction_label(r_end)
            g_dir = _direction_label(g_end)
            r_pk = _peak_syllable_stats(r_st, r_rel, toks, ref_durs)
            g_pk = _peak_syllable_stats(g_st, g_rel, toks, gen_durs)
            inton = compute_intonation_metrics(ref_wave, gen_wave, ref_durs, gen_durs, toks, cfg, ref_f0_info=ref_f0, gen_f0_info=gen_f0)

            q_records.append({
                "id": uid,
                "text_hash": th,
                "speaker": spk,
                "qtype": qtype,
                "text": row["text"],
                "f0_mae_st": inton["f0_mae_st"],
                "f0_corr": inton["f0_corr"],
                "ref_end_st": round(r_end, 3) if r_end is not None else None,
                "gen_end_st": round(g_end, 3) if g_end is not None else None,
                "end_abs_err_st": round(abs(g_end - r_end), 3) if (r_end is not None and g_end is not None) else None,
                "ref_dir": r_dir,
                "gen_dir": g_dir,
                "dir_match": bool(r_dir == g_dir) if (r_dir and g_dir) else None,
                "n_syllables": r_pk["n_syllables"],
                "ref_peak_syl_idx": r_pk["peak_syl_idx"],
                "gen_peak_syl_idx": g_pk["peak_syl_idx"],
                "ref_peak_syl_from_end": r_pk["peak_syl_from_end"],
                "gen_peak_syl_from_end": g_pk["peak_syl_from_end"],
                "peak_syl_abs_err": abs(g_pk["peak_syl_idx"] - r_pk["peak_syl_idx"]) if (r_pk["peak_syl_idx"] is not None and g_pk["peak_syl_idx"] is not None) else None,
                "peak_syl_exact_match": bool(g_pk["peak_syl_idx"] == r_pk["peak_syl_idx"]) if (r_pk["peak_syl_idx"] is not None and g_pk["peak_syl_idx"] is not None) else None,
            })

    q_summary = {}
    for qt in ("all", "totale", "partielle"):
        q_summary[qt] = {}
        for spk in ("all", "F1", "M1"):
            sub = [r for r in q_records if (qt == "all" or r["qtype"] == qt) and (spk == "all" or r["speaker"] == spk)]
            if not sub:
                continue
            ref_dirs = Counter(r["ref_dir"] for r in sub if r["ref_dir"])
            gen_dirs = Counter(r["gen_dir"] for r in sub if r["gen_dir"])
            dir_matches = [r["dir_match"] for r in sub if r["dir_match"] is not None]
            pk_exact = [r["peak_syl_exact_match"] for r in sub if r["peak_syl_exact_match"] is not None]
            q_summary[qt][spk] = {
                "n": len(sub),
                "f0_mae_st": _mean_opt([r["f0_mae_st"] for r in sub]),
                "f0_corr": _mean_opt([r["f0_corr"] for r in sub]),
                "ref_end_mean_st": _mean_opt([r["ref_end_st"] for r in sub]),
                "gen_end_mean_st": _mean_opt([r["gen_end_st"] for r in sub]),
                "end_abs_err_st": _mean_opt([r["end_abs_err_st"] for r in sub]),
                "dir_match_acc": round(float(np.mean(dir_matches)), 4) if dir_matches else None,
                "ref_dir_counts": dict(ref_dirs),
                "gen_dir_counts": dict(gen_dirs),
                "ref_peak_from_end_mean": _mean_opt([r["ref_peak_syl_from_end"] for r in sub]),
                "gen_peak_from_end_mean": _mean_opt([r["gen_peak_syl_from_end"] for r in sub]),
                "peak_syl_abs_err_mean": _mean_opt([r["peak_syl_abs_err"] for r in sub]),
                "peak_syl_exact_acc": round(float(np.mean(pk_exact)), 4) if pk_exact else None,
            }

    # --- PART 3C: French minimal pairs ---
    mp_dir = args.out_dir / "minimal_pairs"
    mp_dir.mkdir(parents=True, exist_ok=True)
    mp_rows = []
    for label, text in MINIMAL_PAIRS:
        ph = phonemize(text)
        toks_t = torch.tensor([ph.ids], dtype=torch.long)
        str_t = torch.tensor([ph.stress], dtype=torch.long)
        tlen_t = torch.tensor([len(ph.ids)], dtype=torch.long)
        toks_np = np.asarray(ph.ids, dtype=np.int64)
        phone_tok = np.isin(toks_np, list(PHONE_IDS))
        for spk_i, spk_name in enumerate(cfg.speakers):
            spk_t = torch.tensor([spk_i], dtype=torch.long)
            with torch.inference_mode():
                h35, m35 = host.encode(toks_t, str_t, tlen_t, spk_t)
                dur35 = host.predict_durations(h35, m35, toks_t, speed=1.0)
                w35_t, l35_t, meta35 = host.synthesize(
                    toks_t, str_t, tlen_t, spk_t, speed=1.0, seed=42, pitch_range=1.0, pitch_jitter=0.0, voiced_vowels=True
                )
            w35 = w35_t[0, :int(l35_t[0])].cpu().numpy()
            d35 = meta35["durations"][0].cpu().numpy()
            _write_wav(mp_dir / f"{label}_{spk_name}_35k.wav", w35, cfg.sample_rate)

            # Also inject e06_8k F0 on exact same 35k predicted durations
            f0_8k = predict_donor_log_f0(donor_8k, toks_t, str_t, tlen_t, spk_t, dur35)
            fake_item = {"tokens": toks_t[0], "stress": str_t[0], "speaker": spk_i}
            w8k, _ = synthesize_decoupled_f0(host, fake_item, dur35, f0_8k, f0_8k, seed=42, voiced_vowels=True)
            _write_wav(mp_dir / f"{label}_{spk_name}_f0_8k_in_35k.wav", w8k, cfg.sample_rate)

            for sys_tag, wav_arr in (("fr_ref_35k", w35), ("f0_8k_in_35k", w8k)):
                t_f = int(d35.sum())
                f0_info = extract_unbiased_f0(wav_arr, cfg, t_f)
                g_speech = np.repeat(phone_tok, d35)[:t_f]
                g_rel = f0_info["trusted"] & g_speech
                g_st = f0_info["continuous_log_f0"] * SEMITONES
                end_st = _terminal_inflection_st(g_st, g_rel)
                pk = _peak_syllable_stats(g_st, g_rel, toks_np, d35)
                spans = _syllable_spans(toks_np, d35)
                last_vow_ms = round((spans[-1][1] - spans[-1][0]) * (cfg.hop / cfg.sample_rate) * 1000.0, 1) if spans else None
                mp_rows.append({
                    "label": label,
                    "text": text,
                    "speaker": spk_name,
                    "system": sys_tag,
                    "duration_s": round(len(wav_arr) / cfg.sample_rate, 3),
                    "last_vowel_ms": last_vow_ms,
                    "end_inflection_st": round(end_st, 3) if end_st is not None else None,
                    "end_dir": _direction_label(end_st),
                    "f0_std_st": round(float(np.std(g_st[g_rel])), 3) if g_rel.any() else None,
                    **pk,
                })

    # --- PART 3D: Francophone Blind A/B Listening Kit (randomized order, tie allowed) ---
    blind_dir = args.out_dir / "blind_listening_p5"
    blind_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed + 505)
    blind_pairs = []
    sheet_lines = [
        "# Protocole d'écoute francophone aveugle — P5 (A vs B, égalité autorisée)",
        "",
        "Consigne : pour chaque paire, écoutez `REF.wav` (référence humaine, si souhaité comme ancre) puis `A.wav` et `B.wav` dans l'ordre présenté.",
        "Notez votre préférence sur l'**expressivité / naturel prosodique** : `A`, `B`, ou `Égalité` (+ commentaire libre sur intonation / rythme / timbre).",
        "",
        "| Paire | Voix | Catégorie | Texte | Choix (`A` / `B` / `Égalité`) | Commentaire |",
        "|---|---|---|---|---|---|",
    ]
    pair_idx = 1
    for uid in blind_ids:
        row = next(r for r in dev_selected if r["id"] == uid)
        _write_wav(blind_dir / f"P{pair_idx:02d}_{uid}_REF.wav", ref_cache[uid]["wave"], cfg.sample_rate)
        # Pair type 1: predicted-duration 35k vs F0_e06_8k -> 35k
        ab = [("fr_ref_35k", blind_wavs_pred[uid]["P_35k_f0_on_35k_dur"]),
              ("f0_e06_8k_in_35k", blind_wavs_pred[uid]["P_e06_8k_f0_on_35k_dur"])]
        rng.shuffle(ab)
        pid = f"P{pair_idx:02d}_predDur_{uid}"
        _write_wav(blind_dir / f"{pid}_A.wav", ab[0][1], cfg.sample_rate)
        _write_wav(blind_dir / f"{pid}_B.wav", ab[1][1], cfg.sample_rate)
        blind_pairs.append({
            "pair_id": pid, "experiment": "pred_dur_35k_vs_8kF0",
            "id": uid, "speaker": row["speaker"], "category": row["category"], "text": row["text"],
            "A": ab[0][0], "B": ab[1][0],
        })
        sheet_lines.append(f"| `{pid}` | {row['speaker']} | {row['category']} | {row['text']} | | |")
        pair_idx += 1

    # Also add 4 cross-F0 pairs (M01_oscRef vs M10_decRef) to hear oscillator vs decoder path
    for uid in blind_ids[:4]:
        row = next(r for r in dev_selected if r["id"] == uid)
        ab = [("M01_dec35k_oscRef", blind_wavs_oracle[uid]["M01_dec35k_oscRef"]),
              ("M10_decRef_osc35k", blind_wavs_oracle[uid]["M10_decRef_osc35k"])]
        rng.shuffle(ab)
        pid = f"P{pair_idx:02d}_crossF0_{uid}"
        _write_wav(blind_dir / f"{pid}_A.wav", ab[0][1], cfg.sample_rate)
        _write_wav(blind_dir / f"{pid}_B.wav", ab[1][1], cfg.sample_rate)
        blind_pairs.append({
            "pair_id": pid, "experiment": "oracle_dur_oscRef_vs_decRef",
            "id": uid, "speaker": row["speaker"], "category": row["category"], "text": row["text"],
            "A": ab[0][0], "B": ab[1][0],
        })
        sheet_lines.append(f"| `{pid}` | {row['speaker']} | {row['category']} | {row['text']} | | |")
        pair_idx += 1

    (blind_dir / "blind_sheet_fr.md").write_text("\n".join(sheet_lines) + "\n", encoding="utf-8")
    (blind_dir / "blind_key.json").write_text(json.dumps({
        "judgments_collected": 0,
        "pairs": blind_pairs,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Save complete JSON report
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoints": {
            "host_35k": {"path": str(args.host_ckpt), "step": host_step, "sha256": sha256_file(args.host_ckpt)},
            "donor_e06_8k": {"path": str(args.donor_8k), "step": step_8k, "sha256": sha256_file(args.donor_8k)},
            "donor_e06_20k": {"path": str(args.donor_e06_20k), "step": step_e06_20k, "sha256": sha256_file(args.donor_e06_20k)},
            "donor_fr10k": {"path": str(args.donor_fr10k), "step": step_fr10k, "sha256": sha256_file(args.donor_fr10k)},
        },
        "dev_utterances": len(dev_selected),
        "dev_text_hashes": sorted({r["text_hash"] for r in dev_selected}),
        "bit_exact_p0_verification": bit_exact_diffs,
        "common_mask_coverage_oracle": common_cov_oracle,
        "common_mask_coverage_pred_dur": common_cov_pred,
        "oracle_duration_summary": {
            cn: {grp: _summarize(oracle_rows[cn], None if grp == "all" else (lambda r, s=grp: r["speaker"] == s)) for grp in ("all", "F1", "M1")}
            for cn in oracle_cond_names
        },
        "oracle_duration_contrasts_95ci": oracle_contrasts,
        "pred_duration_summary": {
            cn: {grp: _summarize(pred_rows[cn], None if grp == "all" else (lambda r, s=grp: r["speaker"] == s)) for grp in ("all", "F1", "M1")}
            for cn in pred_cond_names
        },
        "pred_duration_contrasts_95ci": pred_contrasts,
        "questions_validation_summary": q_summary,
        "questions_validation_records": q_records,
        "minimal_pairs": mp_rows,
        "blind_listening_pairs_count": len(blind_pairs),
        "blind_listening_judgments_collected": 0,
    }
    (args.out_dir / "p5_f0_causal_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Compact readable summary
    print("=== BIT-EXACT CHECK vs P0 B/C ===", bit_exact_diffs)
    print(f"\n=== 2A & 2B: ORACLE DURATIONS (P4-dev n=40, common_cov={common_cov_oracle}) ===")
    for cn in oracle_cond_names:
        s = report["oracle_duration_summary"][cn]["all"]
        f1 = report["oracle_duration_summary"][cn]["F1"]
        m1 = report["oracle_duration_summary"][cn]["M1"]
        print(
            f"  {cn:20s} | dtw={s['mel_dtw']:.4f} melL1={s['mel_l1']:.4f} stft={s['stft_loss']:.4f} "
            f"| predRef={s['pred_vs_ref_yin_mae_st']:.3f} oscWav={s['wav_vs_osc_mae_st']:.3f}(F1:{f1['wav_vs_osc_mae_st']:.3f}/M1:{m1['wav_vs_osc_mae_st']:.3f}) "
            f"| wavRef={s['f0_mae_st']:.3f} commMAE={s['common_mask_f0_mae_st']:.3f} commR={s['common_mask_f0_corr']:.4f} "
            f"cov={s['both_reliable_cov']:.3f} vFP={s['voicing_fp_rate']:.3f} vFN={s['voicing_fn_rate']:.3f}"
        )

    print(f"\n=== 2B: IDENTICAL 35k PREDICTED DURATIONS (P4-dev n=40, common_cov={common_cov_pred}) ===")
    for cn in pred_cond_names:
        s = report["pred_duration_summary"][cn]["all"]
        f1 = report["pred_duration_summary"][cn]["F1"]
        m1 = report["pred_duration_summary"][cn]["M1"]
        print(
            f"  {cn:24s} | dtw={s['mel_dtw']:.4f} | predRef={s['pred_vs_ref_yin_mae_st']:.3f} oscWav={s['wav_vs_osc_mae_st']:.3f} "
            f"| wavRef={s['f0_mae_st']:.3f}(F1:{f1['f0_mae_st']:.3f}/M1:{m1['f0_mae_st']:.3f}) "
            f"commMAE={s['common_mask_f0_mae_st']:.3f} commR={s['common_mask_f0_corr']:.4f} slope={s['f0_slope_mae_st_s']:.2f} end={s['end_delta_abs_err_st']:.3f}"
        )

    print("\n=== KEY STRATIFIED TEXT-LEVEL 95% CIs (all) ===")
    for label, dmap in [("ORACLE", oracle_contrasts), ("PRED_DUR", pred_contrasts)]:
        for cname, gmap in dmap.items():
            g = gmap["all"]
            parts = []
            for mk in ("mel_dtw", "stft_loss", "common_mask_f0_mae_st", "common_mask_f0_corr", "f0_slope_mae_st_s"):
                ci = g.get(mk)
                if ci:
                    st = "*" if ci["significant"] else ""
                    parts.append(f"{mk}={ci['mean_delta']:+.4f}[{ci['ci95_low']:+.4f},{ci['ci95_high']:+.4f}]{st}")
            print(f"  [{label}] {cname:30s}: " + " | ".join(parts))

    print("\n=== 3B: QUESTIONS (validation.jsonl paired, totales vs partielles) ===")
    for qt in ("totale", "partielle"):
        for spk in ("all", "F1", "M1"):
            if spk in q_summary.get(qt, {}):
                s = q_summary[qt][spk]
                print(
                    f"  {qt:9s} {spk:3s} (n={s['n']:2d}): f0_mae={s['f0_mae_st']:.3f} r={s['f0_corr']:.3f} "
                    f"refEnd={s['ref_end_mean_st']:+.2f} genEnd={s['gen_end_mean_st']:+.2f} dirAcc={s['dir_match_acc']:.3f} "
                    f"refDirs={s['ref_dir_counts']} genDirs={s['gen_dir_counts']} "
                    f"pkEndRef={s['ref_peak_from_end_mean']:.2f} pkEndGen={s['gen_peak_from_end_mean']:.2f} pkErr={s['peak_syl_abs_err_mean']:.2f} pkAcc={s['peak_syl_exact_acc']:.3f}"
                )

    print("\n=== 3C: MINIMAL PAIRS (fr_ref_35k) ===")
    for r in mp_rows:
        if r["system"] == "fr_ref_35k":
            print(f"  {r['label']:22s} {r['speaker']}: dur={r['duration_s']:.3f}s lastV={r['last_vowel_ms']}ms end={r['end_inflection_st']:+.2f}st({r['end_dir']}) pkFromEnd={r['peak_syl_from_end']}/{r['n_syllables']}")


if __name__ == "__main__":
    main()
