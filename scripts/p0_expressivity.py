"""P0 expressive prosody diagnostics, unbiased F0 extraction, oracle counterfactuals, and gradient probes.

Designed to run 100% on CPU without modifying model weights, and reusable for E06 comparison (P3).
"""

from collections import defaultdict
import json
import math
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks
import torch
from torch import nn

from tiny_voca.config import Config
from tiny_voca.data import AnnotatedDataset, continuous_pitch
from tiny_voca.losses import Discriminators, generator_gan_loss
from tiny_voca.text import PHONE_IDS, SYMBOL_TO_ID, VOWEL_SET, VOICELESS
from tiny_voca.train import TrainingSystem, WEIGHTS, collate, crops

SEMITONES = 12.0 / math.log(2.0)
VOWEL_IDS = frozenset(SYMBOL_TO_ID[v] for v in VOWEL_SET)
VOICELESS_IDS = frozenset(SYMBOL_TO_ID[v] for v in VOICELESS)
COMMA_TOKENS = frozenset(SYMBOL_TO_ID[s] for s in (",", ";", ":"))
TERMINAL_PUNCT_TOKENS = frozenset(SYMBOL_TO_ID[s] for s in (".", "?", "!", "…"))
_LP_SOS_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def autocorr_harmonicity(wave: np.ndarray, f0_hz: np.ndarray, sr: int = 24000,
                         win: int = 1024, hop: int = 256, frames: int | None = None) -> np.ndarray:
    """Zero-phase low-pass (<1.2 kHz) normalized lag-autocorrelation around ``sr / f0_hz``."""
    wave = np.asarray(wave, dtype=np.float64)
    limit = len(f0_hz) if frames is None else min(len(f0_hz), frames)
    if len(wave) >= 16:
        if sr not in _LP_SOS_CACHE:
            _LP_SOS_CACHE[sr] = butter(4, min(1200.0 / (sr / 2.0), 0.9), btype="low")
        b, a = _LP_SOS_CACHE[sr]
        wave_lp = filtfilt(b, a, wave)
    else:
        wave_lp = wave
    pad = win // 2
    padded = np.pad(wave_lp, (pad, pad))
    starts = np.arange(0, len(wave), hop)[:limit]
    res = np.zeros(len(starts), dtype=np.float32)
    half_win = win // 2
    for i, start in enumerate(starts):
        seg = padded[start : start + win]
        seg = seg - seg.mean()
        if float(np.dot(seg, seg)) < 1e-9:
            continue
        freq = float(f0_hz[i])
        if not math.isfinite(freq) or freq <= 0.0:
            continue
        lag_f = sr / freq
        lo = max(1, int(math.floor(lag_f)) - 2)
        hi = min(half_win, int(math.ceil(lag_f)) + 3)
        best = 0.0
        for lag in range(lo, hi):
            aa, bb = seg[:-lag], seg[lag:]
            den = math.sqrt(float(np.dot(aa, aa)) * float(np.dot(bb, bb))) + 1e-12
            corr = float(np.dot(aa, bb)) / den
            if corr > best:
                best = corr
        res[i] = best
    return res


def extract_unbiased_f0(wave: np.ndarray, config: Config, frames: int | None = None,
                        harm_threshold: float = 0.55, floor_db: float = 38.0) -> dict:
    """Extract waveform F0 via 1024-sample YIN + lag harmonicity + continuity filtering.

    Avoids pyin's HMM onset delay and creak dropouts on low male voices identified in E05.
    """
    wave = np.asarray(wave, dtype=np.float32)
    t_frames = math.ceil(len(wave) / config.hop) if frames is None else int(frames)
    if len(wave) < t_frames * config.hop:
        wave = np.pad(wave, (0, t_frames * config.hop - len(wave)))
    yin_hz = librosa.yin(
        wave, fmin=55.0, fmax=500.0, sr=config.sample_rate,
        frame_length=1024, hop_length=config.hop, trough_threshold=0.1, center=True,
    )[:t_frames].astype(np.float64)
    rms = librosa.feature.rms(
        y=wave, frame_length=1024, hop_length=config.hop, center=True,
    )[0, :t_frames]
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    peak_db = float(np.max(rms_db)) if rms_db.size else -90.0
    harm = autocorr_harmonicity(wave, yin_hz, config.sample_rate, 1024, config.hop, t_frames)

    raw_voiced = (harm >= harm_threshold) & (rms_db >= peak_db - floor_db) & np.isfinite(yin_hz) & (yin_hz >= 55.0) & (yin_hz <= 500.0)
    if raw_voiced.sum() >= 3:
        med_hz = float(np.median(yin_hz[raw_voiced]))
        raw_voiced = raw_voiced & (yin_hz >= med_hz / 1.85) & (yin_hz <= med_hz * 1.85)

    log_f0 = np.where(raw_voiced, np.log(np.maximum(yin_hz, 1.0) / config.pitch_reference_hz), 0.0).astype(np.float32)
    cont_log_f0, trusted_t = continuous_pitch(torch.from_numpy(log_f0), torch.from_numpy(raw_voiced.astype(np.float32)))
    trusted = trusted_t.numpy() > 0.5
    return {
        "f0_hz": yin_hz.astype(np.float32),
        "log_f0": log_f0,
        "continuous_log_f0": cont_log_f0.numpy(),
        "voiced": raw_voiced,
        "trusted": trusted,
        "harm": harm,
        "rms_db": rms_db.astype(np.float32),
    }


def _sentence_category(text: str) -> str:
    if "?" in text:
        return "question"
    if "!" in text:
        return "exclamation"
    if "," in text or ";" in text:
        return "comma_pause"
    return "plain_assert"


def select_representative_utterances(manifest_path: Path, per_category_pairs: int = 5,
                                     speakers: tuple[str, ...] = ("F1", "M1")) -> list[dict]:
    """Select balanced F1/M1 held-out paired utterances across intonations and lengths."""
    rows = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_hash: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["speaker"] in speakers:
            by_hash[row["text_hash"]][row["speaker"]] = row

    paired_by_cat: dict[str, list[dict[str, dict]]] = defaultdict(list)
    for pair in by_hash.values():
        if all(spk in pair for spk in speakers):
            cat = _sentence_category(pair[speakers[0]]["text"])
            paired_by_cat[cat].append(pair)

    selected = []
    for cat in ("plain_assert", "question", "exclamation", "comma_pause"):
        candidates = sorted(paired_by_cat[cat], key=lambda p: (p[speakers[0]]["frames"], p[speakers[0]]["text_hash"]))
        if len(candidates) <= per_category_pairs:
            chosen = candidates
        else:
            idxs = [int(round(q * (len(candidates) - 1))) for q in np.linspace(0.08, 0.92, per_category_pairs)]
            # Deduplicate only if rounding collides on small buckets.
            seen = set()
            chosen = []
            for idx in idxs:
                while idx in seen and idx + 1 < len(candidates):
                    idx += 1
                seen.add(idx)
                chosen.append(candidates[idx])
        for pair in chosen:
            for spk in speakers:
                selected.append({**pair[spk], "category": cat})
    return selected


def warped_token_alignment(ref_durations: np.ndarray, gen_durations: np.ndarray,
                           gen_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise-linearly map ``gen_values`` onto the reference frame grid per active token."""
    t_ref = int(np.sum(ref_durations))
    out = np.zeros(t_ref, dtype=np.float64)
    mapped_mask = np.zeros(t_ref, dtype=bool)
    ref_pos, gen_pos = 0, 0
    for d_r, d_g in zip(ref_durations, gen_durations):
        d_r, d_g = int(d_r), int(d_g)
        if d_r > 0 and d_g > 0:
            seg = np.asarray(gen_values[gen_pos : gen_pos + d_g], dtype=np.float64)
            if d_r == d_g:
                out[ref_pos : ref_pos + d_r] = seg
            elif d_g == 1:
                out[ref_pos : ref_pos + d_r] = seg[0]
            else:
                src_x = np.linspace(0.0, 1.0, d_g)
                dst_x = np.linspace(0.0, 1.0, d_r)
                out[ref_pos : ref_pos + d_r] = np.interp(dst_x, src_x, seg)
            mapped_mask[ref_pos : ref_pos + d_r] = True
        elif d_r > 0 and d_g == 0 and ref_pos > 0:
            out[ref_pos : ref_pos + d_r] = out[ref_pos - 1]
        ref_pos += d_r
        gen_pos += d_g
    return out, mapped_mask


def _local_f0_std(values_st: np.ndarray, mask: np.ndarray, window: int = 14) -> float:
    """Mean rolling-window F0 std (semitones) across sufficiently voiced ~150 ms windows."""
    stds = []
    half = window // 2
    n = len(values_st)
    for idx in range(0, n, max(1, window // 3)):
        seg_mask = mask[max(0, idx - half) : min(n, idx + half + 1)]
        if int(seg_mask.sum()) >= 5:
            stds.append(float(np.std(values_st[max(0, idx - half) : min(n, idx + half + 1)][seg_mask])))
    return float(np.mean(stds)) if stds else 0.0


def _accent_peak_stats(cont_st: np.ndarray, voiced_mask: np.ndarray, win_smooth: int = 37,
                       min_prom_st: float = 1.5, min_dist: int = 12) -> tuple[int, float]:
    """Count F0 accent peaks above a ~400 ms smoothed phrase trendline on voiced regions."""
    if not voiced_mask.any():
        return 0, 0.0
    trend = uniform_filter1d(cont_st.astype(np.float64), size=win_smooth, mode="nearest")
    resid = np.where(voiced_mask, cont_st - trend, -10.0)
    peaks, props = find_peaks(resid, height=min_prom_st, distance=min_dist, prominence=min_prom_st * 0.6)
    valid = [p for p in peaks if voiced_mask[p]]
    if not valid:
        return 0, 0.0
    heights = [float(resid[p]) for p in valid]
    return len(valid), float(np.mean(heights))


def _terminal_inflection_st(cont_st: np.ndarray, voiced_speech_mask: np.ndarray) -> float | None:
    """Signed F0 change (semitones) between last ~250 ms voiced speech and preceding ~300 ms."""
    idx = np.flatnonzero(voiced_speech_mask)
    if len(idx) < 18:
        return None
    tail = cont_st[idx[-min(22, len(idx) // 3) :]]
    body = cont_st[idx[-min(50, len(idx)) : -min(22, len(idx) // 3)]]
    if len(body) < 5 or len(tail) < 5:
        return None
    return float(np.median(tail) - np.median(body))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 4 or float(np.std(a)) < 1e-8 or float(np.std(b)) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def compute_intonation_metrics(ref_wave: np.ndarray, gen_wave: np.ndarray,
                               ref_durations: np.ndarray, gen_durations: np.ndarray,
                               tokens: np.ndarray, config: Config,
                               ref_f0_info: dict | None = None,
                               gen_f0_info: dict | None = None) -> dict:
    """Compute aligned time-series F0, local variance, accents, energy, and voicing metrics."""
    r_dur = np.asarray(ref_durations, dtype=np.int64)
    g_dur = np.asarray(gen_durations, dtype=np.int64)
    tok = np.asarray(tokens, dtype=np.int64)
    t_ref, t_gen = int(r_dur.sum()), int(g_dur.sum())

    r_f0 = ref_f0_info or extract_unbiased_f0(ref_wave, config, t_ref)
    g_f0 = gen_f0_info or extract_unbiased_f0(gen_wave, config, t_gen)

    phone_tok_mask = np.isin(tok, list(PHONE_IDS))
    vowel_tok_mask = np.isin(tok, list(VOWEL_IDS))
    ref_phone_frames = np.repeat(phone_tok_mask, r_dur)[:t_ref]
    ref_vowel_frames = np.repeat(vowel_tok_mask, r_dur)[:t_ref]
    gen_vowel_frames = np.repeat(vowel_tok_mask, g_dur)[:t_gen]

    # Warp generated F0, voicing reliability, and RMS energy onto reference tokens.
    g_cont_warped, mapped = warped_token_alignment(r_dur, g_dur, g_f0["continuous_log_f0"])
    g_trust_warped, _ = warped_token_alignment(r_dur, g_dur, g_f0["trusted"].astype(np.float64))
    g_rms_warped, _ = warped_token_alignment(r_dur, g_dur, g_f0["rms_db"])

    r_reliable = r_f0["trusted"] & ref_phone_frames
    g_reliable_orig = g_f0["trusted"] & (np.repeat(phone_tok_mask, g_dur)[:t_gen])
    g_reliable_warped = (g_trust_warped > 0.7) & mapped & ref_phone_frames
    both_reliable = r_reliable & g_reliable_warped

    r_cont_st = r_f0["continuous_log_f0"] * SEMITONES
    g_cont_st = g_f0["continuous_log_f0"] * SEMITONES
    g_cont_warped_st = g_cont_warped * SEMITONES

    diff_st = g_cont_warped_st[both_reliable] - r_cont_st[both_reliable]
    f0_bias_st = float(np.mean(diff_st)) if diff_st.size else None
    f0_mae_st = float(np.mean(np.abs(diff_st))) if diff_st.size else None
    f0_rmse_st = float(np.sqrt(np.mean(diff_st ** 2))) if diff_st.size else None
    f0_corr = _safe_corr(r_cont_st[both_reliable], g_cont_warped_st[both_reliable])

    # Frame-to-frame F0 slope (semitones / second) on adjacent mutually reliable frames.
    adj = both_reliable[:-1] & both_reliable[1:] & ref_phone_frames[:-1] & ref_phone_frames[1:]
    fps = config.sample_rate / config.hop
    if adj.any():
        r_slope = np.diff(r_cont_st)[adj] * fps
        g_slope = np.diff(g_cont_warped_st)[adj] * fps
        slope_mae = float(np.mean(np.abs(g_slope - r_slope)))
    else:
        slope_mae = None

    # Octave breaks inside contiguous vowel cores (> 4 st step between adjacent voiced vowel frames).
    def octave_break_rate(cont_st_arr: np.ndarray, v_mask: np.ndarray, r_mask: np.ndarray) -> float:
        adj_v = v_mask[:-1] & v_mask[1:] & r_mask[:-1] & r_mask[1:]
        if not adj_v.any():
            return 0.0
        return float(np.mean(np.abs(np.diff(cont_st_arr)[adj_v]) > 4.0))

    r_std = float(np.std(r_cont_st[r_reliable])) if r_reliable.sum() >= 2 else 0.0
    g_std = float(np.std(g_cont_st[g_reliable_orig])) if g_reliable_orig.sum() >= 2 else 0.0
    r_iqr = float(np.percentile(r_cont_st[r_reliable], 75) - np.percentile(r_cont_st[r_reliable], 25)) if r_reliable.sum() >= 2 else 0.0
    g_iqr = float(np.percentile(g_cont_st[g_reliable_orig], 75) - np.percentile(g_cont_st[g_reliable_orig], 25)) if g_reliable_orig.sum() >= 2 else 0.0
    r_p95_05 = float(np.percentile(r_cont_st[r_reliable], 95) - np.percentile(r_cont_st[r_reliable], 5)) if r_reliable.sum() >= 2 else 0.0
    g_p95_05 = float(np.percentile(g_cont_st[g_reliable_orig], 95) - np.percentile(g_cont_st[g_reliable_orig], 5)) if g_reliable_orig.sum() >= 2 else 0.0

    r_loc_std = _local_f0_std(r_cont_st, r_reliable)
    g_loc_std = _local_f0_std(g_cont_st, g_reliable_orig)

    r_pk_cnt, r_pk_prom = _accent_peak_stats(r_cont_st, r_reliable)
    g_pk_cnt, g_pk_prom = _accent_peak_stats(g_cont_st, g_reliable_orig)

    r_end = _terminal_inflection_st(r_cont_st, r_reliable)
    g_end = _terminal_inflection_st(g_cont_st, g_reliable_orig)
    end_err = abs(g_end - r_end) if (r_end is not None and g_end is not None) else None

    # Relative energy variability on active speech phones (centered per utterance).
    gen_phone_frames = np.repeat(phone_tok_mask, g_dur)[:t_gen]
    r_e = r_f0["rms_db"][ref_phone_frames]
    g_e = g_f0["rms_db"][gen_phone_frames]
    r_e_std = float(np.std(r_e)) if r_e.size >= 2 else 0.0
    g_e_std = float(np.std(g_e)) if g_e.size >= 2 else 0.0
    r_e_iqr = float(np.percentile(r_e, 75) - np.percentile(r_e, 25)) if r_e.size >= 2 else 0.0
    g_e_iqr = float(np.percentile(g_e, 75) - np.percentile(g_e, 25)) if g_e.size >= 2 else 0.0
    aligned_e_corr = _safe_corr(
        r_f0["rms_db"][ref_phone_frames & mapped] - float(np.mean(r_f0["rms_db"][ref_phone_frames & mapped])),
        g_rms_warped[ref_phone_frames & mapped] - float(np.mean(g_rms_warped[ref_phone_frames & mapped])),
    ) if (ref_phone_frames & mapped).sum() >= 4 else None

    # Voicing agreement on aligned phone frames.
    g_v_warped, _ = warped_token_alignment(r_dur, g_dur, g_f0["voiced"].astype(np.float64))
    voicing_agree = float(np.mean((r_f0["voiced"][ref_phone_frames & mapped]) == (g_v_warped[ref_phone_frames & mapped] > 0.5))) if (ref_phone_frames & mapped).any() else None

    return {
        "aligned_reliable_frames": int(both_reliable.sum()),
        "f0_bias_st": f0_bias_st,
        "f0_mae_st": f0_mae_st,
        "f0_rmse_st": f0_rmse_st,
        "f0_corr": f0_corr,
        "f0_slope_mae_st_s": slope_mae,
        "ref_f0_std_st": r_std,
        "gen_f0_std_st": g_std,
        "f0_std_ratio": g_std / max(r_std, 1e-6),
        "ref_f0_iqr_st": r_iqr,
        "gen_f0_iqr_st": g_iqr,
        "ref_f0_p95_05_st": r_p95_05,
        "gen_f0_p95_05_st": g_p95_05,
        "ref_f0_local_std_st": r_loc_std,
        "gen_f0_local_std_st": g_loc_std,
        "f0_local_std_ratio": g_loc_std / max(r_loc_std, 1e-6),
        "ref_accent_peaks": r_pk_cnt,
        "gen_accent_peaks": g_pk_cnt,
        "ref_accent_prom_st": r_pk_prom,
        "gen_accent_prom_st": g_pk_prom,
        "ref_end_delta_st": r_end,
        "gen_end_delta_st": g_end,
        "end_delta_abs_err_st": end_err,
        "ref_energy_std_db": r_e_std,
        "gen_energy_std_db": g_e_std,
        "energy_std_ratio": g_e_std / max(r_e_std, 1e-6),
        "ref_energy_iqr_db": r_e_iqr,
        "gen_energy_iqr_db": g_e_iqr,
        "energy_corr": aligned_e_corr,
        "ref_vowel_voiced": float(np.mean(r_f0["voiced"][ref_vowel_frames])) if ref_vowel_frames.any() else 1.0,
        "gen_vowel_voiced": float(np.mean(g_f0["voiced"][gen_vowel_frames])) if gen_vowel_frames.any() else 1.0,
        "voicing_agreement": voicing_agree,
        "ref_f0_break_rate": octave_break_rate(r_cont_st, ref_vowel_frames, r_reliable),
        "gen_f0_break_rate": octave_break_rate(g_cont_st, gen_vowel_frames, g_reliable_orig),
    }


def compute_rhythm_metrics(ref_durations: np.ndarray, gen_durations: np.ndarray,
                           tokens: np.ndarray, config: Config) -> dict:
    """Compute local and global rhythm, relative durations, pre-pausal lengthening, and pause metrics."""
    r = np.asarray(ref_durations, dtype=np.float64)
    g = np.asarray(gen_durations, dtype=np.float64)
    tok = np.asarray(tokens, dtype=np.int64)
    frame_ms = config.hop / config.sample_rate * 1000.0

    phone_idx = np.flatnonzero(np.isin(tok, list(PHONE_IDS)))
    r_ph, g_ph = r[phone_idx], g[phone_idx]
    abs_err_ms = np.abs(g_ph - r_ph) * frame_ms
    rel_r = r_ph / max(float(np.mean(r_ph)), 1e-6)
    rel_g = g_ph / max(float(np.mean(g_ph)), 1e-6)

    # Identify phrase-final phones (immediately followed by a comma/terminal/non-zero internal pause or <eos>).
    is_pause_tok = ~np.isin(tok, list(PHONE_IDS))
    r_pre_pausal = np.zeros(len(phone_idx), dtype=bool)
    g_pre_pausal = np.zeros(len(phone_idx), dtype=bool)
    for k, ti in enumerate(phone_idx):
        if ti + 1 < len(tok) and is_pause_tok[ti + 1]:
            if r[ti + 1] > 0 or tok[ti + 1] in COMMA_TOKENS or tok[ti + 1] in TERMINAL_PUNCT_TOKENS:
                r_pre_pausal[k] = True
            if g[ti + 1] > 0 or tok[ti + 1] in COMMA_TOKENS or tok[ti + 1] in TERMINAL_PUNCT_TOKENS:
                g_pre_pausal[k] = True

    vowel_mask_ph = np.isin(tok[phone_idx], list(VOWEL_IDS))
    def pre_pausal_vowel_ratio(d_ph: np.ndarray, pp: np.ndarray) -> float | None:
        fin = d_ph[vowel_mask_ph & pp]
        nonfin = d_ph[vowel_mask_ph & ~pp]
        if fin.size == 0 or nonfin.size == 0:
            return None
        return float(np.mean(fin) / max(float(np.mean(nonfin)), 1e-6))

    # Sliding 5-phone speaking-rate windows (ms / phoneme -> local tempo swings).
    def window_means(d_ph: np.ndarray, w: int = 5) -> np.ndarray:
        if len(d_ph) < w:
            return np.asarray([float(np.mean(d_ph))], dtype=np.float64)
        kernel = np.ones(w, dtype=np.float64) / w
        return np.convolve(d_ph, kernel, mode="valid") * frame_ms

    w_r = window_means(r_ph)
    w_g = window_means(g_ph)
    rate_cv_r = float(np.std(w_r) / max(float(np.mean(w_r)), 1e-6))
    rate_cv_g = float(np.std(w_g) / max(float(np.mean(w_g)), 1e-6))
    local_tempo_err = float(np.std(w_g / np.maximum(w_r, 1e-3)))

    # Internal pauses (excluding leading <bos> and trailing <eos>).
    int_p_idx = np.flatnonzero(is_pause_tok[1:-1]) + 1
    r_p_pres = r[int_p_idx] > 0
    g_p_pres = g[int_p_idx] > 0
    tp = int(np.sum(r_p_pres & g_p_pres))
    fp = int(np.sum(~r_p_pres & g_p_pres))
    fn = int(np.sum(r_p_pres & ~g_p_pres))
    tp_mae_ms = float(np.mean(np.abs(g[int_p_idx][r_p_pres & g_p_pres] - r[int_p_idx][r_p_pres & g_p_pres])) * frame_ms) if tp > 0 else None

    total_r, total_g = float(np.sum(r)), float(np.sum(g))
    speech_r, speech_g = float(np.sum(r_ph)), float(np.sum(g_ph))
    return {
        "total_dur_s_ref": total_r * frame_ms / 1000.0,
        "total_dur_s_gen": total_g * frame_ms / 1000.0,
        "total_dur_ratio": total_g / max(total_r, 1.0),
        "speech_dur_ratio": speech_g / max(speech_r, 1.0),
        "total_dur_abs_err_pct": abs(total_g - total_r) / max(total_r, 1.0) * 100.0,
        "phone_count": len(phone_idx),
        "phone_mae_ms": float(np.mean(abs_err_ms)),
        "phone_p90_ms": float(np.percentile(abs_err_ms, 90)),
        "phone_max_ms": float(np.max(abs_err_ms)),
        "phone_log_rmse": float(np.sqrt(np.mean((np.log1p(g_ph) - np.log1p(r_ph)) ** 2))),
        "phone_rel_dur_mae": float(np.mean(np.abs(rel_g - rel_r))),
        "phone_dur_corr": _safe_corr(r_ph, g_ph),
        "pre_pausal_ratio_ref": pre_pausal_vowel_ratio(r_ph, r_pre_pausal),
        "pre_pausal_ratio_gen": pre_pausal_vowel_ratio(g_ph, g_pre_pausal),
        "local_rate_cv_ref": rate_cv_r,
        "local_rate_cv_gen": rate_cv_g,
        "local_tempo_ratio_std": local_tempo_err,
        "internal_pause_ref_n": int(np.sum(r_p_pres)),
        "internal_pause_gen_n": int(np.sum(g_p_pres)),
        "pause_tp": tp,
        "pause_fp": fp,
        "pause_fn": fn,
        "internal_pause_ms_ref": float(np.sum(r[int_p_idx])) * frame_ms,
        "internal_pause_ms_gen": float(np.sum(g[int_p_idx])) * frame_ms,
        "pause_tp_mae_ms": tp_mae_ms,
        "bos_ms_ref": float(r[0]) * frame_ms,
        "bos_ms_gen": float(g[0]) * frame_ms,
        "eos_ms_ref": float(r[-1]) * frame_ms,
        "eos_ms_gen": float(g[-1]) * frame_ms,
    }


def _apply_voiced_vowels(model, tokens: torch.Tensor, durations: torch.Tensor,
                         prosody: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    ends = durations.cumsum(1)
    frame = torch.arange(frame_mask.shape[1], device=tokens.device)[None].expand(tokens.shape[0], -1).contiguous()
    owner = torch.searchsorted(ends, frame, right=True).clamp(max=tokens.shape[1] - 1)
    vowel = model.vowel_tokens[tokens].gather(1, owner) & frame_mask
    return torch.cat((prosody[..., :1], torch.where(vowel, prosody[..., 1].clamp_min(2.0), prosody[..., 1])[..., None],
                      prosody[..., 2:]), dim=-1)


@torch.inference_mode()
def oracle_synthesize(model, row: dict, item: dict, mode: str, seed: int = 0,
                      pitch_range: float = 1.0, pitch_jitter: float = 0.0,
                      voiced_vowels: bool = True, speed: float = 1.0) -> tuple[np.ndarray, dict]:
    """Run one counterfactual synthesis mode without altering any weights.

    Modes:
      - A_free:          free-running predicted durations, pauses, F0, voicing, energy.
      - B_oracle_dur:    oracle reference durations & pauses; predicted F0, voicing, energy.
      - C_oracle_dur_f0: oracle reference durations + oracle log_f0 in BOTH decode condition and vocoder.
      - D_oracle_full:   oracle reference durations + full oracle prosody (log_f0, voiced, energy).
    """
    valid_modes = ("A_free", "B_oracle_dur", "C_oracle_dur_f0", "D_oracle_full")
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")

    device = next(model.parameters()).device
    tokens = item["tokens"][None].to(device)
    stress = item["stress"][None].to(device)
    token_lengths = torch.tensor([tokens.shape[1]], dtype=torch.long, device=device)
    speakers = torch.tensor([int(item["speaker"])], dtype=torch.long, device=device)

    if mode == "A_free":
        wave_t, length_t, meta = model.synthesize(
            tokens, stress, token_lengths, speakers, speed=speed, seed=seed,
            pitch_range=pitch_range, pitch_jitter=pitch_jitter, voiced_vowels=voiced_vowels,
        )
        wave = wave_t[0, :int(length_t[0])].cpu().numpy()
        return wave, {
            "mode": mode,
            "durations": meta["durations"][0].cpu().numpy(),
            "prosody": meta["prosody"][0].cpu().numpy(),
            "condition": model.hard_prosody(meta["prosody"])[0].cpu().numpy(),
            "vocoder_f0_hz": model.frequency(meta["prosody"][..., 0])[0].cpu().numpy(),
        }

    oracle_dur = item["durations"][None].to(device)
    oracle_prosody = item["prosody"][None].to(device)
    if int(oracle_dur.sum()) != oracle_prosody.shape[1]:
        raise ValueError("oracle durations do not match oracle prosody frame count")

    hidden, mask = model.encode(tokens, stress, token_lengths, speakers)
    frames, frame_mask, frame_lengths = model.expand(hidden, oracle_dur, mask, speakers)
    prosody_hidden, prosody = model.predict_prosody(frames, frame_mask)

    if mode == "D_oracle_full":
        condition = oracle_prosody
        osc_log_f0 = oracle_prosody[..., 0]
        effective_prosody = oracle_prosody
    else:
        if voiced_vowels:
            prosody = _apply_voiced_vowels(model, tokens, oracle_dur, prosody, frame_mask)
        if mode == "B_oracle_dur":
            prosody = model.widen_pitch(prosody, frame_mask, pitch_range)
            effective_prosody = prosody
            condition = model.hard_prosody(prosody)
            osc_log_f0 = prosody[..., 0]
        else:  # C_oracle_dur_f0: inject oracle log-F0 consistently into condition AND vocoder.
            effective_prosody = torch.cat((oracle_prosody[..., :1], prosody[..., 1:]), dim=-1)
            condition = model.hard_prosody(effective_prosody)
            osc_log_f0 = oracle_prosody[..., 0]

    acoustic = model.decode(prosody_hidden, condition.to(prosody_hidden.dtype), frame_mask)
    samples = int(frame_lengths[0]) * model.config.hop
    noise = torch.randn(1, samples, generator=torch.Generator().manual_seed(seed)).to(device)
    pitch = osc_log_f0
    if pitch_jitter > 0 and mode == "B_oracle_dur":
        pitch = pitch + model.pitch_wobble(pitch.shape, pitch_jitter, seed, device)
    f0_hz = model.frequency(pitch)
    wave_t = model.vocoder(acoustic, f0_hz, frame_lengths, noise=noise)
    wave = wave_t[0, :samples].cpu().numpy()
    return wave, {
        "mode": mode,
        "durations": oracle_dur[0].cpu().numpy(),
        "prosody": effective_prosody[0].cpu().numpy(),
        "condition": condition[0].cpu().numpy(),
        "vocoder_f0_hz": f0_hz[0].cpu().numpy(),
    }


def _group_grad_norms(system: TrainingSystem) -> dict[str, float]:
    m = system.model
    groups = {
        "encoder": list(m.tokens.parameters()) + list(m.stress.parameters()) + list(m.voices.parameters()) + list(m.encoder.parameters()),
        "duration": list(m.duration.parameters()),
        "frame_embed": list(m.frame_voices.parameters()) + list(m.frame_position.parameters()),
        "prosody_stack": list(m.prosody_stack.parameters()),
        "prosody_head": list(m.prosody_head.parameters()),
        "prosody_projection": list(m.prosody_projection.parameters()),
        "acoustic_stack": list(m.acoustic_stack.parameters()),
        "vocoder": list(m.vocoder.parameters()),
        "mel_head": list(system.mel_head.parameters()),
    }
    norms = {}
    for name, params in groups.items():
        sq = 0.0
        for p in params:
            if p.grad is not None:
                sq += float(p.grad.detach().float().pow(2).sum())
        norms[name] = math.sqrt(sq)
    return norms


def measure_training_gradients(last_checkpoint_path: Path, dataset: AnnotatedDataset,
                               indices: list[int] | None = None, seed: int = 0) -> dict:
    """Empirically verify gradient flow across all modules in ``TrainingSystem.objective`` + critics (P0.C).

    Uses real trained ``system`` and ``critics`` loaded from ``last.pt``.
    """
    torch.manual_seed(seed)
    ck = torch.load(last_checkpoint_path, map_location="cpu", weights_only=True)
    config = Config.from_dict(ck["config"])

    system = TrainingSystem(config)
    critics = Discriminators()
    system.load_state_dict(ck.get("system") or ck["model"])
    if "critics" in ck:
        critics.load_state_dict(ck["critics"])
    system.train()
    critics.train()

    idx = indices or [0, 1]
    batch = collate([dataset[i] for i in idx], "cpu")
    recipe = ck.get("recipe", {})
    weights = recipe.get("weights", WEIGHTS)

    probes = {}
    probe_defs = [
        ("spectral_only", {k: (weights[k] if k in ("stft", "mel", "mel_head") else 0.0) for k in WEIGHTS}, False, 0.0),
        ("gan_only", {k: 0.0 for k in WEIGHTS}, True, 0.0),
        ("prosody_only", {k: (weights[k] if k in ("duration", "pause", "pitch", "pitch_spread", "voicing", "energy") else 0.0) for k in WEIGHTS}, False, 0.0),
        ("full_step35k_tp0", weights, True, 0.0),
        ("full_teacher_tp1", weights, True, 1.0),
    ]
    for label, w, use_gan, tp in probe_defs:
        torch.manual_seed(seed)
        system.zero_grad(set_to_none=True)
        critics.zero_grad(set_to_none=True)
        loss, _terms, aux = system.objective(batch, tp, w, random_phase=True)
        if use_gan:
            wave = aux["wave"]
            lengths = batch["frame_lengths"] * config.hop
            fake, real, g_idx = crops(
                wave, batch["wave"], lengths,
                int(recipe.get("crop_samples", 8192)), int(recipe.get("crops", 2)),
            )
            for p in critics.parameters():
                p.requires_grad_(False)
            try:
                with torch.no_grad():
                    real_feat = critics(real)
                adv, fm = generator_gan_loss(real_feat, critics(fake))
                gan = adv + float(recipe.get("feature_matching", 30.0)) * fm
                if w["stft"] > 0 or w["mel"] > 0:
                    recon_g = torch.autograd.grad(aux["reconstruction"], wave, retain_graph=True)[0].gather(1, g_idx)
                else:
                    _, _, aux_ref = system.objective(batch, tp, weights, random_phase=True)
                    recon_g = torch.autograd.grad(aux_ref["reconstruction"], aux_ref["wave"])[0].gather(1, g_idx)
                gan_g = torch.autograd.grad(gan, wave, retain_graph=True)[0].gather(1, g_idx)
                scale = (float(recipe.get("gan_ratio", 0.7)) * recon_g.norm() / gan_g.norm().clamp_min(1e-12)).clamp(
                    max=float(recipe.get("gan_scale_max", 2000.0))
                ).detach()
                (loss + scale * gan).backward()
                gan_scale_val = float(scale)
            finally:
                for p in critics.parameters():
                    p.requires_grad_(True)
        else:
            gan_scale_val = 0.0
            loss.backward()
        probes[label] = {
            "gan_scale_used": round(gan_scale_val, 3),
            "grad_norms": {k: round(v, 6) for k, v in _group_grad_norms(system).items()},
        }
    return probes
