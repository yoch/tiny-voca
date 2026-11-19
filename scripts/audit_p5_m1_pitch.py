"""P5 Item 4 — Independent M1 pitch audit (pYIN vs YIN+harmonicity) before any relabeling.

Samples disagreements between stored dataset pYIN labels and YIN+harmonicity on M1
recordings from validation.jsonl (never touching the reserved untouched test set),
stratified across:
  1. low_vowel_core     (interior vowel frames with candidate F0 < 135 Hz)
  2. high_vowel_core    (interior vowel frames with candidate F0 >= 135 Hz)
  3. voiced_consonant   (interior voiced consonant frames)
  4. transition         (first/last frame of any phoneme or voiced/unvoiced boundary)
  5. voiceless_or_sil   (interior voiceless consonant or pause/silence frames)

Arbitrates disagreements using TWO independent non-YIN/non-pYIN estimators directly
on the 24 kHz waveform:
  - Estimator A (Time-domain): Zero-phase bandpass [50, 450] Hz normalized lag
    autocorrelation R(lag) + AMDF + explicit shortest-period / subharmonic octave check
    + low-band spectral ratio.
  - Estimator B (Cepstral): Real cepstrum quefrency peak + cepstral peak prominence (CPP)
    over [50, 450] Hz.

Also renders zoomed 50 ms raw+bandpass waveform-period sheets for manual visual
verification of a concrete stratified subset of 15 disagreement frames.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from tiny_voca.config import Config
from tiny_voca.data import AnnotatedDataset
from tiny_voca.text import PHONE_SET, SYMBOLS, VOWEL_SET, VOICELESS

from p0_expressivity import SEMITONES, extract_unbiased_f0

_BP_CACHE: dict[tuple[int, float, float], tuple[np.ndarray, np.ndarray]] = {}


def _bandpass(wave: np.ndarray, sr: int, lo_hz: float = 50.0, hi_hz: float = 450.0) -> np.ndarray:
    key = (sr, lo_hz, hi_hz)
    if key not in _BP_CACHE:
        nyq = sr / 2.0
        _BP_CACHE[key] = butter(4, [lo_hz / nyq, hi_hz / nyq], btype="band")
    b, a = _BP_CACHE[key]
    if len(wave) < 32:
        return wave.astype(np.float64)
    return filtfilt(b, a, wave.astype(np.float64))


def estimate_autocorr_amdf_frame(
    wave_raw: np.ndarray,
    wave_bp: np.ndarray,
    center_sample: int,
    sr: int = 24000,
    win: int = 1200,
    fmin: float = 50.0,
    fmax: float = 450.0,
    peak_db: float = 0.0,
) -> dict:
    """Independent time-domain period estimator: bandpass autocorr + AMDF + octave check."""
    half = win // 2
    n = len(wave_raw)
    lo_idx = center_sample - half
    hi_idx = lo_idx + win
    seg_raw = np.zeros(win, dtype=np.float64)
    seg_bp = np.zeros(win, dtype=np.float64)
    s0 = max(0, lo_idx)
    s1 = min(n, hi_idx)
    if s1 > s0:
        d0 = s0 - lo_idx
        seg_raw[d0 : d0 + (s1 - s0)] = wave_raw[s0:s1]
        seg_bp[d0 : d0 + (s1 - s0)] = wave_bp[s0:s1]

    rms = float(np.sqrt(np.mean(seg_raw * seg_raw)) + 1e-12)
    rms_db = 20.0 * math.log10(max(rms, 1e-6))

    # Low-frequency energy share (<1 kHz vs <8 kHz) rejects unvoiced fricatives/hiss.
    spec = np.abs(np.fft.rfft(seg_raw * np.hanning(win))) ** 2
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    e_low = float(spec[(freqs >= 50.0) & (freqs <= 1000.0)].sum())
    e_tot = float(spec[(freqs >= 50.0) & (freqs <= 8000.0)].sum()) + 1e-12
    low_band_ratio = e_low / e_tot

    seg_bp = seg_bp - float(np.mean(seg_bp))
    energy_bp = float(np.dot(seg_bp, seg_bp))
    if energy_bp < 1e-9 or rms_db < peak_db - 36.0:
        return {
            "voiced": False,
            "f0_hz": 0.0,
            "lag": 0,
            "r_peak": 0.0,
            "amdf_norm": 1.0,
            "low_band_ratio": round(low_band_ratio, 4),
            "rms_db": round(rms_db, 2),
            "subharmonic_ratio": 0.0,
        }

    lag_min = max(10, int(math.floor(sr / fmax)))
    lag_max = min(win // 2 - 2, int(math.ceil(sr / fmin)))
    lags = np.arange(lag_min, lag_max + 1)
    r_vals = np.zeros(len(lags), dtype=np.float64)
    amdf_vals = np.ones(len(lags), dtype=np.float64)
    mean_abs = float(np.mean(np.abs(seg_bp))) + 1e-12
    for idx, lag in enumerate(lags):
        a = seg_bp[:-lag]
        b = seg_bp[lag:]
        den = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b))) + 1e-12
        r_vals[idx] = float(np.dot(a, b)) / den
        amdf_vals[idx] = float(np.mean(np.abs(a - b))) / mean_abs

    peaks, _ = find_peaks(r_vals, height=0.35, distance=12)
    if len(peaks) == 0:
        best_i = int(np.argmax(r_vals))
        best_r = float(r_vals[best_i])
        best_lag = int(lags[best_i])
        best_amdf = float(amdf_vals[best_i])
    else:
        max_r = float(np.max(r_vals[peaks]))
        # Fundamental period rule: choose the SHORTEST lag whose autocorrelation peak is
        # within 0.08 of the global max (avoids picking 2*T0 / half-F0), EXCEPT if 2*lag
        # is substantially higher (> 0.08 higher, meaning true period is 2*lag / subharmonic).
        near_max = [p for p in peaks if r_vals[p] >= max_r - 0.08]
        best_i = int(near_max[0])
        best_lag = int(lags[best_i])
        best_r = float(r_vals[best_i])
        best_amdf = float(amdf_vals[best_i])

    # Parabolic interpolation around best_i for sub-sample lag accuracy.
    refined_lag = float(best_lag)
    if 0 < best_i < len(r_vals) - 1:
        y0, y1, y2 = r_vals[best_i - 1], r_vals[best_i], r_vals[best_i + 1]
        denom = 2.0 * (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-12:
            shift = max(-0.5, min(0.5, (y0 - y2) / denom))
            refined_lag = float(best_lag) + shift

    # Subharmonic check at 2 * lag if inside search window.
    double_lag = int(round(2.0 * refined_lag))
    if lag_min <= double_lag <= lag_max:
        lo_d = max(0, double_lag - lag_min - 3)
        hi_d = min(len(r_vals), double_lag - lag_min + 4)
        r_double = float(np.max(r_vals[lo_d:hi_d]))
        sub_ratio = r_double / max(best_r, 1e-6)
    else:
        sub_ratio = 0.0

    voiced = bool(
        best_r >= 0.55
        and best_amdf <= 0.78
        and low_band_ratio >= 0.30
        and rms_db >= peak_db - 36.0
    )
    f0_hz = float(sr / refined_lag) if refined_lag > 0 else 0.0
    return {
        "voiced": voiced,
        "f0_hz": round(f0_hz, 2),
        "lag": round(refined_lag, 2),
        "r_peak": round(best_r, 4),
        "amdf_norm": round(best_amdf, 4),
        "low_band_ratio": round(low_band_ratio, 4),
        "rms_db": round(rms_db, 2),
        "subharmonic_ratio": round(sub_ratio, 4),
    }


def estimate_cepstrum_frame(
    wave_raw: np.ndarray,
    center_sample: int,
    sr: int = 24000,
    win: int = 1200,
    fmin: float = 50.0,
    fmax: float = 450.0,
    peak_db: float = 0.0,
) -> dict:
    """Independent spectral-domain estimator: speech-band (0-4 kHz) real cepstrum + CPP.

    Computing the cepstrum on the speech harmonic band (0-4 kHz, via 3x decimation to 8 kHz
    when sr=24 kHz) avoids 4-12 kHz spectral roll-off / aspiration noise diluting harmonic
    comb prominence on low male vowels (standard Hillenbrand CPP convention).
    """
    half = win // 2
    n = len(wave_raw)
    lo_idx = center_sample - half
    hi_idx = lo_idx + win
    seg = np.zeros(win, dtype=np.float64)
    s0 = max(0, lo_idx)
    s1 = min(n, hi_idx)
    if s1 > s0:
        d0 = s0 - lo_idx
        seg[d0 : d0 + (s1 - s0)] = wave_raw[s0:s1]

    rms = float(np.sqrt(np.mean(seg * seg)) + 1e-12)
    rms_db = 20.0 * math.log10(max(rms, 1e-6))
    if rms_db < peak_db - 36.0:
        return {"voiced": False, "f0_hz": 0.0, "cpp": 0.0, "lag": 0}

    # Restrict to 0-4 kHz harmonic band by keeping the low-frequency rfft bins.
    nfft = 2048
    windowed = (seg - float(np.mean(seg))) * np.hanning(win)
    mag = np.abs(np.fft.rfft(windowed, n=nfft))
    # Band-limit at 4 kHz (sr_eff = 8000 Hz when sr = 24000 -> decimate factor 3).
    decim = max(1, int(round(sr / 8000.0)))
    sr_eff = sr / decim
    nfft_eff = max(256, nfft // decim)
    n_bins = nfft_eff // 2 + 1
    mag_band = mag[:n_bins]
    # Dynamic floor relative to band peak (50 dB range) so narrowband signals don't hit 1e-7 floor.
    band_peak = float(np.max(mag_band)) + 1e-12
    log_mag = np.log(np.maximum(mag_band, band_peak * 1e-3))
    ceps = np.fft.irfft(log_mag, n=nfft_eff)

    q_min = max(8, int(math.floor(sr_eff / fmax)))
    q_max = min(nfft_eff // 2 - 2, int(math.ceil(sr_eff / fmin)))
    band = ceps[q_min : q_max + 1]
    qs = np.arange(q_min, q_max + 1, dtype=np.float64)
    slope, intercept = np.polyfit(qs, band, 1)
    baseline = slope * qs + intercept
    prominence = band - baseline
    best_i = int(np.argmax(prominence))
    cpp = float(prominence[best_i])
    best_q = float(qs[best_i])

    # Parabolic refinement of quefrency peak.
    if 0 < best_i < len(prominence) - 1:
        y0, y1, y2 = prominence[best_i - 1], prominence[best_i], prominence[best_i + 1]
        denom = 2.0 * (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-12:
            best_q += max(-0.5, min(0.5, (y0 - y2) / denom))

    # Avoid half-F0 pick if q/2 has strong cepstral prominence.
    half_q = int(round(best_q / 2.0))
    if q_min <= half_q <= q_max:
        lo_h = max(0, half_q - q_min - 2)
        hi_h = min(len(prominence), half_q - q_min + 3)
        half_best_off = int(np.argmax(prominence[lo_h:hi_h]))
        half_cpp = float(prominence[lo_h + half_best_off])
        if half_cpp >= 0.72 * cpp and half_cpp >= 0.14:
            best_i = lo_h + half_best_off
            cpp = half_cpp
            best_q = float(qs[best_i])

    # Also require meaningful low-frequency energy share so unvoiced fricatives are rejected.
    e_low = float(np.sum(mag_band[: max(4, int(round(1000.0 * nfft / sr)))] ** 2))
    e_all = float(np.sum(mag[: int(round(8000.0 * nfft / sr))] ** 2)) + 1e-12
    low_share = e_low / e_all

    voiced = bool(cpp >= 0.14 and low_share >= 0.25 and rms_db >= peak_db - 36.0)
    f0_hz = float(sr_eff / best_q) if best_q > 0 else 0.0
    return {
        "voiced": voiced,
        "f0_hz": round(f0_hz, 2),
        "cpp": round(cpp, 4),
        "lag": round(best_q * decim, 2),
    }


def _classify_zone(
    frame_idx: int,
    tok_per_frame: list[str],
    pos_in_tok: list[int],
    len_of_tok: list[int],
    cand_f0_hz: float,
    low_hz_cutoff: float = 135.0,
) -> str:
    sym = tok_per_frame[frame_idx]
    pos = pos_in_tok[frame_idx]
    dur = len_of_tok[frame_idx]
    if sym not in PHONE_SET or sym in VOICELESS:
        if pos == 0 or pos == dur - 1:
            # Boundary between a voiceless/pause token and a voiced phone is a transition.
            prev_voiced = frame_idx > 0 and (tok_per_frame[frame_idx - 1] in PHONE_SET and tok_per_frame[frame_idx - 1] not in VOICELESS)
            next_voiced = frame_idx + 1 < len(tok_per_frame) and (tok_per_frame[frame_idx + 1] in PHONE_SET and tok_per_frame[frame_idx + 1] not in VOICELESS)
            if (pos == 0 and prev_voiced) or (pos == dur - 1 and next_voiced):
                return "transition"
        return "voiceless_or_sil"
    if dur <= 2 or pos == 0 or pos == dur - 1:
        return "transition"
    if sym in VOWEL_SET:
        return "low_vowel_core" if (0.0 < cand_f0_hz < low_hz_cutoff) else "high_vowel_core"
    return "voiced_consonant"


def _semitone_diff(f_a: float, f_b: float) -> float:
    if f_a <= 0.0 or f_b <= 0.0:
        return float("nan")
    return SEMITONES * math.log(f_a / f_b)


def _arbitrate_consensus(ac: dict, cep: dict) -> dict:
    """Consensus verdict from the two independent estimators."""
    if ac["voiced"] and cep["voiced"]:
        diff_st = abs(_semitone_diff(ac["f0_hz"], cep["f0_hz"]))
        if diff_st <= 2.0:
            return {
                "verdict": "voiced",
                "f0_hz": round(0.5 * (ac["f0_hz"] + cep["f0_hz"]), 2),
                "estimators_agree": True,
                "est_diff_st": round(diff_st, 3),
            }
        # Both say voiced but disagree on octave/period — prefer AC only as weak signal, mark ambiguous.
        return {
            "verdict": "voiced_ambig_f0",
            "f0_hz": ac["f0_hz"],
            "estimators_agree": False,
            "est_diff_st": round(diff_st, 3),
        }
    if (not ac["voiced"]) and (not cep["voiced"]):
        return {"verdict": "unvoiced", "f0_hz": 0.0, "estimators_agree": True, "est_diff_st": 0.0}
    return {"verdict": "ambiguous_voicing", "f0_hz": ac["f0_hz"] if ac["voiced"] else cep["f0_hz"], "estimators_agree": False, "est_diff_st": None}


def _render_manual_inspection_sheet(
    cases: list[dict],
    waves_by_id: dict[str, tuple[np.ndarray, np.ndarray]],
    config: Config,
    out_path: Path,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(cases)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16.5, 3.2 * rows))
    axes_flat = np.array(axes).reshape(-1)
    sr = config.sample_rate
    hop = config.hop
    half_ms = 25.0
    half_samp = int(round(half_ms * 1e-3 * sr))

    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue
        c = cases[i]
        wave_raw, wave_bp = waves_by_id[c["id"]]
        center = c["frame"] * hop
        lo = max(0, center - half_samp)
        hi = min(len(wave_raw), center + half_samp)
        t_ms = (np.arange(lo, hi) - center) * 1000.0 / sr
        seg_raw = wave_raw[lo:hi]
        seg_bp = wave_bp[lo:hi]
        scale = max(float(np.max(np.abs(seg_raw))), 1e-4)

        ax.plot(t_ms, seg_raw / scale, color="#94a3b8", linewidth=0.8, alpha=0.85, label="raw")
        ax.plot(t_ms, seg_bp / max(float(np.max(np.abs(seg_bp))), 1e-4) * 0.85, color="#0f172a", linewidth=1.2, label="BP 50-450Hz")
        ax.axvline(0.0, color="#ef4444", linewidth=0.9, alpha=0.7)

        # Draw period grid lines from center for YIN and AC (and pYIN if voiced).
        if c["yin_voiced"] and c["yin_hz"] > 0:
            p_ms = 1000.0 / c["yin_hz"]
            for k in range(-3, 4):
                if k != 0 and abs(k * p_ms) <= half_ms:
                    ax.axvline(k * p_ms, color="#2563eb", linestyle="--", linewidth=0.9, alpha=0.7)
        if c["ac"]["voiced"] and c["ac"]["f0_hz"] > 0:
            p_ms = 1000.0 / c["ac"]["f0_hz"]
            for k in range(-3, 4):
                if k != 0 and abs(k * p_ms) <= half_ms:
                    ax.axvline(k * p_ms, color="#16a34a", linestyle=":", linewidth=1.2, alpha=0.85)
        if c["pyin_voiced"] and c["pyin_hz"] > 0:
            p_ms = 1000.0 / c["pyin_hz"]
            for k in range(-2, 3):
                if k != 0 and abs(k * p_ms) <= half_ms:
                    ax.axvline(k * p_ms, color="#d97706", linestyle="-.", linewidth=0.8, alpha=0.65)

        title = (
            f"#{i+1} [{c['zone']}] {c['disagree_type']} | /{c['phone']}/ ({c['context']})\n"
            f"pYIN={'V ' + str(round(c['pyin_hz'],1)) + 'Hz' if c['pyin_voiced'] else 'U'} | "
            f"YIN={'V ' + str(round(c['yin_hz'],1)) + 'Hz' if c['yin_voiced'] else 'U'} (h={c['yin_harm']:.2f}) | "
            f"AC={'V ' + str(round(c['ac']['f0_hz'],1)) if c['ac']['voiced'] else 'U'} (R={c['ac']['r_peak']:.2f}) | "
            f"CEP={'V ' + str(round(c['cep']['f0_hz'],1)) if c['cep']['voiced'] else 'U'} (cpp={c['cep']['cpp']:.2f})"
        )
        ax.set_title(title, fontsize=7.5)
        ax.set_xlim(-half_ms, half_ms)
        ax.set_ylim(-1.15, 1.15)
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc="lower right", fontsize=7)
        ax.set_xlabel("ms relative to frame center", fontsize=7)

    fig.suptitle(
        "P5 Item 4 — M1 Disagreement Manual Waveform Inspection (50 ms window: raw=gray, BP=black, YIN=blue--, AC=green:, pYIN=orange-.)",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, default=Path(".local-runs/colab/fr4k/annotations_e01/validation.jsonl"))
    parser.add_argument("--config", type=Path, default=Path("configs/fr_baseline_192_640.json"))
    parser.add_argument("--out-dir", type=Path, default=Path(".local-runs/p5_audit/m1_pitch"))
    parser.add_argument("--sample-per-zone", type=int, default=250, help="Disagreement frames sampled per zone for independent arbitration")
    parser.add_argument("--seed", type=int, default=20261119)
    args = parser.parse_args()

    config = Config.from_dict(json.loads(args.config.read_text()))
    ds = AnnotatedDataset(args.validation, config, voiced_vowels=False)  # RAW pYIN voicing!

    m1_indices = [i for i, r in enumerate(ds.rows) if r["speaker"] == "M1"]
    print(f"Auditing ALL {len(m1_indices)} M1 utterances in {args.validation} (raw pYIN vs YIN+harmonicity)...", flush=True)

    zone_totals = Counter()
    zone_disagree_counts = defaultdict(Counter)
    zone_candidates: dict[str, list[dict]] = defaultdict(list)
    waves_by_id: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    utts_stats = []

    all_yin_v_hz = []
    all_pyin_v_hz = []
    pyin_vowel_frames = 0
    pyin_vowel_voiced = 0
    yin_vowel_voiced = 0

    for idx in m1_indices:
        row = ds.rows[idx]
        item = ds[idx]
        path = (ds.root / row["path"]).resolve()
        stored = torch.load(path, map_location="cpu", weights_only=True)
        wave_raw = stored["wave"].numpy().astype(np.float64)
        wave_bp = _bandpass(wave_raw, config.sample_rate, 50.0, 450.0)
        waves_by_id[row["id"]] = (wave_raw, wave_bp)

        t_frames = int(row["frames"])
        pyin_log = stored["prosody"][:, 0].numpy().astype(np.float64)
        pyin_v = stored["prosody"][:, 1].numpy() > 0.5
        pyin_hz = np.where(pyin_v, config.pitch_reference_hz * np.exp(pyin_log), 0.0)
        trusted = item["pitch_trusted"].numpy() > 0.5

        yin = extract_unbiased_f0(wave_raw, config, frames=t_frames)
        yin_v = yin["voiced"]
        yin_hz = np.where(yin_v, yin["f0_hz"].astype(np.float64), 0.0)
        yin_harm = yin["harm"].astype(np.float64)
        peak_db = float(np.max(yin["rms_db"])) if yin["rms_db"].size else -90.0

        if yin_v.any():
            all_yin_v_hz.extend(yin_hz[yin_v].tolist())
        if pyin_v.any():
            all_pyin_v_hz.extend(pyin_hz[pyin_v].tolist())

        # Expand tokens to per-frame metadata.
        tok_syms = [SYMBOLS[int(t)] for t in row["tokens"]]
        tok_per_frame = []
        pos_in_tok = []
        len_of_tok = []
        tok_idx_per_frame = []
        for ti, (sym, dur) in enumerate(zip(tok_syms, row["durations"])):
            for p in range(int(dur)):
                tok_per_frame.append(sym)
                pos_in_tok.append(p)
                len_of_tok.append(int(dur))
                tok_idx_per_frame.append(ti)

        for f in range(t_frames):
            sym = tok_per_frame[f]
            if sym in VOWEL_SET:
                pyin_vowel_frames += 1
                pyin_vowel_voiced += int(pyin_v[f])
                yin_vowel_voiced += int(yin_v[f])

            cand_hz = float(yin_hz[f] if yin_v[f] else (pyin_hz[f] if pyin_v[f] else 0.0))
            zone = _classify_zone(f, tok_per_frame, pos_in_tok, len_of_tok, cand_hz, low_hz_cutoff=135.0)
            zone_totals[zone] += 1

            pv, yv = bool(pyin_v[f]), bool(yin_v[f])
            if yv and not pv:
                dtype = "yin_only"
            elif pv and not yv:
                dtype = "pyin_only"
            elif pv and yv:
                d_st = abs(_semitone_diff(float(yin_hz[f]), float(pyin_hz[f])))
                if d_st >= 5.0:
                    dtype = "both_voiced_octave"
                elif d_st >= 1.5:
                    dtype = "both_voiced_shift"
                else:
                    dtype = "both_voiced_agree"
            else:
                dtype = "both_unvoiced"

            zone_disagree_counts[zone][dtype] += 1
            if dtype in ("yin_only", "pyin_only", "both_voiced_octave", "both_voiced_shift"):
                ti = tok_idx_per_frame[f]
                ctx = "".join(tok_syms[max(0, ti - 2) : min(len(tok_syms), ti + 3)])
                zone_candidates[zone].append({
                    "id": row["id"],
                    "frame": f,
                    "zone": zone,
                    "disagree_type": dtype,
                    "phone": sym,
                    "context": ctx,
                    "pyin_voiced": pv,
                    "pyin_trusted": bool(trusted[f]),
                    "pyin_hz": round(float(pyin_hz[f]), 2),
                    "yin_voiced": yv,
                    "yin_hz": round(float(yin_hz[f]), 2),
                    "yin_harm": round(float(yin_harm[f]), 4),
                    "rms_db": round(float(yin["rms_db"][f]), 2),
                    "peak_db": round(peak_db, 2),
                })

        utts_stats.append({
            "id": row["id"],
            "frames": t_frames,
            "pyin_voiced_rate": round(float(pyin_v.mean()), 4),
            "yin_voiced_rate": round(float(yin_v.mean()), 4),
        })

    # Stratified sampling of disagreement frames across each zone and disagreement type.
    rng = random.Random(args.seed)
    sampled: list[dict] = []
    for zone in ("low_vowel_core", "high_vowel_core", "voiced_consonant", "transition", "voiceless_or_sil"):
        cands = zone_candidates[zone]
        by_type = defaultdict(list)
        for c in cands:
            by_type[c["disagree_type"]].append(c)
        # Allocate quota proportionally across types, ensuring rare octave errors are ALL included!
        zone_pick = []
        remaining_quota = args.sample_per_zone
        for rare_type in ("both_voiced_octave", "both_voiced_shift"):
            pool = list(by_type[rare_type])
            rng.shuffle(pool)
            take = min(len(pool), max(30, remaining_quota // 5))
            zone_pick.extend(pool[:take])
            remaining_quota -= len(pool[:take])
        major_pool = list(by_type["yin_only"]) + list(by_type["pyin_only"])
        rng.shuffle(major_pool)
        zone_pick.extend(major_pool[: max(0, remaining_quota)])
        sampled.extend(zone_pick)

    print(f"Arbitrating {len(sampled)} stratified M1 disagreement frames with Autocorrelation+AMDF & Real Cepstrum...", flush=True)
    for rec in sampled:
        wave_raw, wave_bp = waves_by_id[rec["id"]]
        center = rec["frame"] * config.hop
        rec["ac"] = estimate_autocorr_amdf_frame(wave_raw, wave_bp, center, config.sample_rate, 1200, 50.0, 450.0, rec["peak_db"])
        rec["cep"] = estimate_cepstrum_frame(wave_raw, center, config.sample_rate, 1200, 50.0, 450.0, rec["peak_db"])
        rec["consensus"] = _arbitrate_consensus(rec["ac"], rec["cep"])

    # Compute per-zone, per-disagreement-type arbitration summary + corpus-extrapolated rates!
    arbitration_summary = {}
    total_frames = sum(zone_totals.values())
    extrap = {
        "yin_true_recovered_frames": 0.0,
        "yin_false_voicing_frames": 0.0,
        "yin_ambiguous_voicing_frames": 0.0,
        "pyin_true_kept_missed_by_yin": 0.0,
        "pyin_false_voicing_frames": 0.0,
        "yin_octave_errors": 0.0,
        "pyin_octave_errors": 0.0,
        "yin_close_to_indep_on_recovered": 0.0,
    }

    for zone in ("low_vowel_core", "high_vowel_core", "voiced_consonant", "transition", "voiceless_or_sil"):
        z_sub = [r for r in sampled if r["zone"] == zone]
        z_entry = {
            "total_frames": int(zone_totals[zone]),
            "share_of_all_frames": round(zone_totals[zone] / max(total_frames, 1), 4),
            "disagreement_counts": dict(zone_disagree_counts[zone]),
            "by_type": {},
        }
        for dtype in ("yin_only", "pyin_only", "both_voiced_octave", "both_voiced_shift"):
            sub = [r for r in z_sub if r["disagree_type"] == dtype]
            corpus_n = int(zone_disagree_counts[zone][dtype])
            if not sub:
                z_entry["by_type"][dtype] = {"corpus_n": corpus_n, "sampled_n": 0}
                continue
            verdicts = Counter(r["consensus"]["verdict"] for r in sub)
            ac_v = sum(1 for r in sub if r["ac"]["voiced"])
            cep_v = sum(1 for r in sub if r["cep"]["voiced"])
            n_s = len(sub)

            # Accuracy & octave error breakdown against consensus voiced F0.
            yin_ok = 0
            yin_oct = 0
            pyin_ok = 0
            pyin_oct = 0
            n_v_cons = 0
            for r in sub:
                if r["consensus"]["verdict"] == "voiced":
                    n_v_cons += 1
                    ref_f = r["consensus"]["f0_hz"]
                    if r["yin_voiced"]:
                        d_y = abs(_semitone_diff(r["yin_hz"], ref_f))
                        if d_y <= 2.0:
                            yin_ok += 1
                        if d_y >= 5.0:
                            yin_oct += 1
                    if r["pyin_voiced"]:
                        d_p = abs(_semitone_diff(r["pyin_hz"], ref_f))
                        if d_p <= 2.0:
                            pyin_ok += 1
                        if d_p >= 5.0:
                            pyin_oct += 1

            # Extrapolate to full M1 validation corpus.
            scale = corpus_n / n_s
            if dtype == "yin_only":
                extrap["yin_true_recovered_frames"] += scale * (verdicts["voiced"] + verdicts["voiced_ambig_f0"])
                extrap["yin_false_voicing_frames"] += scale * verdicts["unvoiced"]
                extrap["yin_ambiguous_voicing_frames"] += scale * verdicts["ambiguous_voicing"]
                extrap["yin_close_to_indep_on_recovered"] += scale * yin_ok
                extrap["yin_octave_errors"] += scale * yin_oct
            elif dtype == "pyin_only":
                extrap["pyin_true_kept_missed_by_yin"] += scale * (verdicts["voiced"] + verdicts["voiced_ambig_f0"])
                extrap["pyin_false_voicing_frames"] += scale * verdicts["unvoiced"]
                extrap["pyin_octave_errors"] += scale * pyin_oct
            elif dtype in ("both_voiced_octave", "both_voiced_shift"):
                extrap["yin_octave_errors"] += scale * yin_oct
                extrap["pyin_octave_errors"] += scale * pyin_oct

            z_entry["by_type"][dtype] = {
                "corpus_n": corpus_n,
                "sampled_n": n_s,
                "consensus_voiced_rate": round((verdicts["voiced"] + verdicts["voiced_ambig_f0"]) / n_s, 4),
                "consensus_voiced_agree_f0_rate": round(verdicts["voiced"] / n_s, 4),
                "consensus_unvoiced_rate": round(verdicts["unvoiced"] / n_s, 4),
                "ambiguous_voicing_rate": round(verdicts["ambiguous_voicing"] / n_s, 4),
                "ac_voiced_rate": round(ac_v / n_s, 4),
                "cep_voiced_rate": round(cep_v / n_s, 4),
                "n_consensus_voiced": n_v_cons,
                "yin_within_2st_of_indep": round(yin_ok / max(n_v_cons, 1), 4) if n_v_cons else None,
                "yin_octave_err_rate": round(yin_oct / max(n_v_cons, 1), 4) if n_v_cons else None,
                "pyin_within_2st_of_indep": round(pyin_ok / max(n_v_cons, 1), 4) if n_v_cons else None,
                "pyin_octave_err_rate": round(pyin_oct / max(n_v_cons, 1), 4) if n_v_cons else None,
            }
        arbitration_summary[zone] = z_entry

    # Select 15 concrete cases for manual visual waveform inspection (5 low_vowel_core, 5 transition, 5 voiceless_or_sil).
    manual_cases = []
    specs = [
        ("low_vowel_core", "yin_only", 3),
        ("low_vowel_core", "both_voiced_octave", 1),
        ("low_vowel_core", "pyin_only", 1),
        ("transition", "yin_only", 3),
        ("transition", "pyin_only", 2),
        ("voiceless_or_sil", "yin_only", 3),
        ("voiceless_or_sil", "pyin_only", 2),
    ]
    used_keys = set()
    for zone, dtype, count in specs:
        pool = [r for r in sampled if r["zone"] == zone and r["disagree_type"] == dtype and (r["id"], r["frame"]) not in used_keys]
        # Prefer diverse utterances.
        rng.shuffle(pool)
        picked = []
        seen_utts = set()
        for r in pool:
            if r["id"] not in seen_utts:
                picked.append(r)
                seen_utts.add(r["id"])
                used_keys.add((r["id"], r["frame"]))
                if len(picked) == count:
                    break
        if len(picked) < count:
            for r in pool:
                if (r["id"], r["frame"]) not in used_keys:
                    picked.append(r)
                    used_keys.add((r["id"], r["frame"]))
                    if len(picked) == count:
                        break
        manual_cases.extend(picked)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = args.out_dir / "manual_waveform_inspection.png"
    _render_manual_inspection_sheet(manual_cases, waves_by_id, config, sheet_path)

    report = {
        "validation_manifest": str(args.validation),
        "m1_utterances": len(m1_indices),
        "total_m1_frames": int(total_frames),
        "m1_vowel_frames": int(pyin_vowel_frames),
        "m1_vowel_voiced_pyin_raw": round(pyin_vowel_voiced / max(pyin_vowel_frames, 1), 4),
        "m1_vowel_voiced_yin_harm": round(yin_vowel_voiced / max(pyin_vowel_frames, 1), 4),
        "m1_pyin_median_hz": round(float(np.median(all_pyin_v_hz)), 2) if all_pyin_v_hz else None,
        "m1_yin_median_hz": round(float(np.median(all_yin_v_hz)), 2) if all_yin_v_hz else None,
        "sampled_disagreement_frames": len(sampled),
        "zone_arbitration": arbitration_summary,
        "corpus_extrapolation": {
            k: {
                "frames": round(v, 1),
                "pct_of_all_m1_frames": round(100.0 * v / max(total_frames, 1), 3),
            }
            for k, v in extrap.items()
        },
        "manual_inspection_sheet": str(sheet_path),
        "manual_inspection_cases": manual_cases,
    }
    (args.out_dir / "p5_m1_pitch_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== P5 ITEM 4 — M1 PITCH AUDIT SUMMARY ===")
    print(f"M1 utterances: {len(m1_indices)} | Total frames: {total_frames}")
    print(f"Vowel voicing rate: pYIN(raw)={report['m1_vowel_voiced_pyin_raw']:.4f} vs YIN+harm={report['m1_vowel_voiced_yin_harm']:.4f}")
    print(f"Median voiced F0: pYIN={report['m1_pyin_median_hz']} Hz | YIN+harm={report['m1_yin_median_hz']} Hz")
    for zone, ze in arbitration_summary.items():
        dc = ze["disagreement_counts"]
        print(f"\n  [{zone}] total={ze['total_frames']} ({ze['share_of_all_frames']*100:.1f}%) | counts={dc}")
        for dtype, info in ze["by_type"].items():
            if info["sampled_n"] == 0:
                continue
            print(
                f"    {dtype:20s}: corpus={info['corpus_n']:5d} (n={info['sampled_n']:3d}) | "
                f"consV={info['consensus_voiced_rate']:.3f} (agreeF0={info['consensus_voiced_agree_f0_rate']:.3f}) "
                f"consU={info['consensus_unvoiced_rate']:.3f} ambig={info['ambiguous_voicing_rate']:.3f} | "
                f"AC_V={info['ac_voiced_rate']:.3f} CEP_V={info['cep_voiced_rate']:.3f} | "
                f"yin<=2st={info['yin_within_2st_of_indep']} yinOct={info['yin_octave_err_rate']} "
                f"pyin<=2st={info['pyin_within_2st_of_indep']} pyinOct={info['pyin_octave_err_rate']}"
            )
    print("\n=== CORPUS EXTRAPOLATION (M1 validation.jsonl) ===")
    for k, v in report["corpus_extrapolation"].items():
        print(f"  {k:36s}: {v['frames']:7.1f} frames ({v['pct_of_all_m1_frames']:.3f}% of all M1 frames)")
    print(f"\nManual inspection sheet written to: {sheet_path}")


if __name__ == "__main__":
    main()
