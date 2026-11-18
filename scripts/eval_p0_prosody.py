"""Run full P0 expressivity diagnosis (0 GPU): P0.A paired reference analysis,
P0.B A/B/C/D oracle counterfactuals, P0.C real checkpoint gradient audit,
P1 pitch_range 1.0 vs 1.15 diagnosis, and Livrable 2 aligned plots + blind WAVs.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from tiny_voca.config import Config
from tiny_voca.data import AnnotatedDataset
from tiny_voca.evaluate import dtw_distance, pcm16
from tiny_voca.infer import load_model
from tiny_voca.losses import SpectralLoss
from tiny_voca.text import PHONE_SET, SYMBOL_TO_ID, SYMBOLS, VOWEL_SET

from p0_expressivity import (
    SEMITONES,
    compute_intonation_metrics,
    compute_rhythm_metrics,
    extract_unbiased_f0,
    measure_training_gradients,
    oracle_synthesize,
    select_representative_utterances,
    warped_token_alignment,
)

VOWEL_SET_STR = VOWEL_SET
PLOSIVES = frozenset(("p", "t", "k", "b", "d", "ɡ"))
FRICATIVES = frozenset(("f", "v", "s", "z", "ʃ", "ʒ"))
NASAL_LIQUID = frozenset(("m", "n", "ɲ", "ŋ", "l", "ʁ", "j", "w"))


def _phone_class(sym: str) -> str | None:
    if sym in VOWEL_SET_STR:
        return "vowel"
    if sym in PLOSIVES:
        return "plosive"
    if sym in FRICATIVES:
        return "fricative"
    if sym in NASAL_LIQUID:
        return "sonorant"
    return None


def _mean_opt(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return round(float(np.mean(clean)), 4) if clean else None


def _median_opt(values: list[float | None]) -> float | None:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return round(float(np.median(clean)), 4) if clean else None


def _summarize_mode(rows: list[dict], filter_fn=None) -> dict:
    sub = [r for r in rows if (filter_fn is None or filter_fn(r))]
    if not sub:
        return {"n": 0}
    keys = (
        "mel_dtw", "mel_l1", "stft_loss", "f0_mae_st", "f0_rmse_st", "f0_corr", "f0_slope_mae_st_s",
        "ref_f0_std_st", "gen_f0_std_st", "f0_std_ratio", "ref_f0_local_std_st", "gen_f0_local_std_st",
        "f0_local_std_ratio", "ref_accent_peaks", "gen_accent_peaks", "ref_accent_prom_st", "gen_accent_prom_st",
        "end_delta_abs_err_st", "ref_energy_std_db", "gen_energy_std_db", "energy_std_ratio", "energy_corr",
        "ref_vowel_voiced", "gen_vowel_voiced", "voicing_agreement", "ref_f0_break_rate", "gen_f0_break_rate",
    )
    out = {"n": len(sub)}
    for k in keys:
        out[k] = _mean_opt([r.get(k) for r in sub])
    return out


def _plot_utterance_comparison(item: dict, row: dict, ref_mel: np.ndarray,
                               ref_f0: dict, mode_outputs: dict,
                               config: Config, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_ref = int(item["durations"].sum())
    fps = config.sample_rate / config.hop
    sec_ref = np.arange(t_ref) / fps
    r_dur = item["durations"].numpy()
    tok_syms = [SYMBOLS[int(t)] for t in item["tokens"].numpy()]
    ends = np.cumsum(r_dur) / fps
    starts = ends - r_dur / fps

    fig, axes = plt.subplots(4, 1, figsize=(max(12, t_ref / 22), 10.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.5, 1.2, 1.1]})

    # Panel 1: Reference mel spectrogram + boundaries + pause shadings.
    ax = axes[0]
    ax.imshow(ref_mel.T, origin="lower", aspect="auto", cmap="magma",
              extent=(0, t_ref / fps, 0, config.n_mels), vmin=-11, vmax=2)
    for sym, s, e, d in zip(tok_syms, starts, ends, r_dur):
        if d == 0:
            continue
        is_ph = sym in PHONE_SET
        ax.axvline(s, color="white", linewidth=0.5, alpha=0.8)
        if not is_ph:
            for a_i in axes:
                a_i.axvspan(s, e, color="cyan", alpha=0.14)
        lbl = {"<bos>": "BOS", "<eos>": "EOS", "<sp>": "sp"}.get(sym, sym)
        ax.text((s + e) / 2, config.n_mels + 1.5, lbl, ha="center", va="bottom",
                fontsize=7.5, color="black" if is_ph else "teal")
    ax.set_ylabel("Mel bin")
    ax.set_title(f"{row['id']} ({row['speaker']}, {row['category']}) — {row['text']}", fontsize=10.5, pad=20)

    # Panel 2: Aligned log-F0 trajectories (Hz) on reliable voiced cores.
    ax = axes[1]
    ref_hz = np.where(ref_f0["voiced"], config.pitch_reference_hz * np.exp(ref_f0["continuous_log_f0"]), np.nan)
    ax.plot(sec_ref, ref_hz, color="#111827", linewidth=2.2, label="Référence (YIN fiable)")

    # Overlay stored pyin to illustrate pyin dropouts on M1 if present.
    py_v = item["prosody"][:, 1].numpy() > 0.5
    py_hz = np.where(py_v, config.pitch_reference_hz * np.exp(item["prosody"][:, 0].numpy()), np.nan)
    ax.plot(sec_ref, py_hz, color="#9ca3af", linewidth=1.0, linestyle=":", label="Référence (pyin dataset)")

    for mode, label, color, ls in (
        ("A_free", "A: Free-running (warped)", "#dc2626", "-"),
        ("B_oracle_dur", "B: Durées oracle", "#2563eb", "--"),
        ("C_oracle_dur_f0", "C: Durées + F0 oracle", "#16a34a", "-."),
    ):
        mo = mode_outputs[mode]
        f0_m = mo["f0_info"]
        w_log, mapped = warped_token_alignment(r_dur, mo["durations"], f0_m["continuous_log_f0"])
        w_v, _ = warped_token_alignment(r_dur, mo["durations"], f0_m["voiced"].astype(np.float64))
        hz = np.where((w_v > 0.5) & mapped, config.pitch_reference_hz * np.exp(w_log), np.nan)
        ax.plot(sec_ref, hz, color=color, linewidth=1.4, linestyle=ls, alpha=0.9, label=label)
    ax.set_ylabel("F0 (Hz)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=3, fontsize=8)

    # Panel 3: Energy contour (dB relative to utterance phone mean).
    ax = axes[2]
    phone_mask = np.repeat([s in PHONE_SET for s in tok_syms], r_dur)[:t_ref]
    def norm_db(arr: np.ndarray) -> np.ndarray:
        m = float(np.mean(arr[phone_mask])) if phone_mask.any() else 0.0
        return arr - m

    ax.plot(sec_ref, norm_db(ref_f0["rms_db"]), color="#111827", linewidth=1.8, label="Réf énergie rel.")
    for mode, label, color, ls in (
        ("A_free", "A: Free-running (warped)", "#dc2626", "-"),
        ("B_oracle_dur", "B: Durées oracle", "#2563eb", "--"),
        ("D_oracle_full", "D: Prosodie oracle", "#7c3aed", ":"),
    ):
        mo = mode_outputs[mode]
        w_rms, _ = warped_token_alignment(r_dur, mo["durations"], mo["f0_info"]["rms_db"])
        ax.plot(sec_ref, norm_db(w_rms), color=color, linewidth=1.2, linestyle=ls, label=label)
    ax.set_ylabel("Rel. dB")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", ncol=2, fontsize=8)

    # Panel 4: Per-token local speaking rate ratio (dur_A / dur_ref) showing rhythmic speedup/slowdown.
    ax = axes[3]
    a_dur = mode_outputs["A_free"]["durations"]
    mids, vals, colors = [], [], []
    for sym, s, e, dr, dg in zip(tok_syms, starts, ends, r_dur, a_dur):
        if sym in PHONE_SET and dr > 0:
            mids.append((s + e) / 2)
            vals.append(float(dg / dr))
            colors.append("#2563eb" if dg >= dr else "#ea580c")
    ax.bar(mids, vals, width=0.025, color=colors, alpha=0.75)
    ax.axhline(1.0, color="#111827", linewidth=1.0, linestyle="--")
    ax.set_ylim(0.0, max(2.2, max(vals, default=1.5) * 1.1))
    ax.set_ylabel("Dur A / Réf")
    ax.set_xlabel("Temps aligné sur la référence (s)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=135)
    plt.close(fig)


def _plot_p0_summary(mode_summaries: dict, rhythm_rows: list[dict], class_errors: dict, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))

    # 1. Counterfactual progression A -> B -> C -> D on F0 MAE & mel DTW
    ax = axes[0, 0]
    modes = ["A_free", "B_oracle_dur", "C_oracle_dur_f0", "D_oracle_full"]
    labels = ["A\nFree", "B\n+Dur", "C\n+Dur+F0", "D\nFull Oracle"]
    f0_maes = [mode_summaries[m]["all"]["f0_mae_st"] or 0.0 for m in modes]
    f0_corrs = [mode_summaries[m]["all"]["f0_corr"] or 0.0 for m in modes]
    mel_dtws = [mode_summaries[m]["all"]["mel_dtw"] or 0.0 for m in modes]
    x = np.arange(len(modes))
    w = 0.28
    ax.bar(x - w, f0_maes, width=w, color="#dc2626", label="F0 MAE (st)")
    ax.bar(x, mel_dtws, width=w, color="#2563eb", label="Mel DTW")
    ax.bar(x + w, f0_corrs, width=w, color="#059669", label="F0 Corr (r)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Contrefactuels oracle A -> B -> C -> D (40 énoncés test)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    # 2. Local F0 std & Energy std ratios by voice
    ax = axes[0, 1]
    x = np.arange(len(modes))
    for i, (spk, col) in enumerate((("F1", "#db2777"), ("M1", "#1d4ed8"))):
        vals_f0 = [mode_summaries[m][spk]["f0_local_std_ratio"] or 0.0 for m in modes]
        vals_en = [mode_summaries[m][spk]["energy_std_ratio"] or 0.0 for m in modes]
        ax.plot(x, vals_f0, marker="o", color=col, linewidth=1.8, label=f"{spk} F0 local std ratio")
        ax.plot(x, vals_en, marker="s", color=col, linestyle="--", linewidth=1.4, alpha=0.8, label=f"{spk} Energy std ratio")
    ax.axhline(1.0, color="#111827", linestyle=":", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Variabilité locale F0 & énergie (ratio vs Réf = 1.0)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    # 3. Phoneme duration errors by phonetic class
    ax = axes[1, 0]
    classes = ("vowel", "plosive", "fricative", "sonorant")
    maes = [float(np.mean(class_errors[k]["abs_ms"])) if class_errors[k]["abs_ms"] else 0.0 for k in classes]
    p90s = [float(np.percentile(class_errors[k]["abs_ms"], 90)) if class_errors[k]["abs_ms"] else 0.0 for k in classes]
    xc = np.arange(len(classes))
    ax.bar(xc - 0.18, maes, width=0.36, color="#ea580c", label="MAE (ms)")
    ax.bar(xc + 0.18, p90s, width=0.36, color="#f59e0b", label="p90 |err| (ms)")
    ax.set_xticks(xc)
    ax.set_xticklabels(["Voyelles", "Occlusives", "Fricatives", "Sonantes"])
    ax.set_ylabel("Erreur absolue de durée (ms)")
    ax.set_title("Erreur de durée par classe phonémique (A vs Réf)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    # 4. Total duration error vs internal phone error & local tempo CV
    ax = axes[1, 1]
    tot_errs = [r["total_dur_abs_err_pct"] for r in rhythm_rows]
    rel_errs = [r["phone_rel_dur_mae"] * 100.0 for r in rhythm_rows]
    ax.scatter(tot_errs, rel_errs, c=["#db2777" if r["speaker"] == "F1" else "#1d4ed8" for r in rhythm_rows], alpha=0.75)
    ax.set_xlabel("Erreur durée totale (%)")
    ax.set_ylabel("Erreur durée relative interne (%)")
    ax.set_title("Pourquoi une bonne durée totale masque les écarts de rythme")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path(".local-runs/colab/fr_ref_35k/fr_ref_35k_final/inference.pt"))
    parser.add_argument("--last-checkpoint", type=Path, default=Path(".local-runs/colab/fr_ref_35k/fr_ref_35k_final/last.pt"))
    parser.add_argument("--annotations", type=Path, default=Path(".local-runs/colab/fr4k/annotations_e01"))
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--per-category-pairs", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path(".local-runs/p0_expressivity"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, "cpu", check_frontend=True)
    c = model.config
    spectral = SpectralLoss(c.sample_rate)

    manifest_path = args.annotations / f"{args.split}.jsonl"
    ds = AnnotatedDataset(manifest_path, c, cache=False)
    id_to_idx = {r["id"]: i for i, r in enumerate(ds.rows)}
    selected = select_representative_utterances(manifest_path, per_category_pairs=args.per_category_pairs, speakers=c.speakers)

    modes = [
        ("A_free", "A_free", 1.0),
        ("B_oracle_dur", "B_oracle_dur", 1.0),
        ("C_oracle_dur_f0", "C_oracle_dur_f0", 1.0),
        ("D_oracle_full", "D_oracle_full", 1.0),
        ("P1_range_115", "A_free", 1.15),
    ]

    per_mode_rows = defaultdict(list)
    rhythm_rows = []
    class_errors = defaultdict(lambda: {"abs_ms": [], "rel_ratio": []})
    pyin_vs_yin = defaultdict(list)

    # Pick 6 representative examples for aligned plots & blind listening (3 F1, 3 M1 across question, exclamation, comma_pause).
    plot_ids = set()
    for cat in ("question", "exclamation", "comma_pause"):
        cat_f1 = [r for r in selected if r["speaker"] == "F1" and r["category"] == cat]
        cat_m1 = [r for r in selected if r["speaker"] == "M1" and r["category"] == cat]
        if cat_f1:
            plot_ids.add(cat_f1[len(cat_f1) // 2]["id"])
        if cat_m1:
            plot_ids.add(cat_m1[len(cat_m1) // 2]["id"])

    blind_dir = args.out_dir / "blind_listening"
    blind_dir.mkdir(parents=True, exist_ok=True)
    blind_key = {}
    rng = random.Random(20261118)

    for row in selected:
        idx = id_to_idx[row["id"]]
        item = ds[idx]
        ref_wave = item["wave"].numpy()
        ref_dur = item["durations"].numpy()
        tokens = item["tokens"].numpy()
        t_ref = int(ref_dur.sum())
        ref_f0 = extract_unbiased_f0(ref_wave, c, t_ref)

        # Record pyin vs YIN+harm agreement on reference vowels (E05 check).
        vowel_mask = np.repeat([t in {SYMBOL_TO_ID[v] for v in VOWEL_SET_STR} for t in tokens], ref_dur)[:t_ref]
        raw_stored = torch.load(ds.root / row["path"], map_location="cpu", weights_only=True)
        py_voiced = raw_stored["prosody"][:, 1].numpy() > 0.5
        if vowel_mask.any():
            pyin_vs_yin[f"{row['speaker']}_pyin_vowel_voiced"].append(float(np.mean(py_voiced[vowel_mask])))
            pyin_vs_yin[f"{row['speaker']}_yin_vowel_voiced"].append(float(np.mean(ref_f0["voiced"][vowel_mask])))
            both_v = py_voiced & ref_f0["voiced"] & vowel_mask
            if both_v.any():
                py_hz = c.pitch_reference_hz * np.exp(raw_stored["prosody"][:, 0].numpy())
                pyin_vs_yin[f"{row['speaker']}_agree_st_med"].append(float(np.median(np.abs(12.0 * np.log2(ref_f0["f0_hz"][both_v] / py_hz[both_v])))))

        with torch.no_grad():
            ref_mel_t = spectral.log_mel(torch.from_numpy(ref_wave)[None])
            ref_mel_np = ref_mel_t[0].numpy().T  # [T, n_mels] matching plot

        mode_outputs = {}
        for tag, mode_name, p_range in modes:
            wave_g, meta = oracle_synthesize(model, row, item, mode_name, seed=args.seed, pitch_range=p_range)
            t_gen = int(meta["durations"].sum())
            f0_g = extract_unbiased_f0(wave_g, c, t_gen)
            with torch.no_grad():
                gen_mel_t = spectral.log_mel(torch.from_numpy(wave_g)[None])[0].numpy()
                dtw_val = dtw_distance(gen_mel_t, ref_mel_t[0].numpy())
                if t_gen == t_ref:
                    stft_val, mel_val = spectral(torch.from_numpy(wave_g)[None], torch.from_numpy(ref_wave)[None], torch.tensor([t_ref * c.hop]))
                    stft_f, mel_f = float(stft_val), float(mel_val)
                else:
                    stft_f, mel_f = None, None

            im = compute_intonation_metrics(ref_wave, wave_g, ref_dur, meta["durations"], tokens, c, ref_f0_info=ref_f0, gen_f0_info=f0_g)
            im.update({"id": row["id"], "speaker": row["speaker"], "category": row["category"], "mode": tag,
                       "mel_dtw": dtw_val, "mel_l1": mel_f, "stft_loss": stft_f})
            per_mode_rows[tag].append(im)
            mode_outputs[tag] = {"wave": wave_g, "durations": meta["durations"], "f0_info": f0_g}

            if tag == "A_free":
                rm = compute_rhythm_metrics(ref_dur, meta["durations"], tokens, c)
                rm.update({"id": row["id"], "speaker": row["speaker"], "category": row["category"], "text": row["text"]})
                rhythm_rows.append(rm)
                frame_ms = c.hop / c.sample_rate * 1000.0
                for tok_id, dr, dg in zip(tokens, ref_dur, meta["durations"]):
                    cls = _phone_class(SYMBOLS[int(tok_id)])
                    if cls and dr > 0:
                        class_errors[cls]["abs_ms"].append(abs(float(dg - dr)) * frame_ms)
                        class_errors[cls]["rel_ratio"].append(float(dg / dr))

        if row["id"] in plot_ids:
            _plot_utterance_comparison(item, row, item["mel"].numpy(), ref_f0, mode_outputs, c,
                                       args.out_dir / "plots" / f"p0_diag_{row['id']}.png")
            # Save randomized blind WAV pack for listening
            utt_dir = blind_dir / row["id"]
            utt_dir.mkdir(parents=True, exist_ok=True)
            candidates = [
                ("REF", ref_wave),
                ("A_free", mode_outputs["A_free"]["wave"]),
                ("B_oracle_dur", mode_outputs["B_oracle_dur"]["wave"]),
                ("C_oracle_dur_f0", mode_outputs["C_oracle_dur_f0"]["wave"]),
                ("D_oracle_full", mode_outputs["D_oracle_full"]["wave"]),
                ("P1_range_115", mode_outputs["P1_range_115"]["wave"]),
            ]
            shuffled = list(candidates)
            rng.shuffle(shuffled)
            mapping = {}
            for idx_b, (name_b, wav_b) in enumerate(shuffled, start=1):
                fname = f"sample_{idx_b}.wav"
                sf.write(utt_dir / fname, pcm16(wav_b)[0], c.sample_rate, subtype="PCM_16")
                mapping[fname] = name_b
            blind_key[row["id"]] = {"text": row["text"], "speaker": row["speaker"], "category": row["category"], "files": mapping}

    (blind_dir / "blind_key.json").write_text(json.dumps(blind_key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Aggregate summaries across all 5 synthesis modes + categories + P1 variance breakdown
    mode_summaries = {}
    for tag, _, _ in modes:
        m_rows = per_mode_rows[tag]
        mode_summaries[tag] = {
            "all": _summarize_mode(m_rows),
            "F1": _summarize_mode(m_rows, lambda r: r["speaker"] == "F1"),
            "M1": _summarize_mode(m_rows, lambda r: r["speaker"] == "M1"),
            "by_category": {cat: _summarize_mode(m_rows, lambda r, k=cat: r["category"] == k)
                            for cat in ("plain_assert", "question", "exclamation", "comma_pause")},
        }

    # P1 check: compare P1_range_115 vs A_free per utterance split by whether A_free was under-modulated or already >= ref std!
    p1_analysis = {"F1": {}, "M1": {}, "all": {}}
    a_by_id = {r["id"]: r for r in per_mode_rows["A_free"]}
    p1_by_id = {r["id"]: r for r in per_mode_rows["P1_range_115"]}
    for spk in ("all", "F1", "M1"):
        sub_ids = [r["id"] for r in per_mode_rows["A_free"] if spk == "all" or r["speaker"] == spk]
        under = [i for i in sub_ids if a_by_id[i]["gen_f0_std_st"] < a_by_id[i]["ref_f0_std_st"]]
        over = [i for i in sub_ids if a_by_id[i]["gen_f0_std_st"] >= a_by_id[i]["ref_f0_std_st"]]
        def delta_mae(ids):
            return {
                "count": len(ids),
                "a_f0_mae": _mean_opt([a_by_id[i]["f0_mae_st"] for i in ids]),
                "p1_f0_mae": _mean_opt([p1_by_id[i]["f0_mae_st"] for i in ids]),
                "a_std_err": _mean_opt([abs(a_by_id[i]["gen_f0_std_st"] - a_by_id[i]["ref_f0_std_st"]) for i in ids]),
                "p1_std_err": _mean_opt([abs(p1_by_id[i]["gen_f0_std_st"] - p1_by_id[i]["ref_f0_std_st"]) for i in ids]),
                "improved_f0_mae_n": sum(1 for i in ids if (p1_by_id[i]["f0_mae_st"] or 99) < (a_by_id[i]["f0_mae_st"] or 99)),
                "regressed_f0_mae_n": sum(1 for i in ids if (p1_by_id[i]["f0_mae_st"] or 99) > (a_by_id[i]["f0_mae_st"] or 99)),
            }
        p1_analysis[spk] = {"under_modulated_at_1.0": delta_mae(under), "already_wide_at_1.0": delta_mae(over)}

    # Summarize rhythm metrics
    rhythm_summary = {}
    for spk in ("all", "F1", "M1"):
        sub = [r for r in rhythm_rows if spk == "all" or r["speaker"] == spk]
        tp_sum = sum(r["pause_tp"] for r in sub)
        fp_sum = sum(r["pause_fp"] for r in sub)
        fn_sum = sum(r["pause_fn"] for r in sub)
        prec = tp_sum / max(tp_sum + fp_sum, 1)
        rec = tp_sum / max(tp_sum + fn_sum, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        rhythm_summary[spk] = {
            "n": len(sub),
            "total_dur_ratio": _mean_opt([r["total_dur_ratio"] for r in sub]),
            "total_dur_abs_err_pct": _mean_opt([r["total_dur_abs_err_pct"] for r in sub]),
            "speech_dur_ratio": _mean_opt([r["speech_dur_ratio"] for r in sub]),
            "phone_mae_ms": _mean_opt([r["phone_mae_ms"] for r in sub]),
            "phone_p90_ms": _mean_opt([r["phone_p90_ms"] for r in sub]),
            "phone_rel_dur_mae": _mean_opt([r["phone_rel_dur_mae"] for r in sub]),
            "phone_dur_corr": _mean_opt([r["phone_dur_corr"] for r in sub]),
            "pre_pausal_ratio_ref": _mean_opt([r["pre_pausal_ratio_ref"] for r in sub]),
            "pre_pausal_ratio_gen": _mean_opt([r["pre_pausal_ratio_gen"] for r in sub]),
            "local_rate_cv_ref": _mean_opt([r["local_rate_cv_ref"] for r in sub]),
            "local_rate_cv_gen": _mean_opt([r["local_rate_cv_gen"] for r in sub]),
            "local_tempo_ratio_std": _mean_opt([r["local_tempo_ratio_std"] for r in sub]),
            "pause_precision": round(prec, 4),
            "pause_recall": round(rec, 4),
            "pause_f1": round(f1, 4),
            "pause_tp_mae_ms": _mean_opt([r["pause_tp_mae_ms"] for r in sub]),
            "bos_ms_ref": _mean_opt([r["bos_ms_ref"] for r in sub]),
            "bos_ms_gen": _mean_opt([r["bos_ms_gen"] for r in sub]),
            "eos_ms_ref": _mean_opt([r["eos_ms_ref"] for r in sub]),
            "eos_ms_gen": _mean_opt([r["eos_ms_gen"] for r in sub]),
        }

    class_summary = {
        k: {
            "count": len(v["abs_ms"]),
            "mae_ms": round(float(np.mean(v["abs_ms"])), 2),
            "p90_ms": round(float(np.percentile(v["abs_ms"], 90)), 2),
            "mean_ratio": round(float(np.mean(v["rel_ratio"])), 3),
        }
        for k, v in class_errors.items()
    }

    _plot_p0_summary(mode_summaries, rhythm_rows, class_errors, args.out_dir / "plots" / "p0_summary.png")

    # P0.C gradient path verification on a real validation batch using last.pt (model + real trained critics)
    val_ds = AnnotatedDataset(args.annotations / "validation.jsonl", c, cache=False)
    from tiny_voca.train import balanced_indices
    val_idx = balanced_indices(val_ds, 8)
    grad_probes = measure_training_gradients(args.last_checkpoint, val_ds, indices=val_idx, seed=args.seed)

    report = {
        "checkpoint": str(args.checkpoint),
        "last_checkpoint": str(args.last_checkpoint),
        "split": args.split,
        "utterances": len(selected),
        "pyin_vs_yin_reliability": {k: round(float(np.mean(v)), 4) for k, v in pyin_vs_yin.items()},
        "p0a_p0b_modes": mode_summaries,
        "p0a_rhythm_summary": rhythm_summary,
        "p0a_phoneme_class_errors": class_summary,
        "p0c_gradient_probes": grad_probes,
        "p1_pitch_range_115_breakdown": p1_analysis,
        "selected_ids": [r["id"] for r in selected],
        "plot_ids": sorted(plot_ids),
    }
    (args.out_dir / "p0_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (args.out_dir / "p0_per_utterance.json").write_text(
        json.dumps({"rhythm": rhythm_rows, "modes": per_mode_rows}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
