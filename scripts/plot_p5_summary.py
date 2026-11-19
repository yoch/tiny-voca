"""Generate lightweight P5 visual proof figures into docs/experiments/assets/."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_f0_causal(report: dict, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9))

    # Panel 1: 2x2 dec x osc matrix on oracle durations (mel DTW & common-mask F0 MAE)
    ax = axes[0]
    conds = ["M00_dec35k_osc35k", "M10_decRef_osc35k", "M01_dec35k_oscRef", "M11_decRef_oscRef"]
    labels = ["dec=35k\nosc=35k\n(B)", "dec=ref\nosc=35k\n(dec only)", "dec=35k\nosc=ref\n(osc only)", "dec=ref\nosc=ref\n(C)"]
    mel_all = [report["oracle_duration_summary"][c]["all"]["mel_dtw"] for c in conds]
    f0_all = [report["oracle_duration_summary"][c]["all"]["common_mask_f0_mae_st"] for c in conds]
    x = np.arange(len(conds))
    ax2 = ax.twinx()
    b1 = ax.bar(x - 0.18, f0_all, width=0.35, color="#dc2626", alpha=0.85, label="Common-mask F0 MAE (st)")
    b2 = ax2.bar(x + 0.18, mel_all, width=0.35, color="#2563eb", alpha=0.85, label="Mel DTW")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("F0 MAE (semitones)", color="#dc2626")
    ax2.set_ylabel("Mel DTW", color="#2563eb")
    ax2.set_ylim(0.42, 0.57)
    ax.set_title("2A: Decoupled F0 (dec vs osc)\n98–99% of gain flows through oscillator")
    ax.grid(True, axis="y", alpha=0.25)
    lines = [b1, b2]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=7.5)

    # Panel 2: Pre-GAN 8k vs post-GAN donors injected into mature 35k (pred_dur common-mask r)
    ax = axes[1]
    p_conds = ["P_35k_f0_on_35k_dur", "P_fr10k_f0_on_35k_dur", "P_e06_20k_f0_on_35k_dur", "P_e06_8k_f0_on_35k_dur"]
    p_labels = ["35k\n(ref)", "fr10k\n(early GAN)", "e06_20k\n(post-GAN)", "e06_8k\n(pre-GAN)"]
    r_f1 = [report["pred_duration_summary"][c]["F1"]["common_mask_f0_corr"] for c in p_conds]
    r_m1 = [report["pred_duration_summary"][c]["M1"]["common_mask_f0_corr"] for c in p_conds]
    r_all = [report["pred_duration_summary"][c]["all"]["common_mask_f0_corr"] for c in p_conds]
    xp = np.arange(len(p_conds))
    ax.bar(xp - 0.25, r_f1, width=0.24, color="#db2777", label="F1")
    ax.bar(xp, r_m1, width=0.24, color="#1d4ed8", label="M1")
    ax.bar(xp + 0.25, r_all, width=0.24, color="#0f766e", label="All")
    ax.set_xticks(xp)
    ax.set_xticklabels(p_labels, fontsize=8.5)
    ax.set_ylim(0.55, 0.87)
    ax.set_ylabel("Common-mask F0 correlation (r)")
    ax.set_title("2B: Donor F0 in mature 35k\n(identical 35k predicted durations)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 3: Heard F0 decomposition — Predictor vs Ref vs Oscillator->WAV gap
    ax = axes[2]
    m00 = report["oracle_duration_summary"]["M00_dec35k_osc35k"]
    m11 = report["oracle_duration_summary"]["M11_decRef_oscRef"]
    spks = ["F1", "M1"]
    pred_err = [m00[s]["pred_vs_ref_yin_mae_st"] for s in spks]
    osc_wav = [m00[s]["wav_vs_osc_mae_st"] for s in spks]
    tgt_yin = [m11[s]["pred_vs_ref_yin_mae_st"] for s in spks]
    xs = np.arange(len(spks))
    ax.bar(xs - 0.25, pred_err, width=0.24, color="#dc2626", label="Pred vs Ref YIN (predictor err)")
    ax.bar(xs, osc_wav, width=0.24, color="#f59e0b", label="WAV vs Osc cmd (vocoder gap)")
    ax.bar(xs + 0.25, tgt_yin, width=0.24, color="#64748b", label="Stored pYIN vs Ref YIN (label gap)")
    for i in range(2):
        ax.text(xs[i] - 0.25, pred_err[i] + 0.05, f"{pred_err[i]:.2f}", ha="center", fontsize=8)
        ax.text(xs[i], osc_wav[i] + 0.05, f"{osc_wav[i]:.2f}", ha="center", fontsize=8)
        ax.text(xs[i] + 0.25, tgt_yin[i] + 0.05, f"{tgt_yin[i]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(spks, fontsize=10)
    ax.set_ylabel("MAE (semitones)")
    ax.set_ylim(0, 2.75)
    ax.set_title("3A: Error budget of heard F0\nPredictor dominates (6x vocoder, 9x labels)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_questions_and_minimal_pairs(report: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    # Panel 1: Questions totales vs partielles terminal inflection (ref vs 35k)
    ax = axes[0]
    qs = report["questions_validation_summary"]
    groups = [("totale", "F1"), ("totale", "M1"), ("partielle", "F1"), ("partielle", "M1")]
    g_labels = ["Totale\nF1 (n=25)", "Totale\nM1 (n=25)", "Partielle\nF1 (n=22)", "Partielle\nM1 (n=22)"]
    ref_end = [qs[q][s]["ref_end_mean_st"] for q, s in groups]
    gen_end = [qs[q][s]["gen_end_mean_st"] for q, s in groups]
    dir_acc = [qs[q][s]["dir_match_acc"] for q, s in groups]
    x = np.arange(len(groups))
    ax.bar(x - 0.18, ref_end, width=0.34, color="#111827", label="Référence humaine")
    ax.bar(x + 0.18, gen_end, width=0.34, color="#e11d48", label="fr_ref_35k")
    ax.axhline(0.0, color="#64748b", linewidth=0.9)
    for i, acc in enumerate(dir_acc):
        top = max(ref_end[i], gen_end[i], 0.0) + 0.18
        ax.text(x[i], top, f"dirAcc={acc*100:.0f}%", ha="center", fontsize=8, color="#0f172a")
    ax.set_xticks(x)
    ax.set_xticklabels(g_labels, fontsize=8.5)
    ax.set_ylabel("Inflexion terminale moyenne (demi-tons)")
    ax.set_ylim(-1.0, 3.6)
    ax.set_title("3B: Questions (validation.jsonl appariées)\n35k sur-élève les totales et lève trop les partielles")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    # Panel 2: French minimal pairs (terminal delta st for decl vs qtot vs excl/qpart)
    ax = axes[1]
    mp = report["minimal_pairs"]
    pairs = [
        ("Vous venez.\n/ ? / !", "vous_venez_decl", "vous_venez_qtot"),
        ("Il est déjà parti.\n/ ?", "deja_parti_decl", "deja_parti_qtot"),
        ("Tu as trouvé la clé.\n/ ?", "trouve_cle_decl", "trouve_cle_qtot"),
        ("On part demain\nmatin. / ?", "demain_matin_decl", "demain_matin_qtot"),
        ("C'est vraiment\nfini. / ?", "vraiment_fini_decl", "vraiment_fini_qtot"),
    ]
    by_key = {(r["label"], r["speaker"]): r for r in mp if r.get("system") == "fr_ref_35k"}
    xm = np.arange(len(pairs))
    f1_d = [by_key[(d, "F1")]["end_inflection_st"] for _, d, _ in pairs]
    f1_q = [by_key[(q, "F1")]["end_inflection_st"] for _, _, q in pairs]
    m1_d = [by_key[(d, "M1")]["end_inflection_st"] for _, d, _ in pairs]
    m1_q = [by_key[(q, "M1")]["end_inflection_st"] for _, _, q in pairs]
    w = 0.19
    ax.bar(xm - 1.5 * w, f1_d, width=w, color="#f472b6", label="F1 Déclaratif (.)")
    ax.bar(xm - 0.5 * w, f1_q, width=w, color="#be185d", label="F1 Question (?)")
    ax.bar(xm + 0.5 * w, m1_d, width=w, color="#60a5fa", label="M1 Déclaratif (.)")
    ax.bar(xm + 1.5 * w, m1_q, width=w, color="#1d4ed8", label="M1 Question (?)")
    ax.axhline(0.0, color="#64748b", linewidth=0.9)
    ax.set_xticks(xm)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=8)
    ax.set_ylabel("Inflexion terminale (demi-tons)")
    ax.set_title("3C: Paires minimales françaises (fr_ref_35k)\nContraste net . vs ? sur les 2 voix")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_m1_pitch_audit(m1_report: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    zones = ["low_vowel_core", "high_vowel_core", "voiced_consonant", "transition", "voiceless_or_sil"]
    z_labels = ["Low vowel\n(<135Hz)", "High vowel\n(>=135Hz)", "Voiced\ncons.", "Phoneme\ntransition", "Voiceless\n/ silence"]

    # Panel 1: Raw disagreement counts by zone
    ax = axes[0]
    yin_only = [m1_report["zone_arbitration"][z]["disagreement_counts"].get("yin_only", 0) for z in zones]
    pyin_only = [m1_report["zone_arbitration"][z]["disagreement_counts"].get("pyin_only", 0) for z in zones]
    x = np.arange(len(zones))
    ax.bar(x - 0.18, yin_only, width=0.35, color="#2563eb", label="YIN=V, pYIN=U (yin_only)")
    ax.bar(x + 0.18, pyin_only, width=0.35, color="#d97706", label="pYIN=V, YIN=U (pyin_only)")
    ax.set_xticks(x)
    ax.set_xticklabels(z_labels, fontsize=8.5)
    ax.set_ylabel("Frames (M1 validation.jsonl, 55,730 total)")
    ax.set_title("Item 4: M1 pYIN vs YIN+harm disagreements by zone")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    # Panel 2: Independent arbitration (AC+AMDF & 0-4 kHz CPP consensus) on yin_only frames
    ax = axes[1]
    cons_v = [m1_report["zone_arbitration"][z]["by_type"]["yin_only"].get("consensus_voiced_rate", 0.0) * 100 for z in zones]
    cons_u = [m1_report["zone_arbitration"][z]["by_type"]["yin_only"].get("consensus_unvoiced_rate", 0.0) * 100 for z in zones]
    ambig = [m1_report["zone_arbitration"][z]["by_type"]["yin_only"].get("ambiguous_voicing_rate", 0.0) * 100 for z in zones]
    ax.bar(x, cons_v, color="#16a34a", label="Confirmed VOICED (AC & CEP)")
    ax.bar(x, ambig, bottom=cons_v, color="#fbbf24", label="Ambiguous (AC vs CEP)")
    ax.bar(x, cons_u, bottom=np.array(cons_v) + np.array(ambig), color="#dc2626", label="Confirmed UNVOICED (false voicing)")
    ax.set_xticks(x)
    ax.set_xticklabels(z_labels, fontsize=8.5)
    ax.set_ylabel("% of yin_only frames")
    ax.set_ylim(0, 128)
    ax.set_title("Independent arbitration of YIN-only frames\n(True recovery in low vowels vs bleed in transitions/voiceless)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right", ncol=1)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    assets = Path("docs/experiments/assets")
    assets.mkdir(parents=True, exist_ok=True)

    f0_report = json.loads(Path(".local-runs/p5_audit/f0_causal/p5_f0_causal_report.json").read_text())
    plot_f0_causal(f0_report, assets / "p5_f0_causal_matrix.png")
    plot_questions_and_minimal_pairs(f0_report, assets / "p5_questions_minimal_pairs.png")

    m1_path = Path(".local-runs/p5_audit/m1_pitch/p5_m1_pitch_audit.json")
    if m1_path.exists():
        m1_report = json.loads(m1_path.read_text())
        plot_m1_pitch_audit(m1_report, assets / "p5_m1_pitch_breakdown.png")
        src_sheet = Path(".local-runs/p5_audit/m1_pitch/manual_waveform_inspection.png")
        if src_sheet.exists():
            shutil.copy2(src_sheet, assets / "p5_m1_manual_waveforms.png")
    print("Generated P5 summary plots in", assets)


if __name__ == "__main__":
    main()
