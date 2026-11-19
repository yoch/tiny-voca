"""Generate lightweight P6 replay summary plot into docs/experiments/assets/p6_replay_results.png."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=Path(".local-runs/p6_replay/artifacts"))
    parser.add_argument("--eval-report", type=Path, default=Path(".local-runs/p6_replay/eval_dev/p6_eval_dev_report.json"))
    parser.add_argument("--out", type=Path, default=Path("docs/experiments/assets/p6_replay_results.png"))
    args = parser.parse_args()

    rep = json.loads(args.eval_report.read_text(encoding="utf-8"))
    colors = {"C": "#dc2626", "D": "#16a34a", "P": "#2563eb"}
    labels = {"C": "C (GAN std)", "D": "D (GAN différé)", "P": "P (α=0 + clip sép.)"}

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    # Panel 1: Training monitor validation_pitch_error_semitones across 8k -> 11k
    ax = axes[0, 0]
    e06_8k_val = {"step": 8000, "validation_pitch_error_semitones": 1.4845, "validation_predicted_mel": 0.5002}
    for arm in ("C", "D", "P"):
        rows = [e06_8k_val] + [r for r in _load_metrics(args.artifacts_dir / arm / "metrics.jsonl") if "validation_pitch_error_semitones" in r]
        if rows:
            ax.plot([r["step"] for r in rows], [r["validation_pitch_error_semitones"] for r in rows],
                    marker="o", linewidth=2.0, color=colors[arm], label=labels[arm])
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation F0 MAE (semitones)")
    ax.set_title("Monitor 32 : trajectoire F0 pendant le replay 8k → 11k")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5)

    # Panel 2: Monitor validation_predicted_mel across 8k -> 11k
    ax = axes[0, 1]
    for arm in ("C", "D", "P"):
        rows = [e06_8k_val] + [r for r in _load_metrics(args.artifacts_dir / arm / "metrics.jsonl") if "validation_predicted_mel" in r]
        if rows:
            ax.plot([r["step"] for r in rows], [r["validation_predicted_mel"] for r in rows],
                    marker="s", linewidth=2.0, color=colors[arm], label=labels[arm])
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation predicted Mel L1")
    ax.set_title("Monitor 32 : qualité spectrale (predicted Mel) 8k → 11k")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8.5)

    # Panel 3: Primary criterion — Host fr_ref_35k common_mask_f0_corr (All / F1 / M1)
    ax = axes[1, 0]
    inj = rep["injected_summary"]
    names = ["e06_8k", "C", "D", "P"]
    n_labels = ["8k\n(base)", "C\n(GAN std)", "D\n(GAN off)", "P\n(α=0+sep)"]
    x = np.arange(len(names))
    r_f1 = [inj[k]["F1"]["common_mask_f0_corr"] for k in names]
    r_m1 = [inj[k]["M1"]["common_mask_f0_corr"] for k in names]
    r_all = [inj[k]["all"]["common_mask_f0_corr"] for k in names]
    ax.bar(x - 0.25, r_f1, width=0.24, color="#db2777", label="F1")
    ax.bar(x, r_m1, width=0.24, color="#1d4ed8", label="M1")
    ax.bar(x + 0.25, r_all, width=0.24, color="#0f766e", label="All")
    ax.set_xticks(x)
    ax.set_xticklabels(n_labels, fontsize=9)
    lo_y = max(0.0, min(r_f1 + r_m1 + r_all) - 0.05)
    hi_y = min(1.0, max(r_f1 + r_m1 + r_all) + 0.05)
    ax.set_ylim(lo_y, hi_y)
    ax.set_ylabel("Common-mask F0 correlation (r)")
    ax.set_title(f"Critère principal : F0 injectée dans fr_ref_35k (P4-dev, cov={rep['injected_common_mask_coverage']*100:.1f}%)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    # Panel 4: Paired contrasts D - C and P - C with 95% CIs (primary corr + native Mel-DTW)
    ax = axes[1, 1]
    c_dc = rep["injected_contrasts_95ci"]["D_minus_C"]["all"]["common_mask_f0_corr"]
    c_pc = rep["injected_contrasts_95ci"]["P_minus_C"]["all"]["common_mask_f0_corr"]
    f1_dc = rep["injected_contrasts_95ci"]["D_minus_C"]["F1"]["common_mask_f0_corr"]
    f1_pc = rep["injected_contrasts_95ci"]["P_minus_C"]["F1"]["common_mask_f0_corr"]
    m1_dc = rep["injected_contrasts_95ci"]["D_minus_C"]["M1"]["common_mask_f0_corr"]
    m1_pc = rep["injected_contrasts_95ci"]["P_minus_C"]["M1"]["common_mask_f0_corr"]

    items = [
        ("D − C (All)", c_dc, "#16a34a"),
        ("D − C (F1)", f1_dc, "#4ade80"),
        ("D − C (M1)", m1_dc, "#15803d"),
        ("P − C (All)", c_pc, "#2563eb"),
        ("P − C (F1)", f1_pc, "#60a5fa"),
        ("P − C (M1)", m1_pc, "#1e40af"),
    ]
    ys = np.arange(len(items))
    for i, (lbl, ci, col) in enumerate(items):
        m = ci["mean_delta"]
        err_lo = m - ci["ci95_low"]
        err_hi = ci["ci95_high"] - m
        ax.errorbar(m, i, xerr=[[err_lo], [err_hi]], fmt="o", color=col, capsize=4, linewidth=2.0)
    ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([it[0] for it in items], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Δ common_mask_f0_corr (IC 95 % bootstrap texte stratifié)")
    ax.set_title("Contrastes appariés D − C et P − C (critère principal)")
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140)
    plt.close(fig)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
