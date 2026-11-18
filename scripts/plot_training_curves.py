"""Generate full 0-35k training and validation curves from real metrics.jsonl logs (Livrable 1)."""

import argparse
import json
from pathlib import Path
import numpy as np


MILESTONES = (
    (3000, "teacher ramp", "#6b7280", ":"),
    (8000, "GAN start", "#b91c1c", "--"),
    (10000, "teacher=0", "#7c3aed", ":"),
    (11000, "GAN ramp=1", "#ea580c", "-."),
    (20000, "Palier 1->2", "#0f766e", "--"),
)


def _load_metrics(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: row["step"])
    return rows


def _ema(values: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    acc = float(values[0])
    for i, val in enumerate(values):
        acc = alpha * float(val) + (1.0 - alpha) * acc
        out[i] = acc
    return out


def _add_milestones(ax, add_labels: bool = False):
    ylim = ax.get_ylim()
    for idx, (step, label, color, ls) in enumerate(MILESTONES):
        ax.axvline(step, color=color, linestyle=ls, linewidth=1.0, alpha=0.75)
        if add_labels:
            y = ylim[1] - (0.04 + 0.08 * (idx % 2)) * (ylim[1] - ylim[0])
            ax.text(step + 220, y, label, color=color, fontsize=7.5, va="top")


def _series(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in rows:
        val = row.get(key)
        if val is not None:
            xs.append(row["step"])
            ys.append(float(val))
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def plot_training_curves(metrics_path: Path, out_dir: Path, title_prefix: str = "fr_ref (192/640, 0-35k)") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = _load_metrics(metrics_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Spectral losses & GAN / optimization dynamics
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)

    ax = axes[0]
    for key, label, color, raw_alpha in (
        ("mel", "train mel", "#2563eb", 0.25),
        ("mel_head", "train mel_head", "#0891b2", 0.2),
    ):
        sx, sy = _series(rows, key)
        ax.plot(sx, sy, color=color, alpha=raw_alpha, linewidth=0.8)
        ax.plot(sx, _ema(sy), color=color, linewidth=1.6, label=label)
    for key, label, color, marker in (
        ("validation_teacher_forced_mel", "val mel (teacher-forced)", "#059669", "o"),
        ("validation_predicted_mel", "val mel (predicted)", "#dc2626", "s"),
    ):
        sx, sy = _series(rows, key)
        ax.plot(sx, sy, color=color, marker=marker, markersize=3.5, linewidth=1.6, label=label)
    ax.set_ylabel("Log-Mel L1")
    ax.set_title(f"{title_prefix} — Trajectoire spectrale, GAN et optimisation", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=2, fontsize=8.5)
    _add_milestones(ax, add_labels=True)

    ax = axes[1]
    sx, sy = _series(rows, "stft")
    ax.plot(sx, sy, color="#4f46e5", alpha=0.25, linewidth=0.8)
    ax.plot(sx, _ema(sy), color="#4f46e5", linewidth=1.6, label="train stft")
    for key, label, color, marker in (
        ("validation_teacher_forced_stft", "val stft (teacher-forced)", "#059669", "o"),
        ("validation_predicted_stft", "val stft (predicted)", "#dc2626", "s"),
    ):
        vx, vy = _series(rows, key)
        ax.plot(vx, vy, color=color, marker=marker, markersize=3.5, linewidth=1.6, label=label)
    ax.set_ylabel("Multi-Res STFT")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=3, fontsize=8.5)
    _add_milestones(ax)

    ax = axes[2]
    for key, label, color in (
        ("discriminator", "discriminator", "#7c3aed"),
        ("adversarial", "adversarial", "#ea580c"),
        ("feature_matching", "feature_matching", "#16a34a"),
    ):
        gx, gy = _series(rows, key)
        if gx.size:
            ax.plot(gx, _ema(gy), color=color, linewidth=1.5, label=label)
    ax.set_ylabel("GAN losses")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    scx, scy = _series(rows, "gan_scale")
    if scx.size:
        ax2.plot(scx, scy, color="#be123c", linewidth=1.2, linestyle="--", alpha=0.8, label="gan_scale (log)")
        ax2.set_yscale("log")
        ax2.set_ylabel("gan_scale (log)", color="#be123c")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", ncol=2, fontsize=8.5)
    _add_milestones(ax)

    fig.tight_layout()
    spectral_path = out_dir / "fr_ref_0_35k_spectral_gan.png"
    fig.savefig(spectral_path, dpi=140)
    plt.close(fig)

    # 2. Prosody & Rhythm progression (train vs validation, focused after initial warmup)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11.5), sharex=True)

    ax = axes[0]
    px, py = _series(rows, "pitch_error_semitones")
    ax.plot(px, py, color="#2563eb", alpha=0.22, linewidth=0.8)
    ax.plot(px, _ema(py), color="#2563eb", linewidth=1.6, label="train F0 MAE (st)")
    vpx, vpy = _series(rows, "validation_pitch_error_semitones")
    ax.plot(vpx, vpy, color="#dc2626", marker="o", markersize=3.5, linewidth=1.6, label="val F0 MAE (st)")
    ax.set_ylim(1.20, 2.25)
    ax2 = ax.twinx()
    spx, spy = _series(rows, "pitch_spread_ratio")
    vspx, vspy = _series(rows, "validation_pitch_spread_ratio")
    ax2.plot(spx, _ema(spy), color="#0d9488", linewidth=1.2, alpha=0.8, label="train spread ratio")
    ax2.plot(vspx, vspy, color="#f59e0b", marker="^", markersize=3.5, linewidth=1.5, label="val spread ratio")
    ax2.axhline(1.0, color="#374151", linewidth=0.9, linestyle=":")
    ax2.set_ylim(0.84, 1.08)
    ax.set_ylabel("Pitch MAE (semitones)")
    ax2.set_ylabel("Pitch spread ratio (1.0=ref)")
    ax.set_title(f"{title_prefix} — Zoom prosodie & rythme (post-warmup, steps 500–35k)", fontsize=12)
    ax.grid(True, alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", ncol=2, fontsize=8.5)
    _add_milestones(ax, add_labels=True)

    ax = axes[1]
    for key, label, color in (
        ("pitch", "train pitch L1", "#2563eb"),
        ("energy", "train energy L1", "#d97706"),
        ("voicing", "train voicing BCE", "#7c3aed"),
    ):
        ex, ey = _series(rows, key)
        ax.plot(ex, _ema(ey), color=color, linewidth=1.5, label=label)
    ax.set_ylim(0.06, 0.28)
    ax2 = ax.twinx()
    vx, vy = _series(rows, "voicing_accuracy")
    vvx, vvy = _series(rows, "validation_voicing_accuracy")
    ax2.plot(vx, _ema(vy), color="#16a34a", linewidth=1.2, linestyle="--", label="train voicing acc")
    ax2.plot(vvx, vvy, color="#15803d", marker="o", markersize=3.2, linewidth=1.5, label="val voicing acc")
    ax2.set_ylim(0.935, 0.975)
    ax.set_ylabel("Prosody L1 / BCE")
    ax2.set_ylabel("Voicing accuracy")
    ax.grid(True, alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", ncol=2, fontsize=8.5)
    _add_milestones(ax)

    ax = axes[2]
    for key, label, color, marker in (
        ("validation_pause_accuracy", "val pause acc", "#059669", "s"),
        ("validation_pause_length_ratio", "val pause len ratio", "#7c3aed", "^"),
        ("validation_duration_length_ratio", "val dur len ratio", "#2563eb", "d"),
    ):
        rx, ry = _series(rows, key)
        ax.plot(rx, ry, color=color, marker=marker, markersize=3.2, linewidth=1.4, label=label)
    ax.axhline(1.0, color="#374151", linewidth=0.8, linestyle=":")
    ax.set_ylim(0.90, 1.12)
    ax2 = ax.twinx()
    dex, dey = _series(rows, "validation_duration_length_error")
    dx, dy = _series(rows, "duration")
    ax2.plot(dex, dey, color="#dc2626", marker="o", markersize=3.2, linewidth=1.4, label="val dur len err")
    ax2.plot(dx, _ema(dy), color="#ea580c", linewidth=1.3, linestyle=":", label="train dur MSE (log1p)")
    ax2.set_ylim(0.04, 0.12)
    ax.set_ylabel("Ratios / pause accuracy")
    ax2.set_ylabel("Duration errors (MAE/MSE)")
    ax.set_xlabel("Step")
    ax.set_xlim(400, 35400)
    ax.grid(True, alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", ncol=3, fontsize=8.5)
    _add_milestones(ax)

    fig.tight_layout()
    prosody_path = out_dir / "fr_ref_0_35k_prosody_rhythm.png"
    fig.savefig(prosody_path, dpi=140)
    plt.close(fig)

    # 3. Runtime speed, total loss, gradient norm, learning rate & teacher_fraction
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True)

    ax = axes[0]
    lx, ly = _series(rows, "loss")
    bx, by = _series(rows, "base_loss")
    ax.plot(lx, ly, color="#dc2626", linewidth=1.3, alpha=0.75, label="total loss (incl. GAN*scale)")
    ax.plot(bx, by, color="#2563eb", linewidth=1.5, label="base_loss (recon+prosody)")
    gnx, gny = _series(rows, "grad_norm")
    ax.plot(gnx, _ema(gny), color="#ea580c", linewidth=1.2, alpha=0.8, label="grad_norm (EMA)")
    ax.set_yscale("log")
    ax.set_ylabel("Loss / Grad norm (log)")
    ax.set_title(f"{title_prefix} — Coût d'optimisation, cadence et schéma teacher/GAN", fontsize=12)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5)
    _add_milestones(ax, add_labels=True)

    ax = axes[1]
    sx, sy = _series(rows, "steps_per_second")
    tx, ty = _series(rows, "teacher_fraction")
    lrx, lry = _series(rows, "learning_rate")
    ax.plot(sx, sy, color="#0f766e", linewidth=1.5, label="steps/s (L4)")
    ax.set_ylabel("steps/s", color="#0f766e")
    ax2 = ax.twinx()
    ax2.plot(tx, ty, color="#7c3aed", linewidth=1.5, label="teacher_fraction")
    ax2.plot(lrx, lry * 1e3, color="#d97706", linewidth=1.3, linestyle="--", label="lr (x1e-3)")
    ax2.set_ylabel("teacher_fraction / lr (x1e-3)")
    ax.set_xlabel("Step")
    ax.grid(True, alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5)
    _add_milestones(ax)

    fig.tight_layout()
    system_path = out_dir / "fr_ref_0_35k_system_schedule.png"
    fig.savefig(system_path, dpi=140)
    plt.close(fig)

    summary = {
        "rows": len(rows),
        "step_min": rows[0]["step"],
        "step_max": rows[-1]["step"],
        "elapsed_palier1_s": round(next(r["elapsed_seconds"] for r in rows if r["step"] == 20000), 2),
        "elapsed_palier2_s": round(rows[-1]["elapsed_seconds"], 2),
        "plots": [str(spectral_path), str(prosody_path), str(system_path)],
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title", default="fr_ref (192/640, 0-35k)")
    args = parser.parse_args()
    summary = plot_training_curves(args.metrics, args.out_dir, args.title)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
