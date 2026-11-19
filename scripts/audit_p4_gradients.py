"""Mechanistic CPU gradient audit for P4 (Étape 1): norms, pairwise cosines,
global clipping dilution, 8k-11k teacher/GAN overlap, and alpha gate sweep.

Runs strictly on CPU (0 GPU) using real checkpoints and reproducible minibatches.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tiny_voca.config import Config
from tiny_voca.data import AnnotatedDataset, collate
from tiny_voca.losses import Discriminators, generator_gan_loss, masked_mean
from tiny_voca.train import TrainingSystem, WEIGHTS, balanced_indices, crops, masked_std


def _layer_groups(system: TrainingSystem) -> dict[str, list[torch.nn.Parameter]]:
    m = system.model
    groups: dict[str, list[torch.nn.Parameter]] = {}
    for i, layer in enumerate(m.encoder.blocks):
        groups[f"encoder.layer.{i}"] = list(layer.parameters())
    groups["encoder.norm"] = list(m.encoder.norm.parameters())
    groups["encoder.all"] = list(m.encoder.parameters())
    groups["frame_embed"] = list(m.frame_voices.parameters()) + list(m.frame_position.parameters())
    for i, layer in enumerate(m.prosody_stack.blocks):
        groups[f"prosody_stack.layer.{i}"] = list(layer.parameters())
    groups["prosody_stack.norm"] = list(m.prosody_stack.norm.parameters())
    groups["prosody_stack.all"] = list(m.prosody_stack.parameters())
    groups["prosody_head"] = list(m.prosody_head.parameters())
    groups["prosody_projection"] = list(m.prosody_projection.parameters())
    groups["duration"] = list(m.duration.parameters())
    groups["acoustic_stack"] = list(m.acoustic_stack.parameters())
    groups["vocoder"] = list(m.vocoder.parameters())
    groups["system.all"] = list(system.parameters())
    return groups


def _grad_vector(loss: torch.Tensor, params: list[torch.nn.Parameter], retain_graph: bool = True) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph, allow_unused=True)
    parts = [
        (g.detach().float().reshape(-1) if g is not None else torch.zeros(p.numel(), dtype=torch.float32))
        for g, p in zip(grads, params)
    ]
    return torch.cat(parts) if parts else torch.zeros(1, dtype=torch.float32)


def _slice_vecs(full_vec: torch.Tensor, param_slices: dict[str, tuple[int, int]],
                groups_param_ids: dict[str, list[int]]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, pids in groups_param_ids.items():
        if not pids:
            out[name] = torch.zeros(1, dtype=torch.float32)
            continue
        out[name] = torch.cat([full_vec[param_slices[pid][0]:param_slices[pid][1]] for pid in pids])
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(torch.dot(a, b).item() / (na * nb))


def _classify_cosine(val: float | None) -> str:
    if val is None:
        return "zero"
    if val > 0.05:
        return "aligned"
    if val < -0.05:
        return "conflict"
    return "orthogonal"


def _compute_batch_components(
    system: TrainingSystem,
    critics: Discriminators,
    batch: dict,
    recipe: dict,
    teacher_probability: float,
    use_gan: bool,
    prosody_grad_scale: float = 1.0,
    seed: int = 20261119,
):
    m, c = system.model, system.model.config
    weights = recipe.get("weights", WEIGHTS)
    torch.manual_seed(seed)

    hidden, token_mask = m.encode(batch["tokens"], batch["stress"], batch["token_lengths"], batch["speakers"])
    output = m.duration(hidden, token_mask).float()
    present = batch["durations"] > 0
    duration = masked_mean((output[..., 0] - torch.log1p(batch["durations"].float())).square(), token_mask & present)
    pausable = token_mask & ~m.phone_tokens[batch["tokens"]]
    pause = masked_mean(F.binary_cross_entropy_with_logits(output[..., 1], present.float(), reduction="none"), pausable)

    frames, frame_mask, lengths = m.expand(hidden, batch["durations"], token_mask, batch["speakers"])
    prosody_hidden, prediction = m.predict_prosody(frames, frame_mask)
    prediction, target = prediction.float(), batch["prosody"]
    trusted = batch["pitch_trusted"]
    pitch = masked_mean((prediction[..., 0] - target[..., 0]).abs(), frame_mask * (trusted + 0.1 * (1 - trusted)))
    sung = frame_mask * target[..., 1] * trusted
    enough = (sung.sum(1) >= 10).float()
    predicted_spread, recorded_spread = masked_std(prediction[..., 0], sung), masked_std(target[..., 0], sung)
    pitch_spread = masked_mean((predicted_spread - recorded_spread).abs(), enough)
    voicing = masked_mean(F.binary_cross_entropy_with_logits(prediction[..., 1], target[..., 1], reduction="none"), frame_mask)
    energy = masked_mean((prediction[..., 2] - target[..., 2]).abs(), frame_mask)

    teacher = torch.rand(len(lengths), device=lengths.device) < teacher_probability
    condition = torch.where(teacher[:, None, None], target, m.hard_prosody(prediction.detach()))
    if prosody_grad_scale == 1.0:
        gated_hidden = prosody_hidden
    else:
        gated_hidden = prosody_hidden * prosody_grad_scale + prosody_hidden.detach() * (1.0 - prosody_grad_scale)
    acoustic = m.decode(gated_hidden, condition.to(prosody_hidden.dtype), frame_mask)
    mel_target = (batch["mel"] - c.mel_mean) / c.mel_scale
    mel_head = masked_mean((system.mel_head(acoustic).float() - mel_target).abs().mean(-1), frame_mask)
    wave = m.vocoder(acoustic, m.frequency(target[..., 0]), lengths, phase=None)
    stft, mel = system.spectral(wave, batch["wave"], lengths * c.hop)

    loss_f0 = weights["pitch"] * pitch + weights["pitch_spread"] * pitch_spread
    loss_ve = weights["voicing"] * voicing + weights["energy"] * energy
    loss_dur = weights["duration"] * duration + weights["pause"] * pause
    loss_pros_fve = loss_f0 + loss_ve
    loss_pros_all = loss_pros_fve + loss_dur
    loss_spec = weights["mel_head"] * mel_head + weights["stft"] * stft + weights["mel"] * mel
    recon = weights["stft"] * stft + weights["mel"] * mel

    gan_scale = 0.0
    loss_adv = wave.new_zeros(())
    loss_fm = wave.new_zeros(())
    loss_gan = wave.new_zeros(())
    if use_gan:
        fake, real, index = crops(
            wave,
            batch["wave"],
            lengths * c.hop,
            int(recipe.get("crop_samples", 8192)),
            int(recipe.get("crops", 2)),
        )
        critics.requires_grad_(False)
        with torch.no_grad():
            real_features = critics(real)
        adversarial, matching = generator_gan_loss(real_features, critics(fake))
        fm_w = float(recipe.get("feature_matching", 30.0))
        gan_raw = adversarial + fm_w * matching
        recon_g = torch.autograd.grad(recon, wave, retain_graph=True)[0].gather(1, index)
        gan_g = torch.autograd.grad(gan_raw, wave, retain_graph=True)[0].gather(1, index)
        scale = (float(recipe.get("gan_ratio", 0.7)) * recon_g.norm() / gan_g.norm().clamp_min(1e-12)).clamp(
            max=float(recipe.get("gan_scale_max", 2000.0))
        ).detach()
        gan_scale = float(scale.item())
        loss_adv = scale * adversarial
        loss_fm = scale * fm_w * matching
        loss_gan = scale * gan_raw

    return {
        "f0": loss_f0,
        "ve": loss_ve,
        "dur": loss_dur,
        "pros_fve": loss_pros_fve,
        "pros_all": loss_pros_all,
        "spec": loss_spec,
        "adv": loss_adv,
        "fm": loss_fm,
        "gan": loss_gan,
        "gan_scale": gan_scale,
    }


def _build_minibatches(dataset: AnnotatedDataset, n_batches: int, batch_size: int, seed: int) -> list[dict]:
    # Deterministic balanced selection by voice, matching real training lengths
    by_spk: dict[str, list[int]] = {}
    for i, row in enumerate(dataset.rows):
        by_spk.setdefault(row["speaker"], []).append(i)
    rng = np.random.default_rng(seed)
    for spk in by_spk:
        rng.shuffle(by_spk[spk])
    half = batch_size // 2
    batches = []
    for b in range(n_batches):
        idxs = []
        for spk in sorted(by_spk):
            idxs.extend(by_spk[spk][b * half:(b + 1) * half])
        batches.append(collate([dataset[i] for i in idxs], "cpu"))
    return batches


def _analyze_checkpoint(
    ckpt_path: Path,
    train_manifest: Path,
    n_batches: int,
    batch_size: int,
    seed: int,
    teacher_probability: float,
    use_gan: bool,
    prosody_grad_scale: float = 1.0,
) -> dict:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = Config.from_dict(state["config"])
    recipe = state.get("recipe", {})
    grad_clip = float(recipe.get("grad_clip", 10.0))

    system = TrainingSystem(cfg).eval()
    critics = Discriminators().eval()
    system.load_state_dict(state["system"], strict=True)
    critics.load_state_dict(state["critics"], strict=True)

    ds = AnnotatedDataset(train_manifest, cfg, cache=False)
    batches = _build_minibatches(ds, n_batches=n_batches, batch_size=batch_size, seed=seed)

    all_params = list(system.parameters())
    param_slices: dict[int, tuple[int, int]] = {}
    offset = 0
    for p in all_params:
        n = p.numel()
        param_slices[id(p)] = (offset, offset + n)
        offset += n

    groups = _layer_groups(system)
    groups_pids = {name: [id(p) for p in plist] for name, plist in groups.items()}

    per_group_records: dict[str, list[dict]] = {name: [] for name in groups}
    batch_meta = []

    for b_idx, batch in enumerate(batches):
        comps = _compute_batch_components(
            system,
            critics,
            batch,
            recipe,
            teacher_probability=teacher_probability,
            use_gan=use_gan,
            prosody_grad_scale=prosody_grad_scale,
            seed=seed + 101 * b_idx,
        )
        g_f0 = _grad_vector(comps["f0"], all_params, retain_graph=True)
        g_ve = _grad_vector(comps["ve"], all_params, retain_graph=True)
        g_dur = _grad_vector(comps["dur"], all_params, retain_graph=True)
        g_spec = _grad_vector(comps["spec"], all_params, retain_graph=True)
        if use_gan:
            g_adv = _grad_vector(comps["adv"], all_params, retain_graph=True)
            g_fm = _grad_vector(comps["fm"], all_params, retain_graph=False)
        else:
            g_adv = torch.zeros_like(g_spec)
            g_fm = torch.zeros_like(g_spec)

        g_pros_fve = g_f0 + g_ve
        g_pros_all = g_pros_fve + g_dur
        g_gan = g_adv + g_fm
        g_acoust = g_spec + g_gan
        g_total = g_pros_all + g_acoust

        sys_norm = float(g_total.norm().item())
        clip_factor = min(1.0, grad_clip / max(sys_norm, 1e-12))
        batch_meta.append({
            "batch": b_idx,
            "utterances": int(len(batch["frame_lengths"])),
            "frames": int(batch["frame_lengths"].sum().item()),
            "gan_scale": round(comps["gan_scale"], 4),
            "system_grad_norm_pre_clip": round(sys_norm, 4),
            "clip_factor": round(clip_factor, 5),
        })

        v_f0 = _slice_vecs(g_f0, param_slices, groups_pids)
        v_pros_fve = _slice_vecs(g_pros_fve, param_slices, groups_pids)
        v_pros_all = _slice_vecs(g_pros_all, param_slices, groups_pids)
        v_spec = _slice_vecs(g_spec, param_slices, groups_pids)
        v_adv = _slice_vecs(g_adv, param_slices, groups_pids)
        v_fm = _slice_vecs(g_fm, param_slices, groups_pids)
        v_gan = _slice_vecs(g_gan, param_slices, groups_pids)
        v_acoust = _slice_vecs(g_acoust, param_slices, groups_pids)
        v_total = _slice_vecs(g_total, param_slices, groups_pids)

        for name in groups:
            pf, pa, sp, gn, ac, tot = (
                v_pros_fve[name],
                v_pros_all[name],
                v_spec[name],
                v_gan[name],
                v_acoust[name],
                v_total[name],
            )
            f0v = v_f0[name]
            n_pf = float(pf.norm().item())
            n_pa = float(pa.norm().item())
            n_f0 = float(f0v.norm().item())
            n_sp = float(sp.norm().item())
            n_gn = float(gn.norm().item())
            n_ac = float(ac.norm().item())
            n_tot = float(tot.norm().item())
            n_adv = float(v_adv[name].norm().item())
            n_fm = float(v_fm[name].norm().item())

            # Directional progress along prosody (F0/V/E) after global clipping
            proj_pf_post_clip = (
                float(torch.dot(clip_factor * tot, pf).item() / n_pf) if n_pf > 1e-12 else 0.0
            )
            proj_f0_post_clip = (
                float(torch.dot(clip_factor * tot, f0v).item() / n_f0) if n_f0 > 1e-12 else 0.0
            )
            per_group_records[name].append({
                "norm_pros_fve": n_pf,
                "norm_pros_all": n_pa,
                "norm_f0": n_f0,
                "norm_spec": n_sp,
                "norm_adv": n_adv,
                "norm_fm": n_fm,
                "norm_gan": n_gn,
                "norm_acoust": n_ac,
                "norm_total": n_tot,
                "post_clip_pros_fve": clip_factor * n_pf,
                "post_clip_spec": clip_factor * n_sp,
                "post_clip_gan": clip_factor * n_gn,
                "post_clip_total": clip_factor * n_tot,
                "post_clip_proj_pros_fve": proj_pf_post_clip,
                "post_clip_proj_f0": proj_f0_post_clip,
                "retained_pros_fve_ratio": (proj_pf_post_clip / n_pf) if n_pf > 1e-12 else None,
                "cos_pros_spec": _cosine(pf, sp),
                "cos_pros_gan": _cosine(pf, gn),
                "cos_pros_adv": _cosine(pf, v_adv[name]),
                "cos_pros_fm": _cosine(pf, v_fm[name]),
                "cos_pros_acoust": _cosine(pf, ac),
                "cos_f0_acoust": _cosine(f0v, ac),
                "cos_spec_gan": _cosine(sp, gn),
                "cos_pros_total": _cosine(pf, tot),
                "cos_pros_all_acoust": _cosine(pa, ac),
            })

    summary_groups = {}
    for name, recs in per_group_records.items():
        agg = {}
        for k in recs[0].keys():
            vals = [r[k] for r in recs if r[k] is not None]
            if not vals:
                agg[k] = None
            else:
                agg[k] = round(float(np.mean(vals)), 5)
                if k.startswith("cos_"):
                    agg[f"{k}_std"] = round(float(np.std(vals)), 5)
        agg["regime_pros_spec"] = _classify_cosine(agg["cos_pros_spec"])
        agg["regime_pros_gan"] = _classify_cosine(agg["cos_pros_gan"])
        agg["regime_pros_acoust"] = _classify_cosine(agg["cos_pros_acoust"])
        agg["regime_spec_gan"] = _classify_cosine(agg["cos_spec_gan"])
        summary_groups[name] = agg

    return {
        "checkpoint": str(ckpt_path),
        "step": int(state.get("step", -1)),
        "teacher_probability": teacher_probability,
        "use_gan": use_gan,
        "prosody_grad_scale": prosody_grad_scale,
        "grad_clip": grad_clip,
        "batches": batch_meta,
        "mean_gan_scale": round(float(np.mean([b["gan_scale"] for b in batch_meta])), 4),
        "mean_system_grad_norm_pre_clip": round(float(np.mean([b["system_grad_norm_pre_clip"] for b in batch_meta])), 4),
        "mean_clip_factor": round(float(np.mean([b["clip_factor"] for b in batch_meta])), 5),
        "groups": summary_groups,
    }


def _metrics_8k_11k_decomposition(metrics_path: Path) -> dict:
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_step = {int(r["step"]): r for r in rows}
    milestones = {}
    for s in (3000, 5000, 7000, 8000, 9000, 10000, 11000, 15000, 20000, 35000):
        if s in by_step:
            r = by_step[s]
            milestones[str(s)] = {
                "teacher_fraction": round(float(r.get("teacher_fraction", 0.0)), 4),
                "gan_scale": round(float(r.get("gan_scale", 0.0)), 3),
                "grad_norm": round(float(r.get("grad_norm", 0.0)), 2),
                "val_tf_mel": round(float(r["validation_teacher_forced_mel"]), 4) if "validation_teacher_forced_mel" in r else None,
                "val_pred_mel": round(float(r["validation_predicted_mel"]), 4) if "validation_predicted_mel" in r else None,
                "val_f0_err": round(float(r["validation_pitch_error_semitones"]), 4) if "validation_pitch_error_semitones" in r else None,
                "val_spread": round(float(r["validation_pitch_spread_ratio"]), 4) if "validation_pitch_spread_ratio" in r else None,
                "val_dur_err": round(float(r["validation_duration_length_error"]), 4) if "validation_duration_length_error" in r else None,
                "train_f0_err": round(float(r.get("pitch_error_semitones", 0.0)), 4),
                "train_energy": round(float(r.get("energy", 0.0)), 4),
            }
    return milestones


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(".local-runs/colab/fr_ref_35k/fr_ref_35k_final/last.pt"),
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path(".local-runs/colab/fr4k/annotations_e01/train.jsonl"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path(".local-runs/colab/fr_ref_35k/fr_ref_35k_final/metrics.jsonl"),
    )
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--out", type=Path, default=Path(".local-runs/p4_audit/gradient_audit.json"))
    args = parser.parse_args()

    report: dict = {"primary_35k_tp0_gan": None, "overlap_grid_35k": {}, "alpha_sweep_35k": {}, "milestones_0_35k": {}}

    # 1. Primary 35k operating point (tp=0.0, GAN ON, alpha=1.0)
    primary = _analyze_checkpoint(
        args.checkpoint, args.train_manifest, args.batches, args.batch_size, args.seed,
        teacher_probability=0.0, use_gan=True, prosody_grad_scale=1.0,
    )
    report["primary_35k_tp0_gan"] = primary

    # 2. Teacher-forcing vs GAN overlap grid (2x3) on same 4 minibatches
    for gan_on in (False, True):
        for tp in (1.0, 0.5, 0.0):
            key = f"gan_{int(gan_on)}_tp_{tp:.1f}"
            if gan_on and tp == 0.0:
                res = primary
            else:
                res = _analyze_checkpoint(
                    args.checkpoint, args.train_manifest, args.batches, args.batch_size, args.seed,
                    teacher_probability=tp, use_gan=gan_on, prosody_grad_scale=1.0,
                )
            report["overlap_grid_35k"][key] = {
                "mean_gan_scale": res["mean_gan_scale"],
                "mean_system_grad_norm_pre_clip": res["mean_system_grad_norm_pre_clip"],
                "mean_clip_factor": res["mean_clip_factor"],
                "prosody_stack.all": res["groups"]["prosody_stack.all"],
                "encoder.all": res["groups"]["encoder.all"],
                "prosody_head": res["groups"]["prosody_head"],
            }

    # 3. Alpha gradient gate sweep at 35k operating point (tp=0.0, GAN ON)
    for alpha in (1.0, 0.5, 0.25, 0.2, 0.1, 0.0):
        if alpha == 1.0:
            res = primary
        else:
            res = _analyze_checkpoint(
                args.checkpoint, args.train_manifest, args.batches, args.batch_size, args.seed,
                teacher_probability=0.0, use_gan=True, prosody_grad_scale=alpha,
            )
        report["alpha_sweep_35k"][f"alpha_{alpha:.2f}"] = {
            "mean_system_grad_norm_pre_clip": res["mean_system_grad_norm_pre_clip"],
            "mean_clip_factor": res["mean_clip_factor"],
            "prosody_stack.all": res["groups"]["prosody_stack.all"],
            "encoder.all": res["groups"]["encoder.all"],
            "prosody_head": res["groups"]["prosody_head"],
            "acoustic_stack": res["groups"]["acoustic_stack"],
        }

    # 4. Also inspect 20k high-gan_scale state if available locally
    ckpt_20k = Path(".local-runs/colab/fr_ref_20k/fr_ref_20k_final/last.pt")
    if ckpt_20k.exists():
        res20 = _analyze_checkpoint(
            ckpt_20k, args.train_manifest, args.batches, args.batch_size, args.seed,
            teacher_probability=0.0, use_gan=True, prosody_grad_scale=1.0,
        )
        report["reference_20k_tp0_gan"] = {
            "step": res20["step"],
            "mean_gan_scale": res20["mean_gan_scale"],
            "mean_system_grad_norm_pre_clip": res20["mean_system_grad_norm_pre_clip"],
            "mean_clip_factor": res20["mean_clip_factor"],
            "prosody_stack.all": res20["groups"]["prosody_stack.all"],
            "prosody_stack.layer.0": res20["groups"]["prosody_stack.layer.0"],
            "prosody_stack.layer.1": res20["groups"]["prosody_stack.layer.1"],
            "encoder.all": res20["groups"]["encoder.all"],
        }

    if args.metrics.exists():
        report["milestones_0_35k"] = _metrics_8k_11k_decomposition(args.metrics)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Compact readable summary to stdout
    print("=== PRIMARY 35k (tp=0, GAN ON, 4 minibatches x 8 utt) ===")
    print(f"mean_gan_scale={primary['mean_gan_scale']}, pre_clip_sys_norm={primary['mean_system_grad_norm_pre_clip']}, clip_factor={primary['mean_clip_factor']}")
    for gname in (
        "encoder.layer.0", "encoder.layer.1", "encoder.layer.2", "encoder.layer.3", "encoder.all",
        "frame_embed", "prosody_stack.layer.0", "prosody_stack.layer.1", "prosody_stack.norm",
        "prosody_stack.all", "prosody_head", "duration", "acoustic_stack", "vocoder",
    ):
        g = primary["groups"][gname]
        print(
            f"  {gname:22s}: ||pros||={g['norm_pros_fve']:.3f} ||spec||={g['norm_spec']:.3f} ||gan||={g['norm_gan']:.3f} "
            f"||tot||={g['norm_total']:.3f} | cos(p,sp)={g['cos_pros_spec']} cos(p,gan)={g['cos_pros_gan']} "
            f"cos(p,ac)={g['cos_pros_acoust']}({g['regime_pros_acoust']}) cos(sp,gan)={g['cos_spec_gan']} "
            f"cos(p,tot)={g['cos_pros_total']} | postClip_proj_p={g['post_clip_proj_pros_fve']:.4f} (ret={g['retained_pros_fve_ratio']})"
        )
    print("\n=== ALPHA GATE SWEEP (35k, tp=0, GAN ON) ===")
    for k, v in report["alpha_sweep_35k"].items():
        ps = v["prosody_stack.all"]
        en = v["encoder.all"]
        print(
            f"  {k}: clip={v['mean_clip_factor']:.4f} | prosody_stack: ||ac||={ps['norm_acoust']:.3f} "
            f"cos(p,tot)={ps['cos_pros_total']} ret_p={ps['retained_pros_fve_ratio']} | "
            f"encoder: ||ac||={en['norm_acoust']:.3f} cos(p,tot)={en['cos_pros_total']}"
        )
    print("\n=== OVERLAP GRID (35k, gan x tp) ===")
    for k, v in report["overlap_grid_35k"].items():
        ps = v["prosody_stack.all"]
        print(
            f"  {k}: gan_scale={v['mean_gan_scale']:.2f} clip={v['mean_clip_factor']:.4f} | "
            f"ps: ||p||={ps['norm_pros_fve']:.3f} ||sp||={ps['norm_spec']:.3f} ||gan||={ps['norm_gan']:.3f} "
            f"cos(p,ac)={ps['cos_pros_acoust']} cos(p,tot)={ps['cos_pros_total']} ret_p={ps['retained_pros_fve_ratio']}"
        )


if __name__ == "__main__":
    main()
