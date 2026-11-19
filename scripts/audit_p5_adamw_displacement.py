"""P5 Item 5 — CPU 1-step AdamW displacement & held-out ΔL_prosody probe.

Moves beyond raw gradient norms to measure what ACTUALLY moves in parameter space
and on a distinct held-out validation batch after 1 exact AdamW step (with real
checkpoint optimizer moments!) at BOTH:
  1. Pre-GAN boundary `step = 8000` (`e06_20k/step-00008000.pt`, has real `optimizer`
     state & `sampler` state right at `gan_start`!),
  2. Mature checkpoint `step = 35000` (`fr_ref_35k/last.pt`, has real `optimizer`,
     `critics`, and `critic_optimizer` states!).

Compares 4 mechanisms on identical reproducible minibatches (same batch, same RNG,
same teacher_probability):
  - `baseline_gan_on`   : exact `optimize_step` (GAN ON, alpha=1.0, global clip=10)
  - `delayed_gan_off`   : identical step with GAN OFF (`gan_start > step`)
  - `grad_gate_g02`     : GAN ON + `prosody_grad_scale = 0.2`
  - `sep_clip_prosody`  : GAN ON + separate gradient clipping on
                          (encoder + duration + prosody) vs (acoustic + vocoder + mel_head)
                          so vocoder/GAN spikes do NOT crush prosody step size!

Reports:
  - Per-module raw grad norm, post-clip grad norm, clip factor, ACTUAL AdamW parameter
    displacement `||Δθ_m||` and relative displacement `||Δθ_m|| / ||θ_m||`,
  - Held-out validation batch ΔL on `pitch`, `pitch_spread`, `voicing`, `energy`,
    `duration`, `mel`, `stft`, `pitch_error_semitones`, `pitch_spread_ratio`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tiny_voca.config import Config
from tiny_voca.data import AnnotatedDataset, BucketSampler, collate
from tiny_voca.losses import Discriminators, discriminator_loss, generator_gan_loss
from tiny_voca.train import (
    TrainingSystem,
    WEIGHTS,
    balanced_indices,
    crops,
    learning_rate,
    teacher_probability,
)

MODULE_GROUPS = (
    "encoder",
    "duration",
    "prosody_stack",
    "prosody_head",
    "acoustic_stack",
    "vocoder",
    "mel_head",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _param_groups_by_module(system: TrainingSystem) -> dict[str, list[tuple[str, nn.Parameter]]]:
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {k: [] for k in MODULE_GROUPS}
    for name, p in system.named_parameters():
        if name.startswith("mel_head."):
            groups["mel_head"].append((name, p))
        elif name.startswith("model.duration."):
            groups["duration"].append((name, p))
        elif name.startswith("model.prosody_stack."):
            groups["prosody_stack"].append((name, p))
        elif name.startswith("model.prosody_head."):
            groups["prosody_head"].append((name, p))
        elif name.startswith("model.acoustic_stack.") or name.startswith("model.prosody_projection."):
            groups["acoustic_stack"].append((name, p))
        elif name.startswith("model.vocoder."):
            groups["vocoder"].append((name, p))
        else:
            groups["encoder"].append((name, p))
    return groups


def _grad_norm(params: list[tuple[str, nn.Parameter]]) -> float:
    sq = 0.0
    for _, p in params:
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum().item())
    return sq ** 0.5


def _param_norm(params: list[tuple[str, nn.Parameter]]) -> float:
    sq = 0.0
    for _, p in params:
        sq += float(p.detach().float().pow(2).sum().item())
    return sq ** 0.5


@torch.no_grad()
def _eval_batch(system: TrainingSystem, batch: dict, weights: dict, val_seed: int = 999001) -> dict[str, float]:
    """Evaluate on validation batch with FIXED vocoder noise seed across before/after and all variants."""
    rng_state = torch.get_rng_state()
    torch.manual_seed(val_seed)
    system.eval()
    try:
        _, terms, _ = system.objective(batch, teacher_probability=0.0, weights=weights, random_phase=False)
    finally:
        system.train()
        torch.set_rng_state(rng_state)
    keys = (
        "pitch", "pitch_spread", "voicing", "energy", "duration", "pause",
        "mel", "stft", "pitch_error_semitones", "pitch_spread_ratio", "voicing_accuracy",
    )
    return {k: float(terms[k].detach().item()) for k in keys}


def _run_one_step_variant(
    state: dict,
    train_batch: dict,
    val_batch: dict,
    variant: str,
    step_seed: int = 20261119,
) -> dict:
    config = Config.from_dict(state["config"])
    recipe = copy.deepcopy(state["recipe"])
    step = int(state["step"])

    system = TrainingSystem(config)
    critics = Discriminators()
    system.load_state_dict(state["system"])
    critics.load_state_dict(state["critics"])

    optimizer = torch.optim.AdamW(system.parameters(), lr=recipe["learning_rate"], betas=(0.8, 0.99), weight_decay=0.01)
    critic_optimizer = torch.optim.AdamW(critics.parameters(), lr=recipe["learning_rate"], betas=(0.8, 0.99), weight_decay=0.0)
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    critic_optimizer.load_state_dict(copy.deepcopy(state["critic_optimizer"]))

    # Variant overrides.
    use_gan = True
    prosody_grad_scale = 1.0
    separate_clip = False
    if variant == "delayed_gan_off":
        use_gan = False
    elif variant == "grad_gate_g02":
        prosody_grad_scale = 0.2
    elif variant == "grad_gate_g00":
        prosody_grad_scale = 0.0
    elif variant == "sep_clip_prosody":
        separate_clip = True
    elif variant == "detach_plus_sep_clip":
        prosody_grad_scale = 0.0
        separate_clip = True
    elif variant != "baseline_gan_on":
        raise ValueError(f"unknown variant {variant}")

    weights = recipe["weights"]
    val_before = _eval_batch(system, val_batch, weights)

    mod_groups = _param_groups_by_module(system)
    theta_before = {name: p.detach().clone() for name, p in system.named_parameters()}
    theta_norms = {m: _param_norm(ps) for m, ps in mod_groups.items()}

    # Exact LR & teacher probability from train.py.
    start = (recipe.get("branched_from") or {}).get("step", 0)
    lr = learning_rate(step, recipe["learning_rate"], recipe["warmup"], recipe["lr_half_life"], start)
    for g in optimizer.param_groups:
        g["lr"] = lr
    for g in critic_optimizer.param_groups:
        g["lr"] = learning_rate(step, recipe["learning_rate"], 1, recipe["lr_half_life"], start)

    torch.manual_seed(step_seed)
    optimizer.zero_grad(set_to_none=True)
    prob = teacher_probability(step, recipe["teacher_hold"], recipe["teacher_ramp"], recipe["teacher_final"])
    loss, metrics, auxiliary = system.objective(
        train_batch, prob, weights, random_phase=True, prosody_grad_scale=prosody_grad_scale
    )

    gan_scale_val = 0.0
    if use_gan:
        wave = auxiliary["wave"]
        fake, real, index = crops(
            wave,
            train_batch["wave"],
            train_batch["frame_lengths"] * config.hop,
            recipe["crop_samples"],
            recipe["crops"],
        )
        critic_optimizer.zero_grad(set_to_none=True)
        d_loss = discriminator_loss(critics(real), critics(fake.detach()))
        d_loss.backward()
        nn.utils.clip_grad_norm_(critics.parameters(), 10, error_if_nonfinite=True)
        critic_optimizer.step()
        critics.requires_grad_(False)
        try:
            with torch.no_grad():
                real_features = critics(real)
            adversarial, matching = generator_gan_loss(real_features, critics(fake))
            gan = adversarial + recipe["feature_matching"] * matching
            recon_grad = torch.autograd.grad(auxiliary["reconstruction"], wave, retain_graph=True)[0].gather(1, index)
            gan_grad = torch.autograd.grad(gan, wave, retain_graph=True)[0].gather(1, index)
            # At step==gan_start, ramp = 1 / gan_ramp; also simulate steady-state ramp=1 if step > gan_start.
            ramp = min(1.0, (step - recipe["gan_start"] + 1) / max(recipe["gan_ramp"], 1))
            scale = (recipe["gan_ratio"] * ramp * recon_grad.norm() / gan_grad.norm().clamp_min(1e-12)).clamp(max=recipe["gan_scale_max"]).detach()
            gan_scale_val = float(scale.item())
            loss = loss + scale * gan
            loss.backward()
        finally:
            critics.requires_grad_(True)
    else:
        loss.backward()

    raw_norms = {m: _grad_norm(ps) for m, ps in mod_groups.items()}
    total_raw_grad = float(sum(v * v for v in raw_norms.values()) ** 0.5)

    clip_val = float(recipe["grad_clip"])
    if separate_clip:
        prosody_params = [p for m in ("encoder", "duration", "prosody_stack", "prosody_head") for _, p in mod_groups[m]]
        acoustic_params = [p for m in ("acoustic_stack", "vocoder", "mel_head") for _, p in mod_groups[m]]
        g_pros = float(nn.utils.clip_grad_norm_(prosody_params, clip_val, error_if_nonfinite=True))
        g_acou = float(nn.utils.clip_grad_norm_(acoustic_params, clip_val, error_if_nonfinite=True))
        clip_info = {
            "prosody_clip_factor": round(min(1.0, clip_val / max(g_pros, 1e-12)), 5),
            "acoustic_clip_factor": round(min(1.0, clip_val / max(g_acou, 1e-12)), 5),
        }
    else:
        g_tot = float(nn.utils.clip_grad_norm_(system.parameters(), clip_val, error_if_nonfinite=True))
        clip_info = {
            "global_clip_factor": round(min(1.0, clip_val / max(g_tot, 1e-12)), 5),
        }

    post_clip_norms = {m: _grad_norm(ps) for m, ps in mod_groups.items()}
    optimizer.step()

    # Measure ACTUAL AdamW parameter displacement per module!
    disp_abs = {}
    disp_rel = {}
    for m, ps in mod_groups.items():
        sq = 0.0
        for name, p in ps:
            diff = (p.detach() - theta_before[name]).float()
            sq += float(diff.pow(2).sum().item())
        d = sq ** 0.5
        disp_abs[m] = round(d, 7)
        disp_rel[m] = round(d / max(theta_norms[m], 1e-12), 7)

    val_after = _eval_batch(system, val_batch, weights)
    val_delta = {k: round(val_after[k] - val_before[k], 6) for k in val_before}

    return {
        "variant": variant,
        "step": step,
        "lr": round(lr, 8),
        "teacher_probability": round(prob, 5),
        "gan_scale": round(gan_scale_val, 4),
        "total_raw_grad_norm": round(total_raw_grad, 4),
        "clip_info": clip_info,
        "raw_grad_norms": {m: round(v, 4) for m, v in raw_norms.items()},
        "post_clip_grad_norms": {m: round(v, 4) for m, v in post_clip_norms.items()},
        "adamw_disp_l2": disp_abs,
        "adamw_disp_rel": disp_rel,
        "val_delta": val_delta,
    }


def _probe_checkpoint(ckpt_path: Path, train_manifest: Path, val_manifest: Path, batch_size: int = 8) -> dict:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    config = Config.from_dict(state["config"])
    train_ds = AnnotatedDataset(train_manifest, config, voiced_vowels=state["recipe"].get("voiced_vowel_targets", True))
    val_ds = AnnotatedDataset(val_manifest, config, voiced_vowels=state["recipe"].get("voiced_vowel_targets", True))

    # Reproducible balanced 8-utterance train batch & 16-utterance validation batch.
    t_idx = balanced_indices(train_ds, batch_size)
    v_idx = balanced_indices(val_ds, 16)
    train_batch = collate([train_ds[i] for i in t_idx], "cpu")
    val_batch = collate([val_ds[i] for i in v_idx], "cpu")

    # Also test what happens at step = gan_start + 1500 (mid-ramp, 50% GAN) on the 8k checkpoint
    # to capture BOTH exact step=8000 (1/3000 ramp) AND steady-state GAN ramp!
    variants = ("baseline_gan_on", "delayed_gan_off", "grad_gate_g02", "grad_gate_g00", "sep_clip_prosody", "detach_plus_sep_clip")
    results = {}
    for v in variants:
        results[v] = _run_one_step_variant(state, train_batch, val_batch, v)

    # If step == gan_start (first GAN step has ramp=1/3000), ALSO probe with ramp=1.0 (full GAN strength on the 8k weights/optimizer)!
    full_gan_results = None
    if int(state["step"]) == int(state["recipe"]["gan_start"]):
        state_full = copy.deepcopy(state)
        state_full["recipe"]["gan_ramp"] = 1
        full_gan_results = {
            v: _run_one_step_variant(state_full, train_batch, val_batch, v)
            for v in ("baseline_gan_on", "grad_gate_g02", "grad_gate_g00", "sep_clip_prosody", "detach_plus_sep_clip")
        }

    return {
        "checkpoint": str(ckpt_path),
        "sha256": _sha256(ckpt_path),
        "step": int(state["step"]),
        "gan_start": int(state["recipe"]["gan_start"]),
        "gan_ramp": int(state["recipe"]["gan_ramp"]),
        "has_optimizer": "optimizer" in state,
        "has_sampler": "sampler" in state,
        "has_torch_rng": "torch_rng" in state,
        "has_cuda_rng": state.get("cuda_rng") is not None,
        "train_batch_ids": [train_ds.rows[i]["id"] for i in t_idx],
        "val_batch_ids": [val_ds.rows[i]["id"] for i in v_idx],
        "results": results,
        "full_gan_ramp_results": full_gan_results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path(".local-runs/colab/fr4k/annotations_e01/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path(".local-runs/colab/fr4k/annotations_e01/validation.jsonl"))
    parser.add_argument("--ckpt-8k", type=Path, default=Path(".local-runs/colab/e06_20k/e06_20k_final/step-00008000.pt"))
    parser.add_argument("--ckpt-35k", type=Path, default=Path(".local-runs/colab/fr_ref_35k/fr_ref_35k_final/last.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path(".local-runs/p5_audit/adamw_displacement"))
    args = parser.parse_args()

    out = {
        "e06_8k": _probe_checkpoint(args.ckpt_8k, args.train, args.validation),
        "fr_ref_35k": _probe_checkpoint(args.ckpt_35k, args.train, args.validation),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "p5_adamw_displacement.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    for tag in ("e06_8k", "fr_ref_35k"):
        block = out[tag]
        print(f"\n=== {tag} (step={block['step']}, sha={block['sha256'][:12]}) ===")
        for label, group in (("exact_step", block["results"]), ("full_gan_ramp", block["full_gan_ramp_results"])):
            if not group:
                continue
            print(f"  [{label}]")
            for v, r in group.items():
                d = r["adamw_disp_rel"]
                vd = r["val_delta"]
                print(
                    f"    {v:18s}: rawG={r['total_raw_grad_norm']:7.2f} clip={r['clip_info']} | "
                    f"relΔθ(enc={d['encoder']:.2e}, pros={d['prosody_stack']:.2e}, pHead={d['prosody_head']:.2e}, voc={d['vocoder']:.2e}) | "
                    f"valΔ(f0_st={vd['pitch_error_semitones']:+.4f}, spreadR={vd['pitch_spread_ratio']:+.4f}, mel={vd['mel']:+.5f})"
                )


if __name__ == "__main__":
    main()
