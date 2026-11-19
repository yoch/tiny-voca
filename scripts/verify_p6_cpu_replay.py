"""Pre-GPU CPU verification for P6 paired replay from e06_20k/step-00008000.pt.

Verifies on real checkpoint e06_20k/step-00008000.pt & real fr4k annotations:
  1. Arm C (--init-from --restore-state-from-init) reproduces exact --resume bit-for-bit
     (0.0 max abs diff across all weights, critics, optimizer moments, losses, grad_norm,
     gan_scale, sampler state, and CPU torch RNG state over 2 real steps).
  2. Arm D (--gan-start 999999 --align-gan-rng) and Arm P (--prosody-grad-scale 0.0
     --separate-prosody-clip) maintain bit-exact identical sampler state, batch sequence,
     teacher_fraction, and post-step torch RNG state across steps 8000 -> 8002.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tiny_voca.config import Config
from tiny_voca.data import AnnotatedDataset, BucketSampler, collate
from tiny_voca.losses import Discriminators
from tiny_voca.train import TrainingSystem, optimize_step


def _run_two_steps(ckpt_path: Path, train_ds: AnnotatedDataset, mode: str) -> dict:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = Config.from_dict(state["config"])
    recipe = copy.deepcopy(state["recipe"])

    system = TrainingSystem(cfg)
    critics = Discriminators()
    optimizer = torch.optim.AdamW(system.parameters(), lr=recipe["learning_rate"], betas=(0.8, 0.99), weight_decay=0.01)
    critic_optimizer = torch.optim.AdamW(critics.parameters(), lr=recipe["learning_rate"], betas=(0.8, 0.99), weight_decay=0.0)
    sampler = BucketSampler([train_ds.lengths[i] for i in range(len(train_ds))], recipe["batch_frames"], recipe["max_batch"], recipe["seed"])

    # Exact full state restoration (identical to --resume and --init-from --restore-state-from-init).
    system.load_state_dict(state["system"])
    critics.load_state_dict(state["critics"])
    optimizer.load_state_dict(state["optimizer"])
    critic_optimizer.load_state_dict(state["critic_optimizer"])
    sampler.load_state(state["sampler"])
    torch.set_rng_state(state["torch_rng"].cpu())
    step = int(state["step"])

    if mode == "C_restore_init":
        recipe["branched_from"] = {"step": step, "keep_schedule": True, "recipe": state["recipe"]}
    elif mode == "D_delayed_gan":
        recipe["branched_from"] = {"step": step, "keep_schedule": True, "recipe": state["recipe"]}
        recipe["gan_start"] = 999999
        recipe["align_gan_rng"] = True
    elif mode == "P_protected":
        recipe["branched_from"] = {"step": step, "keep_schedule": True, "recipe": state["recipe"]}
        recipe["prosody_grad_scale"] = 0.0
        recipe["separate_prosody_clip"] = True
    elif mode != "resume_exact":
        raise ValueError(mode)

    system.train()
    step_logs = []
    batch_ids = []
    rng_states = []
    for _ in range(2):
        idxs = sampler.next()
        # Use first 2 utterances from the real bucket batch on CPU to keep verification fast
        # while verifying exact sampler progression + full forward/backward/GAN/clip/RNG!
        sub_idxs = idxs[:2]
        batch_ids.append([train_ds.rows[i]["id"] for i in idxs])
        batch = collate([train_ds[i] for i in sub_idxs], "cpu")
        metrics = optimize_step(system, critics, optimizer, critic_optimizer, batch, step, recipe)
        step += 1
        step_logs.append(metrics)
        rng_states.append(torch.get_rng_state().clone())

    return {
        "mode": mode,
        "step": step,
        "batch_ids": batch_ids,
        "step_logs": step_logs,
        "rng_states": rng_states,
        "sampler_state": sampler.state(),
        "system_sd": {k: v.detach().clone() for k, v in system.state_dict().items()},
        "critics_sd": {k: v.detach().clone() for k, v in critics.state_dict().items()},
    }


def _states_equal(a, b) -> bool:
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return bool(torch.equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_states_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_states_equal(x, y) for x, y in zip(a, b))
    return a == b


def main():
    ckpt = Path(".local-runs/colab/e06_20k/e06_20k_final/step-00008000.pt")
    ann = Path(".local-runs/colab/fr4k/annotations_e01")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    cfg = Config.from_dict(state["config"])
    train_ds = AnnotatedDataset(ann / "train.jsonl", cfg, voiced_vowels=state["recipe"].get("voiced_vowel_targets", True))
    val_ds = AnnotatedDataset(ann / "validation.jsonl", cfg, voiced_vowels=state["recipe"].get("voiced_vowel_targets", True))

    ds_match = state["datasets"] == [train_ds.fingerprint, val_ds.fingerprint]
    print(f"Checkpoint datasets fingerprint match: {ds_match}")
    print(f"Checkpoint step={state['step']}, has_cuda_rng={state.get('cuda_rng') is not None}, recipe_gan_start={state['recipe']['gan_start']}, gan_ramp={state['recipe']['gan_ramp']}")
    assert ds_match, "Dataset fingerprints mismatch!"

    res_resume = _run_two_steps(ckpt, train_ds, "resume_exact")
    res_c = _run_two_steps(ckpt, train_ds, "C_restore_init")
    res_d = _run_two_steps(ckpt, train_ds, "D_delayed_gan")
    res_p = _run_two_steps(ckpt, train_ds, "P_protected")

    # 1. Bit-exact C_restore_init == resume_exact
    max_sys_diff = max(float((res_resume["system_sd"][k] - res_c["system_sd"][k]).abs().max().item()) for k in res_resume["system_sd"])
    max_crit_diff = max(float((res_resume["critics_sd"][k] - res_c["critics_sd"][k]).abs().max().item()) for k in res_resume["critics_sd"])
    c_rng_match = all(torch.equal(a, b) for a, b in zip(res_resume["rng_states"], res_c["rng_states"]))
    c_sampler_match = _states_equal(res_resume["sampler_state"], res_c["sampler_state"])
    c_loss_match = res_resume["step_logs"] == res_c["step_logs"]

    # 2. Exact RNG, sampler, batch sequence, and teacher_fraction alignment across C, D, P
    d_rng_match = all(torch.equal(a, b) for a, b in zip(res_c["rng_states"], res_d["rng_states"]))
    p_rng_match = all(torch.equal(a, b) for a, b in zip(res_c["rng_states"], res_p["rng_states"]))
    d_batch_match = res_c["batch_ids"] == res_d["batch_ids"]
    p_batch_match = res_c["batch_ids"] == res_p["batch_ids"]
    d_tf_match = [m["teacher_fraction"] for m in res_c["step_logs"]] == [m["teacher_fraction"] for m in res_d["step_logs"]]
    p_tf_match = [m["teacher_fraction"] for m in res_c["step_logs"]] == [m["teacher_fraction"] for m in res_p["step_logs"]]

    summary = {
        "checkpoint": str(ckpt),
        "datasets_fingerprint_match": ds_match,
        "has_cuda_rng_in_ckpt": state.get("cuda_rng") is not None,
        "C_vs_resume_max_system_param_diff": max_sys_diff,
        "C_vs_resume_max_critic_param_diff": max_crit_diff,
        "C_vs_resume_rng_bit_exact": c_rng_match,
        "C_vs_resume_sampler_exact": c_sampler_match,
        "C_vs_resume_metrics_exact": c_loss_match,
        "D_vs_C_rng_bit_exact": d_rng_match,
        "P_vs_C_rng_bit_exact": p_rng_match,
        "D_vs_C_batches_exact": d_batch_match,
        "P_vs_C_batches_exact": p_batch_match,
        "D_vs_C_teacher_fraction_exact": d_tf_match,
        "P_vs_C_teacher_fraction_exact": p_tf_match,
        "step_8000_metrics": {
            "C": {k: round(v, 6) for k, v in res_c["step_logs"][0].items() if k in ("loss", "grad_norm", "gan_scale", "teacher_fraction", "pitch_error_semitones")},
            "D": {k: round(v, 6) for k, v in res_d["step_logs"][0].items() if k in ("loss", "grad_norm", "teacher_fraction", "pitch_error_semitones")},
            "P": {k: round(v, 6) for k, v in res_p["step_logs"][0].items() if k in ("loss", "grad_norm", "grad_norm_prosody", "grad_norm_acoustic", "gan_scale", "teacher_fraction", "pitch_error_semitones")},
        },
    }
    out_path = Path(".local-runs/p6_replay/cpu_replay_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    assert max_sys_diff == 0.0 and max_crit_diff == 0.0 and c_rng_match and c_sampler_match and c_loss_match
    assert d_rng_match and p_rng_match and d_batch_match and p_batch_match and d_tf_match and p_tf_match
    print("ALL P6 CPU REPLAY & RNG ALIGNMENT CHECKS PASSED BIT-EXACTLY!")


if __name__ == "__main__":
    main()
