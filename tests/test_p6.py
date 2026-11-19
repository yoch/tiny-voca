"""Unit tests for P6 paired replay mechanisms in tiny_voca.train:
- split_prosody_acoustic_parameters / clip_system_gradients
- --restore-state-from-init bit-exact equivalence with --resume
- --align-gan-rng exact RNG alignment between GAN-on (C/P) and delayed-GAN (D) arms
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from tiny_voca.config import ANNOTATION_VERSION, Config
from tiny_voca.losses import Discriminators
from tiny_voca.text import SYMBOL_TO_ID, frontend_fingerprint, frontend_rules_fingerprint
from tiny_voca.train import (
    TrainingSystem,
    WEIGHTS,
    clip_system_gradients,
    optimize_step,
    split_prosody_acoustic_parameters,
)


def _tiny_config() -> Config:
    return Config(
        channels=32,
        heads=4,
        feedforward=64,
        encoder_layers=2,
        prosody_layers=2,
        acoustic_layers=2,
        predictor_channels=32,
        speakers=("F1", "M1"),
        max_frames=400,
    )


def _synthetic_batch(cfg: Config, batch_size: int = 2, n_tokens: int = 8, frames_per_tok: int = 6) -> dict:
    g = torch.Generator().manual_seed(123)
    t_frames = n_tokens * frames_per_tok
    tokens = torch.full((batch_size, n_tokens), SYMBOL_TO_ID["a"], dtype=torch.long)
    tokens[:, 0] = SYMBOL_TO_ID["<bos>"]
    tokens[:, -1] = SYMBOL_TO_ID["<eos>"]
    stress = torch.zeros_like(tokens)
    durations = torch.full((batch_size, n_tokens), frames_per_tok, dtype=torch.long)
    t = torch.linspace(0, 1, t_frames)
    f0 = (0.25 * torch.sin(2.0 * 3.14159 * t))[None].expand(batch_size, -1).clone()
    voiced = torch.ones((batch_size, t_frames), dtype=torch.float32)
    energy = torch.full((batch_size, t_frames), 0.8, dtype=torch.float32)
    prosody = torch.stack((f0, voiced, energy), dim=-1)
    return {
        "tokens": tokens,
        "stress": stress,
        "durations": durations,
        "token_lengths": torch.full((batch_size,), n_tokens, dtype=torch.long),
        "frame_lengths": torch.full((batch_size,), t_frames, dtype=torch.long),
        "speakers": torch.arange(batch_size, dtype=torch.long) % len(cfg.speakers),
        "prosody": prosody,
        "pitch_trusted": voiced.clone(),
        "mel": torch.randn(batch_size, t_frames, cfg.n_mels, generator=g) * 0.5 - 5.0,
        "wave": torch.randn(batch_size, t_frames * cfg.hop, generator=g) * 0.05,
    }


class P6ReplayTests(unittest.TestCase):
    def test_split_prosody_acoustic_parameters_partitions_system_exactly(self):
        system = TrainingSystem(_tiny_config())
        pros, acou = split_prosody_acoustic_parameters(system)
        all_ids = {id(p) for p in system.parameters()}
        pros_ids = {id(p) for p in pros}
        acou_ids = {id(p) for p in acou}
        self.assertEqual(len(pros_ids & acou_ids), 0)
        self.assertEqual(pros_ids | acou_ids, all_ids)
        self.assertGreater(len(pros), 0)
        self.assertGreater(len(acou), 0)

    def test_separate_clip_protects_prosody_from_vocoder_gradient_spike(self):
        system = TrainingSystem(_tiny_config())
        pros, acou = split_prosody_acoustic_parameters(system)
        for p in pros:
            p.grad = torch.full_like(p, 0.01)
        for p in acou:
            p.grad = torch.full_like(p, 100.0)
        pros_norm_before = sum(float(p.grad.pow(2).sum()) for p in pros) ** 0.5
        self.assertLess(pros_norm_before, 10.0)
        _, info = clip_system_gradients(system, max_norm=10.0, separate_prosody=True)
        pros_norm_after = sum(float(p.grad.pow(2).sum()) for p in pros) ** 0.5
        acou_norm_after = sum(float(p.grad.pow(2).sum()) for p in acou) ** 0.5
        self.assertAlmostEqual(pros_norm_after, pros_norm_before, places=5)
        self.assertAlmostEqual(acou_norm_after, 10.0, places=4)
        self.assertIn("grad_norm_prosody", info)
        self.assertIn("grad_norm_acoustic", info)

    def test_align_gan_rng_matches_gan_arm_rng_state_bit_for_bit(self):
        cfg = _tiny_config()
        batch = _synthetic_batch(cfg)
        base_recipe = {
            "learning_rate": 3e-4,
            "warmup": 10,
            "lr_half_life": 100000.0,
            "grad_clip": 10.0,
            "gan_start": 5,
            "gan_ratio": 0.7,
            "gan_ramp": 10,
            "feature_matching": 30.0,
            "gan_scale_max": 2000.0,
            "crop_samples": 1024,
            "crops": 2,
            "teacher_hold": 0,
            "teacher_ramp": 10,
            "teacher_final": 0.0,
            "weights": dict(WEIGHTS),
            "branched_from": {"step": 5, "keep_schedule": True},
        }
        torch.manual_seed(77)
        s0, c0 = TrainingSystem(cfg), Discriminators()
        rng_start = torch.get_rng_state().clone()

        def run_arm(overrides: dict):
            sys_m = copy.deepcopy(s0).train()
            crit_m = copy.deepcopy(c0).train()
            opt = torch.optim.AdamW(sys_m.parameters(), lr=3e-4, betas=(0.8, 0.99), weight_decay=0.01)
            copt = torch.optim.AdamW(crit_m.parameters(), lr=3e-4, betas=(0.8, 0.99), weight_decay=0.0)
            rec = {**base_recipe, **overrides}
            torch.set_rng_state(rng_start.clone())
            m1 = optimize_step(sys_m, crit_m, opt, copt, batch, 5, rec)
            r1 = torch.get_rng_state().clone()
            m2 = optimize_step(sys_m, crit_m, opt, copt, batch, 6, rec)
            r2 = torch.get_rng_state().clone()
            return m1, m2, r1, r2

        mc1, mc2, rc1, rc2 = run_arm({})
        md1, md2, rd1, rd2 = run_arm({"gan_start": 999999, "align_gan_rng": True})
        mp1, mp2, rp1, rp2 = run_arm({"prosody_grad_scale": 0.0, "separate_prosody_clip": True})

        self.assertTrue(torch.equal(rc1, rd1))
        self.assertTrue(torch.equal(rc2, rd2))
        self.assertTrue(torch.equal(rc1, rp1))
        self.assertTrue(torch.equal(rc2, rp2))
        self.assertEqual(mc1["teacher_fraction"], md1["teacher_fraction"])
        self.assertEqual(mc2["teacher_fraction"], md2["teacher_fraction"])
        self.assertEqual(mc1["teacher_fraction"], mp1["teacher_fraction"])
        self.assertEqual(mc2["teacher_fraction"], mp2["teacher_fraction"])

    def test_cli_restore_state_from_init_matches_resume_bit_exact(self):
        cfg = _tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feat_dir = root / "features"
            ann_dir = root / "ann"
            feat_dir.mkdir()
            ann_dir.mkdir()
            cfg_path = root / "cfg.json"
            cfg_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")

            ann_fp = "test_ann_p6"
            meta = {
                "complete": True,
                "fingerprint": ann_fp,
                "acoustic_root": str(feat_dir),
                "frontend_rules": frontend_rules_fingerprint(),
                "annotation_version": ANNOTATION_VERSION,
            }
            (ann_dir / "annotations.json").write_text(json.dumps(meta), encoding="utf-8")

            def make_split(name: str, count: int, seed_base: int):
                rows = []
                for i in range(count):
                    g = torch.Generator().manual_seed(seed_base + i)
                    t_frames = 36
                    tokens = [SYMBOL_TO_ID["<bos>"], SYMBOL_TO_ID["a"], SYMBOL_TO_ID["t"], SYMBOL_TO_ID["i"], SYMBOL_TO_ID["<eos>"]]
                    durs = [4, 10, 8, 10, 4]
                    spk = "F1" if i % 2 == 0 else "M1"
                    uid = f"{name}_{i}_{spk}"
                    rel = f"{uid}.pt"
                    torch.save({
                        "mel": torch.randn(t_frames, cfg.n_mels, generator=g) * 0.5 - 5.0,
                        "prosody": torch.stack((
                            torch.linspace(-0.1, 0.2, t_frames),
                            torch.ones(t_frames),
                            torch.full((t_frames,), -2.0),
                        ), dim=-1),
                        "wave": torch.randn(t_frames * cfg.hop, generator=g) * 0.05,
                    }, feat_dir / rel)
                    rows.append({
                        "id": uid,
                        "text_hash": f"h_{name}_{i // 2}",
                        "text": f"phrase {name} {i}",
                        "speaker": spk,
                        "path": rel,
                        "frames": t_frames,
                        "tokens": tokens,
                        "stress": [0] * len(tokens),
                        "durations": durs,
                        "frontend": frontend_fingerprint(),
                        "frontend_rules": frontend_rules_fingerprint(),
                        "acoustic": cfg.acoustic_fingerprint,
                        "annotation": ann_fp,
                    })
                (ann_dir / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

            make_split("train", 4, 100)
            make_split("validation", 2, 200)

            common = [
                sys.executable, "-m", "tiny_voca.train",
                "--train", str(ann_dir / "train.jsonl"),
                "--validation", str(ann_dir / "validation.jsonl"),
                "--config", str(cfg_path),
                "--device", "cpu",
                "--batch-frames", "200",
                "--max-batch", "2",
                "--crop-samples", "1024",
                "--crops", "2",
                "--gan-start", "2",
                "--gan-ramp", "4",
                "--teacher-hold", "1",
                "--teacher-ramp", "4",
                "--warmup", "2",
                "--save-every", "2",
                "--log-every", "1",
                "--seed", "42",
                "--no-preload",
            ]
            # Base run 0 -> 2 (saves step-00000002.pt right at gan_start == 2!)
            base_out = root / "base"
            subprocess.run(common + ["--output", str(base_out), "--steps", "2"], check=True, capture_output=True)
            ckpt2 = base_out / "step-00000002.pt"
            self.assertTrue(ckpt2.exists())

            # Copy base dir for --resume 2 -> 4
            resume_out = root / "resume_run"
            resume_out.mkdir()
            subprocess.run(common + ["--output", str(resume_out), "--resume", str(ckpt2), "--steps", "4"], check=True, capture_output=True)

            # Fresh dir for --init-from --restore-state-from-init 2 -> 4
            init_out = root / "init_run"
            subprocess.run(common + ["--output", str(init_out), "--init-from", str(ckpt2), "--restore-state-from-init", "--steps", "4"], check=True, capture_output=True)

            st_res = torch.load(resume_out / "last.pt", map_location="cpu", weights_only=True)
            st_ini = torch.load(init_out / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(st_res["step"], 4)
            self.assertEqual(st_ini["step"], 4)
            for k in st_res["system"]:
                self.assertTrue(torch.equal(st_res["system"][k], st_ini["system"][k]), f"mismatch in system.{k}")
            for k in st_res["critics"]:
                self.assertTrue(torch.equal(st_res["critics"][k], st_ini["critics"][k]), f"mismatch in critics.{k}")
            self.assertTrue(torch.equal(st_res["torch_rng"], st_ini["torch_rng"]))
            self.assertEqual(st_res["sampler"].keys(), st_ini["sampler"].keys())
            for k in st_res["sampler"]:
                va, vb = st_res["sampler"][k], st_ini["sampler"][k]
                if isinstance(va, torch.Tensor):
                    self.assertTrue(torch.equal(va, vb), f"sampler.{k} mismatch")
                else:
                    self.assertEqual(va, vb, f"sampler.{k} mismatch")


if __name__ == "__main__":
    unittest.main()
