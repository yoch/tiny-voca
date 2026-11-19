"""Unit tests for P4 causal interventions: gradient gate (G), local F0 shape loss (F),
and reproducible 2x2 --init-from branching."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch

from tiny_voca.config import ANNOTATION_VERSION, Config
from tiny_voca.losses import pitch_shape_loss, scale_gradient
from tiny_voca.text import SYMBOL_TO_ID, frontend_fingerprint, frontend_rules_fingerprint
from tiny_voca.train import TrainingSystem, WEIGHTS


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
    torch.manual_seed(123)
    t_frames = n_tokens * frames_per_tok
    tokens = torch.full((batch_size, n_tokens), SYMBOL_TO_ID["a"], dtype=torch.long)
    tokens[:, 0] = SYMBOL_TO_ID["<bos>"]
    tokens[:, -1] = SYMBOL_TO_ID["<eos>"]
    stress = torch.zeros_like(tokens)
    durations = torch.full((batch_size, n_tokens), frames_per_tok, dtype=torch.long)
    t = torch.linspace(0, 1, t_frames)
    f0 = (0.25 * torch.sin(2.0 * 3.14159 * t))[None].expand(batch_size, -1).clone()
    voiced = torch.ones((batch_size, t_frames), dtype=torch.float32)
    voiced[:, :4] = 0.0
    voiced[:, -4:] = 0.0
    energy = torch.full((batch_size, t_frames), 0.8, dtype=torch.float32)
    prosody = torch.stack((f0, voiced, energy), dim=-1)
    pitch_trusted = voiced.clone()
    return {
        "tokens": tokens,
        "stress": stress,
        "durations": durations,
        "token_lengths": torch.full((batch_size,), n_tokens, dtype=torch.long),
        "frame_lengths": torch.full((batch_size,), t_frames, dtype=torch.long),
        "speakers": torch.arange(batch_size, dtype=torch.long) % len(cfg.speakers),
        "prosody": prosody,
        "pitch_trusted": pitch_trusted,
        "mel": torch.randn(batch_size, t_frames, cfg.n_mels) * 0.5 - 5.0,
        "wave": torch.randn(batch_size, t_frames * cfg.hop) * 0.05,
    }


class P4InterventionTests(unittest.TestCase):
    def test_scale_gradient_preserves_forward_and_scales_backward(self):
        x = torch.randn(2, 10, 8, requires_grad=True)
        for alpha in (0.0, 0.2, 0.5, 1.0):
            x_in = x.detach().clone().requires_grad_(True)
            y = scale_gradient(x_in, alpha)
            self.assertTrue(torch.equal(y.detach(), x_in.detach()))
            (y.square().sum()).backward()
            expected = 2.0 * alpha * x_in.detach()
            self.assertTrue(torch.allclose(x_in.grad, expected, atol=1e-6))

    def test_prosody_grad_gate_scales_only_acoustic_path_into_prosody_stack(self):
        cfg = _tiny_config()
        torch.manual_seed(7)
        system = TrainingSystem(cfg).eval()
        # Activate vocoder weights so wave->acoustic path is non-zero
        with torch.no_grad():
            system.model.vocoder.harmonic.weight.normal_(0.0, 0.02)
            system.model.vocoder.noise.weight.normal_(0.0, 0.02)
        batch = _synthetic_batch(cfg)

        def grads_for(alpha: float, spec_only: bool):
            system.zero_grad(set_to_none=True)
            torch.manual_seed(99)
            _, terms, _ = system.objective(batch, teacher_probability=0.0, weights=WEIGHTS, random_phase=False, prosody_grad_scale=alpha)
            if spec_only:
                loss = WEIGHTS["mel"] * terms["mel"] + WEIGHTS["stft"] * terms["stft"] + WEIGHTS["mel_head"] * terms["mel_head"]
            else:
                loss = WEIGHTS["pitch"] * terms["pitch"] + WEIGHTS["voicing"] * terms["voicing"] + WEIGHTS["energy"] * terms["energy"]
            loss.backward()
            return {
                "prosody_stack": torch.cat([p.grad.detach().reshape(-1) for p in system.model.prosody_stack.parameters() if p.grad is not None]),
                "prosody_head": torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in system.model.prosody_head.parameters()]),
                "acoustic_stack": torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in system.model.acoustic_stack.parameters()]),
            }

        g_spec_1 = grads_for(1.0, spec_only=True)
        g_spec_02 = grads_for(0.2, spec_only=True)
        # Forward identical -> acoustic_stack spectral gradients unchanged; prosody_stack scaled by exact 0.2
        self.assertTrue(torch.allclose(g_spec_02["acoustic_stack"], g_spec_1["acoustic_stack"], atol=1e-5))
        self.assertTrue(torch.allclose(g_spec_02["prosody_stack"], 0.2 * g_spec_1["prosody_stack"], atol=1e-5))
        self.assertEqual(float(g_spec_02["prosody_head"].norm().item()), 0.0)

        # Direct prosody supervision into prosody_stack & prosody_head is completely untouched by alpha
        g_pros_1 = grads_for(1.0, spec_only=False)
        g_pros_02 = grads_for(0.2, spec_only=False)
        self.assertTrue(torch.allclose(g_pros_02["prosody_stack"], g_pros_1["prosody_stack"], atol=1e-6))
        self.assertTrue(torch.allclose(g_pros_02["prosody_head"], g_pros_1["prosody_head"], atol=1e-6))

    def test_pitch_shape_loss_masking_and_anti_oscillation(self):
        t_len = 60
        x = torch.linspace(0, 1, t_len)
        target = (0.20 * torch.sin(2.0 * 3.14159 * x))[None]
        reliable = torch.zeros((1, t_len), dtype=torch.float32)
        reliable[:, 10:50] = 1.0
        energy = torch.full((1, t_len), 0.9, dtype=torch.float32)

        # 1. Exact match -> strictly zero
        self.assertAlmostEqual(float(pitch_shape_loss(target, target, reliable, energy=energy).item()), 0.0, places=7)

        # 2. Untrusted / unvoiced / boundary / low-energy perturbations must produce zero gradient and zero loss
        perturbed = target.clone()
        perturbed[:, :11] += 2.5   # unvoiced + left boundary edge (index 10)
        perturbed[:, 49:] -= 2.5   # right boundary edge (index 49) + unvoiced
        self.assertAlmostEqual(float(pitch_shape_loss(perturbed, target, reliable, energy=energy).item()), 0.0, places=7)

        # Low-energy tail inside voiced segment is also masked out
        low_e = energy.clone()
        low_e[:, 40:] = -1.2
        perturbed_low_e = target.clone()
        perturbed_low_e[:, 39:] += 1.5
        self.assertAlmostEqual(float(pitch_shape_loss(perturbed_low_e, target, reliable, energy=low_e).item()), 0.0, places=7)

        # 3. Flattened contour has strictly positive shape loss
        flat = torch.zeros_like(target)
        flat_loss = float(pitch_shape_loss(flat, target, reliable, energy=energy).item())
        self.assertGreater(flat_loss, 0.01)

        # 4. Artificial high-frequency oscillation with SAME variance as target is strictly penalized (not rewarded!)
        osc = target + 0.12 * torch.sin(torch.arange(t_len, dtype=torch.float32) * 2.1)[None]
        osc_loss = float(pitch_shape_loss(osc, target, reliable, energy=energy).item())
        self.assertGreater(osc_loss, 0.05)

        # 5. Short utterances (T < 4) stay finite and safe
        short = torch.zeros((1, 2))
        self.assertEqual(float(pitch_shape_loss(short, short, torch.ones((1, 2))).item()), 0.0)

    def test_train_cli_init_from_absolute_steps_and_p4_flags(self):
        cfg = _tiny_config()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            feat_dir = root / "features"
            ann_dir = root / "annotations"
            feat_dir.mkdir()
            ann_dir.mkdir()
            ann_fp = "p4-test-ann"
            fr_fp = frontend_fingerprint()
            rules_fp = frontend_rules_fingerprint()

            rows = []
            for idx in range(4):
                spk = cfg.speakers[idx % 2]
                t_frames = 36
                pt_rel = f"utt_{idx}.pt"
                torch.save({
                    "mel": torch.randn(t_frames, cfg.n_mels) * 0.5 - 5.0,
                    "prosody": torch.stack((
                        torch.linspace(-0.1, 0.1, t_frames),
                        torch.ones(t_frames),
                        torch.full((t_frames,), -2.2),
                    ), dim=-1),
                    "wave": torch.randn(t_frames * cfg.hop) * 0.05,
                }, feat_dir / pt_rel)
                toks = [SYMBOL_TO_ID["<bos>"], SYMBOL_TO_ID["b"], SYMBOL_TO_ID["ɔ̃"], SYMBOL_TO_ID["ʒ"], SYMBOL_TO_ID["u"], SYMBOL_TO_ID["ʁ"], SYMBOL_TO_ID["<eos>"]]
                durs = [2, 5, 8, 5, 8, 6, 2]
                rows.append({
                    "id": f"utt_{idx}",
                    "speaker": spk,
                    "text": f"Bonjour numéro {idx}",
                    "text_hash": f"h{idx}",
                    "path": pt_rel,
                    "frames": t_frames,
                    "tokens": toks,
                    "stress": [0] * len(toks),
                    "durations": durs,
                    "frontend": fr_fp,
                    "frontend_rules": rules_fp,
                    "acoustic": cfg.acoustic_fingerprint,
                    "annotation": ann_fp,
                })
            (ann_dir / "annotations.json").write_text(json.dumps({
                "complete": True,
                "fingerprint": ann_fp,
                "acoustic_root": str(feat_dir),
                "frontend_rules": rules_fp,
                "annotation_version": ANNOTATION_VERSION,
            }))
            (ann_dir / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows[:2]) + "\n")
            (ann_dir / "validation.jsonl").write_text("\n".join(json.dumps(r) for r in rows[2:]) + "\n")
            cfg_path = root / "cfg.json"
            cfg_path.write_text(json.dumps(cfg.to_dict()))

            # Base run: 0 -> 2
            base_out = root / "base"
            cmd_base = [
                sys.executable, "-m", "tiny_voca.train",
                "--train", str(ann_dir / "train.jsonl"),
                "--validation", str(ann_dir / "validation.jsonl"),
                "--output", str(base_out),
                "--config", str(cfg_path),
                "--device", "cpu",
                "--steps", "2",
                "--gan-start", "1",
                "--gan-ramp", "1",
                "--crop-samples", "2048",
                "--crops", "1",
                "--save-every", "2",
                "--log-every", "1",
                "--validation-size", "2",
                "--seed", "42",
            ]
            res0 = subprocess.run(cmd_base, capture_output=True, text=True)
            self.assertEqual(res0.returncode, 0, res0.stderr)

            # Branch GF: absolute --steps 4 from step 2 -> runs 2 steps, saves step=4
            gf_out = root / "branch_gf"
            cmd_gf = [
                sys.executable, "-m", "tiny_voca.train",
                "--train", str(ann_dir / "train.jsonl"),
                "--validation", str(ann_dir / "validation.jsonl"),
                "--output", str(gf_out),
                "--config", str(cfg_path),
                "--init-from", str(base_out / "last.pt"),
                "--device", "cpu",
                "--steps", "4",
                "--warmup", "1",
                "--gan-start", "1",
                "--gan-ramp", "1",
                "--crop-samples", "2048",
                "--crops", "1",
                "--prosody-grad-scale", "0.2",
                "--weights", '{"pitch_shape": 4.0}',
                "--save-every", "2",
                "--log-every", "1",
                "--validation-size", "2",
                "--seed", "42",
            ]
            res1 = subprocess.run(cmd_gf, capture_output=True, text=True)
            self.assertEqual(res1.returncode, 0, res1.stderr)
            st = torch.load(gf_out / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(st["step"], 4)
            self.assertEqual(st["recipe"]["prosody_grad_scale"], 0.2)
            self.assertEqual(st["recipe"]["weights"]["pitch_shape"], 4.0)
            self.assertTrue(st["recipe"]["branched_from"]["loaded"]["optimizer"])
            self.assertTrue(st["recipe"]["branched_from"]["loaded"]["critics"])
            metrics_lines = [json.loads(l) for l in (gf_out / "metrics.jsonl").read_text().splitlines() if l.strip()]
            self.assertEqual([m["step"] for m in metrics_lines], [3, 4])
            self.assertIn("pitch_shape", metrics_lines[-1])
            self.assertIn("validation_pitch_shape", metrics_lines[-1])


if __name__ == "__main__":
    unittest.main()
