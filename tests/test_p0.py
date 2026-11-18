"""Unit tests for P0 prosody diagnostics, unbiased F0 extraction, oracle synthesis, and gradient paths."""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from tiny_voca.config import ANNOTATION_VERSION, Config
from tiny_voca.data import AnnotatedDataset
from tiny_voca.losses import Discriminators
from tiny_voca.model import TinyVoca, pulse_train
from tiny_voca.text import FRONTEND_RULES_VERSION, SYMBOL_TO_ID, frontend_fingerprint, frontend_rules_fingerprint
from tiny_voca.train import TrainingSystem, WEIGHTS

from p0_expressivity import (
    compute_rhythm_metrics,
    extract_unbiased_f0,
    measure_training_gradients,
    oracle_synthesize,
)


class P0ExpressivityTests(unittest.TestCase):
    def test_unbiased_f0_tracks_male_and_female_contours_and_rejects_noise(self):
        c = Config()
        frames = 100
        # 1) Male-range low glide (85 -> 130 Hz) + 30-frame quiet noise tail.
        f0_m = torch.linspace(85.0, 130.0, 70)[None]
        voiced_sig = pulse_train(f0_m, c.sample_rate, c.hop).numpy()[0] * 0.25
        noise_tail = np.random.default_rng(7).normal(0.0, 0.005, size=30 * c.hop).astype(np.float32)
        wave_m = np.concatenate((voiced_sig.astype(np.float32), noise_tail))

        res_m = extract_unbiased_f0(wave_m, c, frames=frames)
        interior = np.arange(5, 65)
        self.assertGreater(float(np.mean(res_m["voiced"][interior])), 0.90)
        err_st = 12.0 * np.abs(np.log2(res_m["f0_hz"][interior] / f0_m[0].numpy()[interior]))
        self.assertLess(float(np.median(err_st)), 0.15)
        self.assertLess(float(np.mean(res_m["voiced"][75:])), 0.10)

        # 2) Female-range contour (190 -> 260 Hz).
        f0_f = torch.linspace(190.0, 260.0, frames)[None]
        wave_f = (pulse_train(f0_f, c.sample_rate, c.hop).numpy()[0] * 0.25).astype(np.float32)
        res_f = extract_unbiased_f0(wave_f, c, frames=frames)
        inner_f = np.arange(5, frames - 5)
        err_f = 12.0 * np.abs(np.log2(res_f["f0_hz"][inner_f] / f0_f[0].numpy()[inner_f]))
        self.assertLess(float(np.median(err_f)), 0.15)

    def test_rhythm_metrics_expose_internal_errors_despite_equal_total_duration(self):
        c = Config()
        # <bos> a <sp> b c , d <eos>
        tokens = np.array([
            SYMBOL_TO_ID["<bos>"], SYMBOL_TO_ID["a"], SYMBOL_TO_ID["<sp>"], SYMBOL_TO_ID["b"],
            SYMBOL_TO_ID["o"], SYMBOL_TO_ID[","], SYMBOL_TO_ID["d"], SYMBOL_TO_ID["i"], SYMBOL_TO_ID["<eos>"],
        ])
        ref_d = np.array([10, 8, 12, 6, 14, 15, 7, 18, 10])
        # Exact identity.
        id_m = compute_rhythm_metrics(ref_d, ref_d, tokens, c)
        self.assertAlmostEqual(id_m["phone_mae_ms"], 0.0)
        self.assertAlmostEqual(id_m["phone_rel_dur_mae"], 0.0)
        self.assertEqual(id_m["pause_fp"], 0)
        self.assertEqual(id_m["pause_fn"], 0)

        # Pure uniform slowdown (2x): total doubles, relative phoneme durations stay identical!
        slow_m = compute_rhythm_metrics(ref_d, ref_d * 2, tokens, c)
        self.assertAlmostEqual(slow_m["total_dur_ratio"], 2.0)
        self.assertAlmostEqual(slow_m["phone_rel_dur_mae"], 0.0, places=6)

        # Compensating internal errors + missed comma pause + inserted <sp> -> total sum STILL 100!
        distorted = np.array([10, 16, 0, 12, 7, 25, 13, 7, 10])  # sum == 100 == sum(ref_d)
        self.assertEqual(int(distorted.sum()), int(ref_d.sum()))
        dis_m = compute_rhythm_metrics(ref_d, distorted, tokens, c)
        self.assertAlmostEqual(dis_m["total_dur_ratio"], 1.0)
        self.assertGreater(dis_m["phone_mae_ms"], 40.0)
        self.assertGreater(dis_m["phone_rel_dur_mae"], 0.4)
        self.assertEqual(dis_m["pause_fn"], 1)

    def test_oracle_synthesize_modes_and_consistent_f0_injection(self):
        torch.manual_seed(0)
        c = Config(channels=32, heads=2, feedforward=64, encoder_layers=2, prosody_layers=1, acoustic_layers=2, predictor_channels=32)
        model = TinyVoca(c).eval()
        with torch.no_grad():
            model.vocoder.harmonic.weight.normal_(0.0, 0.05)
            model.vocoder.noise.weight.normal_(0.0, 0.05)
        tokens = torch.tensor([SYMBOL_TO_ID["<bos>"], SYMBOL_TO_ID["a"], SYMBOL_TO_ID["l"], SYMBOL_TO_ID["o"], SYMBOL_TO_ID["<eos>"]], dtype=torch.long)
        stress = torch.zeros_like(tokens)
        durations = torch.tensor([4, 12, 8, 16, 4], dtype=torch.long)
        t = int(durations.sum())
        oracle_f0 = torch.linspace(-0.4, 0.35, t)
        oracle_v = torch.ones(t)
        oracle_e = torch.linspace(-1.0, 1.2, t)
        item = {
            "tokens": tokens,
            "stress": stress,
            "durations": durations,
            "speaker": 0,
            "prosody": torch.stack((oracle_f0, oracle_v, oracle_e), dim=-1),
        }
        row = {"id": "stub", "speaker": "F1", "text": "Allô."}

        seen = {}
        def cap_decode(_mod, args):
            seen["decode_cond"] = args[1][0].detach().clone()
        def cap_vocoder(_mod, args):
            seen["vocoder_f0"] = args[1][0].detach().clone()

        h1 = model.decode if False else model.acoustic_stack.register_forward_pre_hook(lambda m, a: None)
        h1.remove()
        # Hook prosody_projection to inspect exact condition tensor passed to decode, and vocoder for f0.
        p_hook = model.prosody_projection.register_forward_pre_hook(lambda _m, a: seen.__setitem__("decode_cond", a[0][0].detach().clone()))
        v_hook = model.vocoder.register_forward_pre_hook(lambda _m, a: seen.__setitem__("vocoder_f0", a[1][0].detach().clone()))
        try:
            wave_b, meta_b = oracle_synthesize(model, row, item, "B_oracle_dur", seed=123)
            self.assertEqual(len(wave_b), t * c.hop)

            wave_c, meta_c = oracle_synthesize(model, row, item, "C_oracle_dur_f0", seed=123)
            self.assertEqual(len(wave_c), t * c.hop)
            torch.testing.assert_close(seen["decode_cond"][:, 0], oracle_f0.clamp(-2.0, 1.5))
            torch.testing.assert_close(seen["vocoder_f0"], model.frequency(oracle_f0[None])[0])

            wave_d, meta_d = oracle_synthesize(model, row, item, "D_oracle_full", seed=123)
            self.assertEqual(len(wave_d), t * c.hop)
            torch.testing.assert_close(seen["decode_cond"], item["prosody"])
            torch.testing.assert_close(seen["vocoder_f0"], model.frequency(oracle_f0[None])[0])
            self.assertGreater(float(np.abs(wave_d - wave_c).max()), 1e-4)
        finally:
            p_hook.remove()
            v_hook.remove()

    def test_gradient_paths_match_p0_c_predictions(self):
        torch.manual_seed(0)
        c = Config(channels=32, heads=2, feedforward=64, encoder_layers=2, prosody_layers=1, acoustic_layers=2, predictor_channels=32)
        system = TrainingSystem(c)
        with torch.no_grad():
            system.model.vocoder.harmonic.weight.normal_(0.0, 0.05)
            system.model.vocoder.noise.weight.normal_(0.0, 0.05)
        critics = Discriminators()

        tokens = [SYMBOL_TO_ID["<bos>"], SYMBOL_TO_ID["b"], SYMBOL_TO_ID["o"], SYMBOL_TO_ID["n"], SYMBOL_TO_ID["<eos>"]]
        durations = [4, 10, 16, 10, 4]
        frames = sum(durations)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            feat_dir = tmp_dir / "feat"
            (feat_dir / "test").mkdir(parents=True)
            pt_rel = "test/u0.pt"
            torch.save({
                "wave": torch.randn(frames * c.hop) * 0.1,
                "mel": torch.randn(frames, c.n_mels),
                "prosody": torch.stack((
                    torch.zeros(frames),
                    torch.ones(frames),
                    torch.full((frames,), -4.0),
                ), dim=-1),
                "speaker": 0,
            }, feat_dir / pt_rel)

            ann_dir = tmp_dir / "ann"
            ann_dir.mkdir()
            (ann_dir / "annotations.json").write_text(json.dumps({
                "fingerprint": "annfp",
                "complete": True,
                "acoustic_root": str(feat_dir),
                "frontend_rules": frontend_rules_fingerprint(),
                "annotation_version": ANNOTATION_VERSION,
            }), encoding="utf-8")
            row = {
                "id": "u0_F1", "speaker": "F1", "text": "bon", "path": pt_rel,
                "frames": frames, "tokens": tokens, "stress": [0] * len(tokens), "durations": durations,
                "frontend": frontend_fingerprint(), "frontend_rules": frontend_rules_fingerprint(),
                "acoustic": c.acoustic_fingerprint, "annotation": "annfp",
            }
            (ann_dir / "test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            ds = AnnotatedDataset(ann_dir / "test.jsonl", c, cache=False)

            ckpt_path = tmp_dir / "last.pt"
            torch.save({
                "step": 100,
                "config": c.to_dict(),
                "system": system.state_dict(),
                "critics": critics.state_dict(),
                "recipe": {"weights": WEIGHTS, "crop_samples": 4096, "crops": 1, "feature_matching": 30.0, "gan_ratio": 0.7, "gan_scale_max": 2000.0},
            }, ckpt_path)

            probes = measure_training_gradients(ckpt_path, ds, indices=[0], seed=0)
            # Spectral-only and GAN-only cannot reach duration or prosody_head (prediction.detach()),
            # BUT they DO flow into prosody_stack via non-detached prosody_hidden!
            for k in ("spectral_only", "gan_only"):
                g = probes[k]["grad_norms"]
                self.assertEqual(g["duration"], 0.0, msg=k)
                self.assertEqual(g["prosody_head"], 0.0, msg=k)
                self.assertGreater(g["prosody_stack"], 0.0, msg=k)
                self.assertGreater(g["encoder"], 0.0, msg=k)

            # Prosody supervision reaches duration, prosody_head, prosody_stack, and encoder, not decoder/vocoder.
            gp = probes["prosody_only"]["grad_norms"]
            self.assertEqual(gp["vocoder"], 0.0)
            self.assertEqual(gp["acoustic_stack"], 0.0)
            self.assertEqual(gp["prosody_projection"], 0.0)
            self.assertGreater(gp["duration"], 0.0)
            self.assertGreater(gp["prosody_head"], 0.0)
            self.assertGreater(gp["prosody_stack"], 0.0)
            self.assertGreater(gp["encoder"], 0.0)


if __name__ == "__main__":
    unittest.main()
