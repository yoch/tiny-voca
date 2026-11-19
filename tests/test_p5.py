"""Unit tests for P5 reproducibility guards, stratified text-level bootstrap,
decoupled F0 synthesis, French question classification, and independent M1 pitch estimators."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from tiny_voca.config import Config
from tiny_voca.model import TinyVoca
from tiny_voca.text import SYMBOL_TO_ID

from audit_p5_m1_pitch import estimate_autocorr_amdf_frame, estimate_cepstrum_frame, _bandpass
from eval_p4_ablation import load_exclusion_sets, stratified_text_bootstrap_ci
from eval_p5_f0_causal import classify_french_question, synthesize_decoupled_f0
from p0_expressivity import oracle_synthesize


class P5AuditTests(unittest.TestCase):
    def test_load_exclusion_sets_fails_when_missing_for_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "does_not_exist.json"
            with self.assertRaises(FileNotFoundError):
                load_exclusion_sets([missing])

            empty_file = Path(tmp) / "empty.json"
            empty_file.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_exclusion_sets([empty_file])

            valid = Path(tmp) / "p0.json"
            valid.write_text(json.dumps({"selected_ids": ["abc_F1"], "selected_text_hashes": ["h1"]}), encoding="utf-8")
            ids, hashes = load_exclusion_sets([valid])
            self.assertEqual(ids, {"abc_F1"})
            self.assertEqual(hashes, {"h1"})

    def test_stratified_text_bootstrap_keeps_f1_m1_paired_per_text(self):
        # Construct 4 texts (2 categories x 2 texts), where F1 has +1.0 and M1 has -1.0
        # for EVERY text. Text-level mean delta is identically 0.0 on EVERY text,
        # so keeping F1/M1 paired per text MUST produce exact [0.0, 0.0] CI!
        # (An unpaired recording-level bootstrap would produce a wide non-zero CI.)
        rows_a, rows_b = [], []
        for ci, cat in enumerate(("question", "plain_assert")):
            for ti in range(2):
                th = f"hash_{ci}_{ti}"
                for spk, delta in (("F1", 1.0), ("M1", -1.0)):
                    rid = f"{th}_{spk}"
                    rows_a.append({"id": rid, "text_hash": th, "speaker": spk, "category": cat, "m": delta})
                    rows_b.append({"id": rid, "text_hash": th, "speaker": spk, "category": cat, "m": 0.0})
        res = stratified_text_bootstrap_ci({"A": rows_a, "B": rows_b}, {"A": 1.0, "B": -1.0}, "m", n_boot=500, seed=42)
        self.assertEqual(res["n_texts"], 4)
        self.assertEqual(res["n_recordings"], 8)
        self.assertAlmostEqual(res["mean_delta"], 0.0, places=6)
        self.assertAlmostEqual(res["ci95_low"], 0.0, places=6)
        self.assertAlmostEqual(res["ci95_high"], 0.0, places=6)
        self.assertFalse(res["significant"])

    def test_decoupled_f0_matches_p0_oracle_b_and_c_bit_exact(self):
        cfg = Config(
            channels=32,
            heads=4,
            feedforward=64,
            encoder_layers=2,
            prosody_layers=2,
            acoustic_layers=2,
            predictor_channels=32,
            speakers=("F1", "M1"),
            max_frames=200,
        )
        torch.manual_seed(7)
        model = TinyVoca(cfg).eval()
        with torch.no_grad():
            model.vocoder.harmonic.weight.normal_(0.0, 0.02)
            model.vocoder.noise.weight.normal_(0.0, 0.02)

        tokens = torch.tensor([SYMBOL_TO_ID["<bos>"], SYMBOL_TO_ID["a"], SYMBOL_TO_ID["t"], SYMBOL_TO_ID["<eos>"]], dtype=torch.long)
        durations = torch.tensor([4, 12, 8, 4], dtype=torch.long)
        t_frames = int(durations.sum())
        row = {"id": "toy_F1", "speaker": "F1"}
        item = {
            "tokens": tokens,
            "stress": torch.zeros_like(tokens),
            "durations": durations,
            "speaker": 0,
            "prosody": torch.stack((
                torch.linspace(-0.2, 0.3, t_frames),
                torch.ones(t_frames),
                torch.full((t_frames,), 0.5),
            ), dim=-1),
        }
        p0_b, _ = oracle_synthesize(model, row, item, "B_oracle_dur", seed=1234)
        p0_c, _ = oracle_synthesize(model, row, item, "C_oracle_dur_f0", seed=1234)
        ref_f0 = item["prosody"][:, 0][None]
        dur_b = durations[None]
        m00, _ = synthesize_decoupled_f0(model, item, dur_b, dec_log_f0=None, osc_log_f0=None, seed=1234)
        m11, _ = synthesize_decoupled_f0(model, item, dur_b, dec_log_f0=ref_f0, osc_log_f0=ref_f0, seed=1234)
        m01, _ = synthesize_decoupled_f0(model, item, dur_b, dec_log_f0=None, osc_log_f0=ref_f0, seed=1234)
        self.assertEqual(float(np.max(np.abs(m00 - p0_b))), 0.0)
        self.assertEqual(float(np.max(np.abs(m11 - p0_c))), 0.0)
        # Changing osc F0 while holding dec F0 fixed must change the waveform.
        self.assertGreater(float(np.max(np.abs(m01 - m00))), 1e-4)

    def test_french_question_classifier_totale_vs_partielle(self):
        self.assertEqual(classify_french_question("Vous venez ?"), "totale")
        self.assertEqual(classify_french_question("Est-ce que tu viens demain ?"), "totale")
        self.assertEqual(classify_french_question("Peux-tu me rappeler si Karim préfère venir lundi ?"), "totale")
        self.assertEqual(classify_french_question("Pourquoi tu pars maintenant ?"), "partielle")
        self.assertEqual(classify_french_question("Alors qu'est-ce que vous devenez, Madame Marquet ?"), "partielle")
        self.assertEqual(classify_french_question("Où avez-vous laissé les clés ?"), "partielle")
        self.assertEqual(classify_french_question("Comment fait-on pour ouvrir cette porte ?"), "partielle")
        self.assertEqual(classify_french_question("Combien coûte ce billet ?"), "partielle")

    def test_independent_m1_pitch_estimators_voiced_and_unvoiced(self):
        sr = 24000
        dur = 0.25
        n = int(sr * dur)
        t = np.arange(n) / sr
        # Synthetic low male harmonic voice at 108 Hz (1st-6th harmonics).
        f0_true = 108.0
        voiced_wave = sum((0.6 ** k) * np.sin(2.0 * math.pi * (k + 1) * f0_true * t) for k in range(6))
        voiced_wave = voiced_wave / (np.max(np.abs(voiced_wave)) + 1e-9) * 0.5
        voiced_bp = _bandpass(voiced_wave, sr, 50.0, 450.0)
        center = n // 2
        peak_db = 20.0 * math.log10(float(np.sqrt(np.mean(voiced_wave ** 2))))

        ac = estimate_autocorr_amdf_frame(voiced_wave, voiced_bp, center, sr=sr, peak_db=peak_db)
        cep = estimate_cepstrum_frame(voiced_wave, center, sr=sr, peak_db=peak_db)
        self.assertTrue(ac["voiced"])
        self.assertTrue(cep["voiced"])
        self.assertLess(abs(ac["f0_hz"] - f0_true), 2.5)
        self.assertLess(abs(cep["f0_hz"] - f0_true), 2.5)

        # Synthetic unvoiced fricative (high-passed noise > 2.5 kHz).
        rng = np.random.default_rng(7)
        noise = rng.standard_normal(n)
        spec = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        spec[freqs < 2500.0] = 0.0
        fric = np.fft.irfft(spec, n=n)
        fric = fric / (np.max(np.abs(fric)) + 1e-9) * 0.3
        fric_bp = _bandpass(fric, sr, 50.0, 450.0)
        ac_u = estimate_autocorr_amdf_frame(fric, fric_bp, center, sr=sr, peak_db=peak_db)
        cep_u = estimate_cepstrum_frame(fric, center, sr=sr, peak_db=peak_db)
        self.assertFalse(ac_u["voiced"])
        self.assertFalse(cep_u["voiced"])


if __name__ == "__main__":
    unittest.main()
