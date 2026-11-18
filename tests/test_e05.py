"""Unit tests for E05 physical acoustic boundary probes."""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_e05_aligner import lf_hf_band_ratio_db, rms_db_profile


class E05ProbeTests(unittest.TestCase):
    def test_lf_hf_crossing_locates_synthetic_fricative_vowel_boundary_within_2ms(self):
        sr = 24000
        rng = np.random.default_rng(7)
        t_total = 0.400
        n = int(sr * t_total)
        split_ms = 200
        split_idx = int(sr * split_ms / 1000)

        # First 200 ms: high-pass / 5 kHz fricative energy; last 200 ms: 180 Hz voiced vowel + weak 500 Hz harmonic
        t_v = np.arange(n - split_idx) / sr
        fric = 0.08 * np.sin(2 * np.pi * 5500 * np.arange(split_idx) / sr) + 0.02 * rng.standard_normal(split_idx)
        vowel = 0.15 * np.sin(2 * np.pi * 180 * t_v) + 0.08 * np.sin(2 * np.pi * 360 * t_v)
        wave = np.concatenate((fric, vowel)).astype(np.float32)

        ratio = lf_hf_band_ratio_db(wave, sr=sr, win=256, hop=24)
        cross = [t for t in range(1, len(ratio)) if ratio[t - 1] < 0.0 <= ratio[t]]
        self.assertTrue(cross)
        nearest = min(cross, key=lambda t: abs(t - split_ms))
        self.assertLessEqual(abs(nearest - split_ms), 2)

    def test_rms_profile_detects_speech_bounds(self):
        sr = 24000
        silent = np.zeros(int(sr * 0.1), dtype=np.float32)
        speech = 0.2 * np.sin(2 * np.pi * 220 * np.arange(int(sr * 0.2)) / sr).astype(np.float32)
        wave = np.concatenate((silent, speech, silent))
        rms = rms_db_profile(wave, hop=64, win=256)
        loud = np.percentile(rms, 95)
        active = np.flatnonzero(rms >= loud - 20.0)
        start_ms = active[0] * 64 / sr * 1000.0
        end_ms = active[-1] * 64 / sr * 1000.0
        self.assertAlmostEqual(start_ms, 100.0, delta=8.0)
        self.assertAlmostEqual(end_ms, 300.0, delta=8.0)


if __name__ == "__main__":
    unittest.main()
