import itertools
import unittest
import numpy as np
import torch

from tiny_voca.align import AlignmentSettings, assemble, ctc_viterbi, frame_energy_db, silence_mask, spans_from_states, token_spans
from tiny_voca.annotate import boundary_offsets, gate, summarize_offsets
from tiny_voca.data import continuous_pitch
from tiny_voca.refine import banded_viterbi, frame_labels, symbol_class

SETTINGS = AlignmentSettings()
HOP, RATE = 256, 24000


def collapse(path, blank=0):
    out, previous = [], blank
    for label in path:
        if label != blank and label != previous:
            out.append(label)
        previous = label
    return out


class AlignmentTests(unittest.TestCase):
    def test_ctc_viterbi_matches_exhaustive_search_and_torchaudio(self):
        rng = np.random.default_rng(3)
        for labels in ([1, 2], [1, 1, 2], [2, 1, 2]):
            log_probs = np.log(rng.dirichlet(np.ones(3), size=7)).astype(np.float32)
            best, best_path = -np.inf, None
            for path in itertools.product(range(3), repeat=7):
                if collapse(path) == labels:
                    score = sum(log_probs[t, s] for t, s in enumerate(path))
                    if score > best:
                        best, best_path = score, path
            states = ctc_viterbi(log_probs, np.array(labels))
            extended = [0 if s % 2 == 0 else labels[s // 2] for s in states]
            self.assertEqual(tuple(extended), best_path)
            # The production path (torchaudio when installed) gives the same emission spans.
            self.assertEqual(token_spans(log_probs, labels), spans_from_states(states, len(labels)))
        with self.assertRaises(ValueError):
            token_spans(np.zeros((2, 3), dtype=np.float32), [1, 1])     # a repeat needs a blank between

    def test_durations_follow_emission_onsets_and_cover_every_frame(self):
        symbols = ("<bos>", "b", "ɔ̃", "<sp>", "ʒ", "u", ",", "<sp>", "t", "u", ".", "<eos>")
        # Emission frames (20 ms each) of the six phones; a long gap before "t" holds a pause.
        spans = [(10, 10), (15, 16), (30, 30), (34, 34), (70, 70), (75, 75)]
        frames = 190
        silent = np.zeros(frames, dtype=bool)
        silent[:16] = True          # leading silence
        silent[90:120] = True       # pause after the comma
        silent[170:] = True         # trailing silence
        durations = assemble(symbols, spans, frames, silent, HOP, RATE, SETTINGS)
        self.assertEqual(int(durations.sum()), frames)
        self.assertEqual(len(durations), len(symbols))
        self.assertTrue(all(durations[i] >= 1 for i in (1, 2, 4, 5, 8, 9)))
        self.assertEqual(durations[0], 16)                       # BOS takes the leading silence
        self.assertEqual(durations[3], 0)                        # word boundary without a pause
        self.assertEqual(durations[6], 30)                       # the comma carries the pause, not the <sp> after it
        self.assertEqual(durations[7], 0)
        self.assertEqual(durations[10], 0)                       # final punctuation: EOS owns trailing silence
        self.assertEqual(durations[11], 20)
        # "ɔ̃" starts 19 ms before its first emission frame (15 * 20 ms) and lasts to the next onset.
        start = int(np.ceil((15 * 0.02 + SETTINGS.onset_offset_seconds) * RATE / HOP))
        self.assertEqual(int(durations[:2].sum()), start)
        self.assertEqual(int(durations[:3].sum()), int(np.ceil((30 * 0.02 + SETTINGS.onset_offset_seconds) * RATE / HOP)))

    def test_short_silences_stay_with_phones_and_bad_input_is_rejected(self):
        symbols = ("<bos>", "a", "t", "<sp>", "e", "t", "<eos>")
        spans = [(5, 5), (12, 12), (22, 22), (30, 30)]
        silent = np.zeros(80, dtype=bool)
        silent[28:32] = True                                     # a 4-frame stop closure is not a pause
        durations = assemble(symbols, spans, 80, silent, HOP, RATE, SETTINGS)
        self.assertEqual(durations[3], 0)
        self.assertEqual(int(durations.sum()), 80)
        with self.assertRaises(ValueError):
            assemble(symbols, spans[:-1], 80, silent, HOP, RATE, SETTINGS)
        with self.assertRaises(ValueError):                      # more phones than frames can hold
            assemble(symbols, [(0, 0), (1, 1), (2, 2), (3, 3)], 3, np.zeros(3, dtype=bool), HOP, RATE, SETTINGS)

    def test_energy_silence_detection(self):
        wave = np.zeros(HOP * 100)
        wave[HOP * 30:HOP * 70] = 0.3 * np.sin(np.arange(HOP * 40) * 0.2)
        wave += 1e-4 * np.random.default_rng(0).standard_normal(len(wave))
        level = frame_energy_db(wave, HOP)
        silent, threshold = silence_mask(level, SETTINGS)
        self.assertEqual(len(level), 100)
        self.assertTrue(silent[:28].all() and silent[72:].all())
        self.assertFalse(silent[31:69].any())
        self.assertLess(threshold, -30)

    def test_banded_refinement_moves_boundaries_only_within_the_band(self):
        truth = [6, 9, 5]
        classes = np.repeat([0, 1, 2], truth)
        scores = np.full((20, 3), -5.0)
        scores[np.arange(20), classes] = 0.0
        coarse_starts, coarse_ends = np.array([0, 8, 13]), np.array([8, 13, 20])      # first boundary 2 late, second 2 early
        self.assertEqual(banded_viterbi(scores, coarse_starts, coarse_ends, band=3).tolist(), truth)
        limited = banded_viterbi(scores, coarse_starts, coarse_ends, band=1)
        self.assertEqual(limited.tolist(), [7, 7, 6])                                # moves by at most one frame
        self.assertEqual(int(limited.sum()), 20)
        self.assertEqual(frame_labels(("<bos>", "t", "<sp>", "a"), [2, 1, 0, 3]).tolist(),
                         [symbol_class("<bos>")] * 2 + [symbol_class("t")] + [symbol_class("a")] * 3)

    def test_voicing_probe_measures_boundary_offsets(self):
        symbols = ("<bos>", "s", "i", "t", "<eos>")
        durations = [2, 6, 10, 5, 2]
        voiced = np.zeros(25, dtype=bool)
        voiced[7:19] = True                      # vowel is voiced one frame early and ends one frame late
        found = boundary_offsets(symbols, durations, voiced)
        self.assertEqual(found, {"vowel_to_voiceless": [1], "fricative_to_vowel": [-1]})
        summary = summarize_offsets({"vowel_to_voiceless": [1, 0, None], "fricative_to_vowel": []})
        self.assertEqual(summary["vowel_to_voiceless"]["boundaries"], 3)
        self.assertAlmostEqual(summary["vowel_to_voiceless"]["flip_found"], 0.6667, places=3)
        self.assertEqual(summary["vowel_to_voiceless"]["within_1_frame"], 1.0)

    def test_quality_gate_names_every_reason(self):
        limits = {"max_weak_phone_fraction": 0.25, "min_token_probability": 0.5, "max_phone_frames": 70,
                  "min_vowel_voiced": 0.5, "min_phones_per_second": 4.0, "max_phones_per_second": 30.0}
        good = {"unknown": [], "qc": {"weak_phone_fraction": 0.02, "token_probability_mean": 0.85, "max_phone_frames": 30,
                                      "vowel_voiced": 0.93, "phones_per_second": 12.5}}
        self.assertEqual(gate(good, limits), [])
        bad = {"unknown": ["ʘ"], "qc": {"weak_phone_fraction": 0.4, "token_probability_mean": 0.3, "max_phone_frames": 90,
                                        "vowel_voiced": 0.2, "phones_per_second": 40.0}}
        self.assertEqual(gate(bad, limits), ["unknown_phoneme", "weak_alignment", "low_confidence", "overlong_phone",
                                             "vowels_not_voiced", "implausible_rate"])

    def test_pitch_is_made_continuous_and_octave_errors_are_distrusted(self):
        pitch = torch.tensor([0.0, 0.0, 0.1, 0.1, 0.8, 0.1, 0.1, 0.0, 0.0, 0.3, 0.3, 0.3, 0.0])
        voiced = torch.tensor([0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0.0])
        curve, trusted = continuous_pitch(pitch, voiced)
        self.assertEqual(trusted.tolist(), [0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0])
        torch.testing.assert_close(curve[:7], torch.full((7,), 0.1))             # outlier replaced, start extended
        torch.testing.assert_close(curve[6:10], torch.tensor([0.1, 0.1 + 0.2 / 3, 0.1 + 0.4 / 3, 0.3]))
        self.assertAlmostEqual(float(curve[-1]), 0.3, places=6)
        flat, none = continuous_pitch(torch.zeros(5), torch.zeros(5), default=-0.5)
        self.assertEqual(flat.tolist(), [-0.5] * 5)
        self.assertEqual(float(none.sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
