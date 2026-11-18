"""Small CPU tests for the E05 measurements and the alternate pause policy."""

import unittest
from dataclasses import replace

import numpy as np
from alignment_probe import (
    BASELINE,
    HOP,
    RATE,
    AlignmentSettings,
    assemble,
    compare,
    edit_path,
    pause_mask,
    protected_spans,
)


class ProbeTests(unittest.TestCase):
    def test_fragmented_quiet_frames_do_not_create_a_single_pause(self):
        spans = [(4, 4), (20, 20)]
        silent = np.zeros(60, dtype=bool)
        silent[12:16] = True
        silent[26:30] = True
        symbols = ("<bos>", "a", "<sp>", "s", "<eos>")
        legacy = assemble(
            symbols, spans, len(silent), silent, HOP, RATE, AlignmentSettings()
        )
        self.assertEqual(legacy[2], 18)
        masked = pause_mask(silent, spans, "contiguous", 7)
        candidate = assemble(
            symbols, spans, len(silent), masked, HOP, RATE, AlignmentSettings()
        )
        self.assertEqual(candidate[2], 0)
        self.assertEqual(candidate.sum(), len(silent))
        self.assertGreaterEqual(int(candidate[[1, 3]].min()), 1)

    def test_standalone_edge_island_and_one_frame_bridge(self):
        spans = [(4, 4), (20, 20)]
        silent = np.zeros(60, dtype=bool)
        first, last = protected_spans(spans, len(silent))
        silent[:2] = True
        silent[first - 3 : first - 1] = True
        silent[12:16] = silent[17:21] = True
        silent[last + 2 : last + 4] = silent[-2:] = True
        mask = pause_mask(silent, spans, "bridge_one", 7)
        expected = np.zeros_like(silent)
        expected[:2] = expected[12:21] = expected[-2:] = True
        np.testing.assert_array_equal(mask, expected)

    def test_exact_correspondence_ignores_insertion(self):
        edit = edit_path(["a", "s", "i"], ["a", "z", "s", "i"])
        self.assertEqual(
            edit,
            [("equal", 0, 0), ("insert", None, 1), ("equal", 1, 2), ("equal", 2, 3)],
        )

    def test_reference_pause_counts_and_boundary_error(self):
        seconds = HOP / RATE
        ref = [
            dict(phone="a", start=seconds, end=3 * seconds),
            dict(phone="s", start=4 * seconds, end=5 * seconds),
        ]
        row = compare(
            ("<bos>", "a", "<sp>", "s", "<eos>"),
            np.array([1, 2, 1, 1, 1]),
            ref,
            [(0, 0), (1, 1)],
        )
        self.assertEqual(row["pause_tp"], 3)
        self.assertEqual(row["pause_fp"] + row["pause_fn"], 0)
        self.assertEqual([x["onset_ms"] for x in row["boundaries"]], [0, 0])

    def test_offset_never_drops_anchored_phones(self):
        spans = [(i, i) for i in range(5)]
        symbols = ("<bos>", "ə", "p", "s", "ɛ̃", "ʁ", "<eos>")
        for offset in (BASELINE[1], -0.05, 0.01):
            d = assemble(
                symbols,
                spans,
                20,
                np.zeros(20, dtype=bool),
                HOP,
                RATE,
                replace(AlignmentSettings(), onset_offset_seconds=offset),
            )
            self.assertEqual(d.sum(), 20)
            self.assertGreaterEqual(int(d[1:-1].min()), 1)


if __name__ == "__main__":
    unittest.main()
