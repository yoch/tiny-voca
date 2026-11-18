"""Validation tests for the E02 unseen French prosody dataset and evaluation harness."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_e02_prosody
from eval_e02_prosody import DEFAULT_SENTENCES, _stratified_ids, check_corpus_isolation, load_sentences

HAVE_ESPEAK = shutil.which("espeak-ng") is not None or os.path.exists("/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")


class E02ProtocolTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_ESPEAK, "requires espeak-ng or libespeak-ng")
    def test_e02_unseen_sentences_are_clean_and_disjoint(self):
        items = load_sentences(DEFAULT_SENTENCES)
        self.assertGreaterEqual(len(items), 50)
        excludes = [
            ROOT / "data" / "scripts_fr_4k" / "train.jsonl",
            ROOT / "data" / "scripts_fr_4k" / "validation.jsonl",
            ROOT / "data" / "scripts_fr_4k" / "test.jsonl",
            ROOT / "tests" / "french_phoneme_benchmark.json",
        ]
        info = check_corpus_isolation(items, excludes)
        self.assertEqual(info["overlap_ids"], [])
        self.assertEqual(info["unknowns"], [])
        self.assertEqual(len(info["categories"]), 8)

    @unittest.skipUnless(HAVE_ESPEAK, "requires espeak-ng or libespeak-ng")
    def test_e02_cli_dry_run_and_readiness_gate(self):
        script = ROOT / "scripts" / "eval_e02_prosody.py"
        env = {**os.environ, "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"}
        dry = subprocess.run([sys.executable, str(script), "--dry-run"], capture_output=True, text=True, check=True, env=env)
        payload = json.loads(dry.stdout)
        self.assertEqual(payload["status"], "dry_run_ok")
        self.assertEqual(payload["isolation"]["sentence_count"], 64)

        with tempfile.TemporaryDirectory() as tmp:
            rep = Path(tmp) / "report.json"
            rep.write_text(json.dumps({"summary": {"novel": {"all": {"generated_wer": 0.2109}}}}), encoding="utf-8")
            blocked = subprocess.run(
                [sys.executable, str(script), "--report", str(rep), "--out-dir", str(Path(tmp) / "out")],
                capture_output=True, text=True, env=env,
            )
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn("E02 BLOCKED", blocked.stderr)

    def test_stratified_ids_picks_balanced_subset(self):
        synthetic = [
            {"id": "a1", "category": "cat_a"},
            {"id": "a2", "category": "cat_a"},
            {"id": "a3", "category": "cat_a"},
            {"id": "b1", "category": "cat_b"},
            {"id": "b2", "category": "cat_b"},
        ]
        self.assertEqual(_stratified_ids(synthetic, 2), {"a1", "a2", "b1", "b2"})

        items = load_sentences(DEFAULT_SENTENCES)
        picked = _stratified_ids(items, 4)
        self.assertEqual(len(picked), 32)
        by_cat = {}
        for row in items:
            if row["id"] in picked:
                by_cat[row["category"]] = by_cat.get(row["category"], 0) + 1
        self.assertEqual(len(by_cat), 8)
        self.assertTrue(all(count == 4 for count in by_cat.values()))

    @unittest.skipUnless(HAVE_ESPEAK, "requires espeak-ng or libespeak-ng")
    def test_e02_cli_with_asr_flags_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            rep = tmp_dir / "report.json"
            rep.write_text(json.dumps({"summary": {"novel": {"all": {"generated_wer": 0.1165}}}}), encoding="utf-8")
            ckpt = tmp_dir / "inference.pt"
            ckpt.write_bytes(b"stub")
            out_dir = tmp_dir / "out"
            argv = [
                "eval_e02_prosody.py",
                "--checkpoint", str(ckpt),
                "--report", str(rep),
                "--out-dir", str(out_dir),
                "--limit", "16",
                "--with-asr",
                "--asr-model", "small",
                "--asr-threads", "2",
                "--asr-per-category", "4",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(eval_e02_prosody, "evaluate_variants", return_value={"baseline": {}}) as mocked_eval,
                mock.patch.object(eval_e02_prosody, "sha256_file", return_value="deadbeef"),
                mock.patch("builtins.print"),
            ):
                eval_e02_prosody.main()

            mocked_eval.assert_called_once()
            _, kwargs = mocked_eval.call_args
            self.assertTrue(kwargs["with_asr"])
            self.assertEqual(kwargs["asr_model"], "small")
            self.assertEqual(kwargs["asr_threads"], 2)
            self.assertEqual(kwargs["asr_per_category"], 4)
            self.assertEqual(kwargs["limit"], 16)
            saved = json.loads((out_dir / "e02_results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["asr_model"], "faster-whisper small")
            self.assertEqual(saved["checkpoint_sha256"], "deadbeef")


if __name__ == "__main__":
    unittest.main()
