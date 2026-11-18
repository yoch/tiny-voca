"""Validation tests for the E02 unseen French prosody dataset and evaluation harness."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from eval_e02_prosody import DEFAULT_SENTENCES, check_corpus_isolation, load_sentences

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


if __name__ == "__main__":
    unittest.main()
