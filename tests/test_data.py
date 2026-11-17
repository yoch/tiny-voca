from dataclasses import replace
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

from tiny_voca import text
from tiny_voca.annotate import load_acoustic_rows
from tiny_voca.config import Config
from tiny_voca.data import AnnotatedDataset, BucketSampler, collate
from tiny_voca.journal import append, recover
from tiny_voca.losses import SpectralLoss, mel_filterbank
from tiny_voca.train import balanced_indices, check_leakage
from helpers import small_config, write_annotations

# Fingerprint stored in every feature manifest written by the byte-token preparation
# on the real corpus (n4ze3m/tiny-voca @ 93dfa15b). Those files remain usable.
REAL_LEGACY_FEATURES = "8b1517b100ff83bbd732952b23880cef2d6e761e3f8e996aa0e9d5373e095ee1"


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_existing_acoustic_features_stay_compatible_while_tokens_are_versioned_separately(self):
        c = Config()
        self.assertEqual(Config(speakers=("F2", "M2")).legacy_feature_fingerprint, REAL_LEGACY_FEATURES)
        changed = replace(c, channels=128, dropout=0.2, acoustic_layers=3, cep_causal=64)
        self.assertEqual(c.acoustic_fingerprint, changed.acoustic_fingerprint)      # model changes reuse features
        self.assertNotEqual(c.fingerprint, changed.fingerprint)
        self.assertNotEqual(c.acoustic_fingerprint, replace(c, n_mels=40).acoustic_fingerprint)
        baseline = c.acoustic_fingerprint
        with patch.object(text, "FRONTEND_VERSION", 999):
            self.assertEqual(c.acoustic_fingerprint, baseline)                      # a front-end change does not touch acoustics

    def test_dataset_reads_annotations_and_ignores_stale_byte_tokens(self):
        c = small_config()
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_annotations(directory, c, "train")
            data = AnnotatedDataset(manifest, c)
            item = data[1]
            self.assertEqual(item["tokens"].tolist(), data.rows[1]["tokens"])        # not the [1, 60, 75, 2] stored in the file
            self.assertEqual(int(item["durations"].sum()), item["mel"].shape[0])
            self.assertEqual(item["wave"].numel(), item["mel"].shape[0] * c.hop)
            self.assertEqual(item["prosody"].shape, (item["mel"].shape[0], 3))
            # Pitch is continuous through unvoiced edges; energy is normalized.
            self.assertTrue(bool((item["prosody"][:3, 0] == item["prosody"][data.rows[1]["durations"][0], 0]).all()))
            self.assertAlmostEqual(float(item["prosody"][0, 2]), (-3.0 - c.energy_mean) / c.energy_scale, places=6)
            # Vowel frames are marked voiced when asked; the pitch trust mask is left alone.
            from tiny_voca.data import VOWEL_IDS
            stored = torch.load(Path(directory) / "features" / data.rows[1]["path"], weights_only=True)
            stored["prosody"][:, 1] = 0                                   # a tracker that heard no voicing at all
            torch.save(stored, Path(directory) / "features" / data.rows[1]["path"])
            plain = AnnotatedDataset(manifest, c, cache=False)[1]
            fixed = AnnotatedDataset(manifest, c, cache=False, voiced_vowels=True)[1]
            vowel = torch.repeat_interleave(torch.tensor([t in VOWEL_IDS for t in data.rows[1]["tokens"]]), torch.tensor(data.rows[1]["durations"]))
            self.assertEqual(float(plain["prosody"][:, 1].sum()), 0.0)
            self.assertTrue(bool((fixed["prosody"][:, 1].bool() == vowel).all()))
            self.assertGreater(int(vowel.sum()), 0)
            torch.testing.assert_close(fixed["pitch_trusted"], plain["pitch_trusted"])
            batch = collate([data[0], data[3]])
            self.assertEqual(batch["frame_lengths"].tolist(), [data.rows[0]["frames"], data.rows[3]["frames"]])
            self.assertEqual(batch["durations"].sum(1).tolist(), batch["frame_lengths"].tolist())

    def test_stale_or_inconsistent_annotations_are_rejected_explicitly(self):
        c = small_config()
        cases = [
            (dict(edit=lambda r: r.update(frontend="front-end-2-byte-tokens")), "different front end"),
            (dict(edit=lambda r: r.pop("durations")), "different front end"),
            (dict(edit=lambda r: r.update(acoustic="other")), "acoustic feature settings"),
            (dict(edit=lambda r: r.update(annotation="another-run")), "different annotation run"),
            (dict(edit=lambda r: r["durations"].__setitem__(4, r["durations"][4] - 1)), "do not cover the frames"),
            (dict(edit=lambda r: (r["durations"].__setitem__(1, 0), r["durations"].__setitem__(4, r["durations"][4] + 4))), "phoneme without frames"),
            (dict(edit=lambda r: r["tokens"].__setitem__(1, 200)), "invalid token"),
            (dict(complete=False), "incomplete"),
        ]
        for options, message in cases:
            with tempfile.TemporaryDirectory() as directory, self.subTest(message=message):
                manifest = write_annotations(directory, c, "train", **options)
                with self.assertRaisesRegex(ValueError, message):
                    AnnotatedDataset(manifest, c)
        with tempfile.TemporaryDirectory() as directory:
            # A manifest from the old preparation (no annotation layer next to it) is refused outright.
            legacy = Path(directory) / "train.jsonl"
            legacy.write_text(json.dumps({"id": "a", "features": REAL_LEGACY_FEATURES, "path": "a.pt"}) + "\n")
            with self.assertRaisesRegex(ValueError, "legacy byte-token manifests are stale"):
                AnnotatedDataset(legacy, c)
            # The annotation step itself accepts those feature files (audio only) and current ones.
            (Path(directory) / "train.preparation.json").write_text(json.dumps({"complete": True}))
            rows = [{"id": "a", "features": Config().legacy_feature_fingerprint, "speaker": "F1"}, {"id": "b", "acoustic": Config().acoustic_fingerprint, "speaker": "M1"}]
            legacy.write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(len(load_acoustic_rows(Path(directory), "train", Config())), 2)
            legacy.write_text(json.dumps({"id": "a", "features": "something-else", "speaker": "F1"}) + "\n")
            with self.assertRaisesRegex(ValueError, "compatible audio/DSP"):
                load_acoustic_rows(Path(directory), "train", Config())

    def test_unicode_separators_are_not_record_boundaries_and_journals_recover(self):
        c = small_config()
        with tempfile.TemporaryDirectory() as directory:
            manifest = write_annotations(directory, c, "train", edit=lambda r: r.update(text="one two three\u0085four"))
            data = AnnotatedDataset(manifest, c)
            self.assertEqual(len(data), 4)
            self.assertEqual(data.rows[0]["text"], "one two three\u0085four")
            journal = Path(directory) / "journal.jsonl"
            with journal.open("a", encoding="utf-8") as stream:
                append(stream, {"id": 1}, sync=False)
                append(stream, {"id": 2})
                stream.write('{"id": 3, "torn":')
            self.assertEqual([r["id"] for r in recover(journal)], [1, 2])
            self.assertTrue(journal.read_text().endswith("\n"))

    def test_bucket_sampler_respects_the_frame_budget_and_restores_exactly(self):
        lengths = [50, 60, 400, 410, 55, 900, 65, 420]
        sampler = BucketSampler(lengths, frame_budget=1000, max_batch=3, seed=5, pool=16)
        batches = [sampler.next() for _ in range(40)]
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            self.assertTrue(len(batch) == 1 or len(batch) * max(lengths[i] for i in batch) <= 1000)
        self.assertEqual({i for b in batches for i in b}, set(range(8)))
        state = json.loads(json.dumps({"pending": sampler.state()["pending"]}))
        generator = sampler.state()["generator"].clone()
        following = [sampler.next() for _ in range(25)]
        restored = BucketSampler(lengths, frame_budget=1000, max_batch=3, seed=999, pool=16)
        restored.load_state({"generator": generator, "pending": state["pending"]})
        self.assertEqual([restored.next() for _ in range(25)], following)

    def test_balanced_selection_and_leakage_report(self):
        rows = [{"speaker": "F1"}] * 8 + [{"speaker": "M1"}] * 8
        indices = balanced_indices(SimpleNamespace(rows=rows), 4)
        self.assertEqual([rows[i]["speaker"] for i in indices], ["F1", "M1", "F1", "M1"])
        train = SimpleNamespace(rows=[{"id": "a_F1", "text_hash": "same", "text": "Bonjour."}])
        validation = SimpleNamespace(rows=[{"id": "a_M1", "text_hash": "same", "text": "Bonjour."}])
        with self.assertRaisesRegex(ValueError, "a_M1.*a_F1"):
            check_leakage(train, validation)

    def test_spectral_loss_masks_padding_and_survives_silence(self):
        loss = SpectralLoss(24000)
        torch.manual_seed(1)
        target = 0.1 * torch.randn(2, 12000)
        predicted = (target + 0.05 * torch.randn(2, 12000)).requires_grad_()
        lengths = torch.tensor([12000, 7936])
        stft, mel = loss(predicted, target, lengths)
        (stft + mel).backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())
        self.assertEqual(float(predicted.grad[1, 9500:].abs().max()), 0.0)          # beyond the short utterance (plus one window)
        alone = loss(predicted[1:, :7936].detach(), target[1:, :7936], lengths[1:])
        padded = loss(torch.cat((predicted[1:, :7936].detach(), torch.zeros(1, 4064)), 1),
                      torch.cat((target[1:, :7936], torch.zeros(1, 4064)), 1), lengths[1:])
        torch.testing.assert_close(torch.stack(alone), torch.stack(padded), atol=1e-5, rtol=1e-5)
        same = loss(target, target, lengths)
        self.assertLess(float(same[0]), 1e-3)
        self.assertLess(float(same[1]), 1e-4)
        for amplitude in (0.0, 1e-7):
            output = (amplitude * torch.randn(1, 4096)).requires_grad_()
            stft, mel = loss(output, torch.zeros(1, 4096), torch.tensor([4096]))
            (stft + mel).backward()
            self.assertTrue(torch.isfinite(output.grad).all())
            self.assertTrue(math.isfinite(float(stft)) and float(stft) < 100)
        filters = mel_filterbank(24000, 1024, 80)
        self.assertEqual(filters.shape, (80, 513))
        self.assertTrue(bool((filters >= 0).all()) and bool((filters.sum(1) > 0).all()))

    @unittest.skipUnless(importlib.util.find_spec("librosa") and importlib.util.find_spec("pyarrow"), "requires preparation extras")
    def test_acoustic_preparation_interrupt_resume_provenance_and_bad_rows(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        import soundfile as sf
        from tiny_voca.prepare import extract, prepare_split
        c = Config()
        time = np.arange(12000) / 24000

        def audio(rate):
            stream = io.BytesIO()
            sf.write(stream, 0.1 * np.sin(2 * np.pi * 200 * time), rate, format="WAV")
            return {"bytes": stream.getvalue()}
        rows = [{"id": str(i), "text": "Bonjour.", "voice_id": "F1", "audio": audio(24000), "reference_audio": b"reference bytes",
                 "source_license": "test-synthetic"} for i in range(5)]
        rows[2]["text"] = None
        rows[3]["audio"] = None
        rows[4]["audio"] = audio(16000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.parquet"
            pq.write_table(pa.Table.from_pylist(rows), source)
            output = root / "features"
            output.mkdir()
            args = SimpleNamespace(output=output, inputs=[source], split="train", text_column="text", speaker_column="voice_id", resume=False)
            calls = {"count": 0}

            def interrupted(wave, config):
                calls["count"] += 1
                if calls["count"] == 2:
                    raise KeyboardInterrupt
                return extract(wave, config)
            with patch("tiny_voca.prepare.extract", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    prepare_split(args, c)
            committed = recover(output / "train.jsonl")
            self.assertEqual(len(committed), 1)
            self.assertNotIn("phonemes", committed[0])                # acoustic only: no tokens to go stale
            self.assertEqual(committed[0]["acoustic"], c.acoustic_fingerprint)
            self.assertEqual(committed[0]["provenance"]["reference_audio"]["binary_bytes"], 15)
            self.assertEqual(committed[0]["provenance"]["source_license"], "test-synthetic")
            first_path = output / committed[0]["path"]
            first_time = first_path.stat().st_mtime_ns
            with (output / "train.jsonl").open("ab") as stream:
                stream.write(b'{"interrupted":')
            args.resume = True
            with patch("tiny_voca.prepare.extract", wraps=extract) as dsp:
                prepare_split(args, c)
                self.assertEqual(dsp.call_count, 1)
            self.assertEqual(first_path.stat().st_mtime_ns, first_time)
            self.assertEqual(len(recover(output / "train.jsonl")), 2)
            rejected = recover(output / "train.rejected.jsonl")
            self.assertEqual(len(rejected), 3)
            self.assertTrue(any("24000" in r["reason"] for r in rejected))
            self.assertTrue(json.loads((output / "train.preparation.json").read_text())["complete"])
            stored = torch.load(first_path, weights_only=True)
            self.assertEqual(set(stored), {"wave", "mel", "prosody", "speaker"})
            voiced = stored["prosody"][:, 1].bool()
            self.assertGreater(int(voiced.sum()), 10)
            self.assertLess(float(stored["prosody"][voiced, 0].abs().median()), 0.1)   # 200 Hz -> log(F0/200) = 0
            self.assertEqual(len(load_acoustic_rows(output, "train", c)), 2)


if __name__ == "__main__":
    unittest.main()
