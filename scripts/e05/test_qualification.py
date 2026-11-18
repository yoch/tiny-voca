import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from alignment_probe import BASELINE, FIXED_CANDIDATE, HOP, RATE, digest, probe
from prepare_rhapsodie import apply_reviewed_text
from prepare_target import prepare
from review_disagreements import validate_review
from scipy.io import wavfile

from tiny_voca import annotate
from tiny_voca.config import ANNOTATION_VERSION, Config
from tiny_voca.data import AnnotatedDataset
from tiny_voca.text import (
    PHONE_SET,
    UNK,
    frontend_fingerprint,
    frontend_rules_fingerprint,
)


def target_index(root):
    clips = []
    for speaker, text, frequency in (("F1", "Pas ici", 220), ("M1", "Pas dehors", 140)):
        source = root / f"{speaker}.wav"
        wave = np.sin(np.arange(RATE) * (2 * np.pi * frequency / RATE)).astype(
            np.float32
        )
        wavfile.write(source, RATE, wave)
        clips.append(
            {
                "id": speaker + "-01",
                "speaker": speaker,
                "recording": speaker + "-session",
                "text": text,
                "wav": source.name,
                "sha256": digest(source),
                "oracle": ["<bos>", "p", "a", "<sp>", "i", "<eos>"],
                "reference": [
                    {"phone": "p", "start": 0.05, "end": 0.15},
                    {"phone": "a", "start": 0.15, "end": 0.35},
                    {"phone": "i", "start": 0.5, "end": 0.8},
                ],
            }
        )
    index = {
        "dataset_revision": "synthetic-test",
        "voice_revision": "test-voice",
        "annotation_revision": "test-phones",
        "clips": clips,
    }
    labels = root / "target.json"
    labels.write_text(json.dumps(index))
    exclusion = root / "train.jsonl"
    exclusion.write_text(json.dumps({"text": "Apprendre à lire."}) + "\n")
    return labels, exclusion


class QualificationTests(unittest.TestCase):
    def test_f1_m1_measurement_is_fixed_and_reports_per_voice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, exclusion = target_index(root)
            corpus = root / "holdout"
            prepare(index, root, [exclusion], corpus)
            model = root / "cached.onnx"
            model.write_bytes(b"inference fixture")
            vocab = root / "vocab.json"
            vocabulary = {s: i + 1 for i, s in enumerate(sorted(PHONE_SET | {UNK}))}
            vocabulary["<pad>"] = 0
            vocab.write_text(json.dumps(vocabulary))
            cache = root / "cache"
            cache.mkdir()
            for row in json.loads((corpus / "manifest.json").read_text()):
                key = row["fixture_sha256"] + "-" + digest(model)
                np.savez(
                    cache / (key + ".npz"), log_probs=np.zeros((48, len(vocabulary)))
                )
            result_path = root / "result.json"
            args = SimpleNamespace(
                corpus=corpus,
                target=True,
                model=model,
                vocab=vocab,
                threads=1,
                cache=cache,
                output=result_path,
            )
            with patch(
                "alignment_probe.ort.InferenceSession", side_effect=AssertionError
            ):
                probe(args)
            result = json.loads(result_path.read_text())
            self.assertIsNone(result["best_dev"])
            self.assertEqual(
                set(result["summaries"]),
                {
                    f"holdout/{mode}/{policy}/{offset:+.3f}"
                    for mode in ("oracle", "g2p")
                    for policy, offset in (BASELINE, FIXED_CANDIDATE)
                },
            )
            f1 = result["fixed_comparison"]["holdout/oracle/F1"]["baseline"]
            self.assertEqual(f1["onset"]["count"], 3)
            self.assertEqual(f1["by_group"]["stops"]["end"]["count"], 1)
            self.assertIn("recall", f1["pause_internal"])

    def test_target_rejects_training_text_overlap_and_replaced_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index, exclusion = target_index(root)
            exclusion.write_text(json.dumps({"text": " PAS ici ! "}) + "\n")
            with self.assertRaisesRegex(ValueError, "overlapping text"):
                prepare(index, root, [exclusion], root / "holdout")
            exclusion.write_text(json.dumps({"text": "Autre texte."}) + "\n")
            (root / "F1.wav").write_bytes(b"replaced file")
            with self.assertRaisesRegex(ValueError, "WAV hash mismatch"):
                prepare(index, root, [exclusion], root / "holdout")
            payload = json.loads(index.read_text())
            payload["clips"][0]["sha256"] = digest(root / "F1.wav")
            index.write_text(json.dumps(payload))
            with self.assertRaises((ValueError, RuntimeError)):
                prepare(index, root, [exclusion], root / "holdout")

    def test_text_edits_and_disagreement_reviews_reject_stale_inputs(self):
        original = "pantalons ensuite"
        edits = {
            "clip": {
                "text_sha256": hashlib.sha256(original.encode()).hexdigest(),
                "before": original,
                "after": "pantalons. Ensuite",
            }
        }
        self.assertEqual(
            apply_reviewed_text(original, "clip", edits), "pantalons. Ensuite"
        )
        with self.assertRaisesRegex(ValueError, "Stale"):
            apply_reviewed_text("autres pantalons ensuite", "clip", edits)
        case = {"id": "clip:1", "edit": ["insert", "∅", "z"], "wav_anchor_ms": 42}
        validate_review([case], [{**case, "category": "text_context"}])
        with self.assertRaisesRegex(ValueError, "Stale review"):
            validate_review([case], [{**case, "wav_anchor_ms": 43}])

    def test_full_versioned_annotations_load_with_e01(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features"
            features.mkdir()
            config = Config()
            frames = 50
            for split, text, voice in zip(
                annotate.SPLITS, ("Bonjour", "Salut", "Bonsoir"), ("F1", "M1", "F1")
            ):
                path = split + ".pt"
                torch.save(
                    {
                        "mel": torch.zeros(frames, config.n_mels),
                        "prosody": torch.ones(frames, 3),
                        "wave": torch.zeros(frames * HOP),
                    },
                    features / path,
                )
                row = {
                    "id": split,
                    "speaker": voice,
                    "text": text,
                    "text_hash": annotate.text_key(text),
                    "frames": frames,
                    "path": path,
                    "acoustic": config.acoustic_fingerprint,
                }
                (features / (split + ".jsonl")).write_text(json.dumps(row) + "\n")
                (features / (split + ".preparation.json")).write_text(
                    '{"complete":true}'
                )

            def align_stub(_, phonemes, wave, voiced, config, settings):
                durations = [3 if s in PHONE_SET else 0 for s in phonemes.symbols]
                durations[0] = 2 if settings.onset_offset_seconds == BASELINE[1] else 1
                durations[-1] = len(voiced) - sum(durations)
                return durations, {
                    "weak_phone_fraction": 0,
                    "token_probability_mean": 1,
                    "max_phone_frames": 3,
                    "vowel_voiced": 1,
                    "phones_per_second": 10,
                }

            def annotate_cli(profile, output, extra=()):
                command = [
                    "tiny-voca-annotate",
                    "--device",
                    "cpu",
                    "--features",
                    str(features),
                    "--output",
                    str(output),
                    "--dataset-revision",
                    "test-e01",
                    "--alignment-profile",
                    profile,
                    *extra,
                ]
                with (
                    patch("sys.argv", command),
                    patch("builtins.print"),
                    patch.object(
                        annotate,
                        "PhoneRecognizer",
                        return_value=SimpleNamespace(revision="test"),
                    ),
                    patch.object(annotate, "align_utterance", side_effect=align_stub),
                ):
                    annotate.main()

            default_settings, provenance = annotate.annotation_settings(7, None)
            self.assertEqual(default_settings.onset_offset_seconds, BASELINE[1])
            self.assertEqual(provenance, {})
            before, after = root / "before", root / "after"
            annotate_cli("e01-refresh", before)
            annotate_cli("e05-onset-candidate", after)
            baseline = AnnotatedDataset(before / "train.jsonl", config)
            candidate = AnnotatedDataset(after / "train.jsonl", config)
            self.assertNotEqual(baseline.annotation, candidate.annotation)
            self.assertEqual(baseline.rows[0]["tokens"], candidate.rows[0]["tokens"])
            self.assertEqual(candidate.rows[0]["frontend"], frontend_fingerprint())
            self.assertEqual(
                candidate.rows[0]["frontend_rules"], frontend_rules_fingerprint()
            )
            metadata = json.loads((after / "annotations.json").read_text())
            self.assertEqual(
                metadata["identity"]["annotation_version"], ANNOTATION_VERSION
            )
            self.assertNotEqual(
                baseline.rows[0]["durations"], candidate.rows[0]["durations"]
            )
            self.assertEqual(candidate[0]["durations"].sum().item(), frames)
            with self.assertRaisesRegex(ValueError, "stale"):
                annotate_cli("e05-onset-candidate", before, ("--resume",))


if __name__ == "__main__":
    unittest.main()
